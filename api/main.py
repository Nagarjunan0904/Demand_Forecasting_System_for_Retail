# --- main.py ---
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import traceback
from prophet.serialize import model_from_json

# Prophet import (optional)
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Retail Demand Forecasting API", version="1.0")

# --- ✅ CORS configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic schemas ---
class PredictRequest(BaseModel):
    model: str
    records: List[Dict[str, Any]]

class PredictResponse(BaseModel):
    model: str
    predictions: List[float]

class SchemaResponse(BaseModel):
    features: List[str]
    categorical: List[str]
    numeric: List[str]

# --- Helper to load model safely ---
def _load_model(model_name: str):
    name = model_name.lower()
    if name == "lightgbm":
        path = BASE_DIR.parent / "lgbm_retail_v1.joblib"
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"Model file for {name} not found at {path}")
        return joblib.load(path)

    elif name == "xgboost":
        pre_path = BASE_DIR.parent / "xgb_preprocessor_v1.joblib"
        model_path = BASE_DIR.parent / "xgb_booster_v1.json"
        meta_path = BASE_DIR.parent / "xgb_meta_v1.joblib"

        if not (pre_path.exists() and model_path.exists()):
            raise HTTPException(status_code=400, detail=f"Missing XGBoost artifacts in {BASE_DIR.parent}")
        return {"pre_path": pre_path, "model_path": model_path, "meta_path": meta_path}

    elif name == "prophet":
        # dynamic find of prophet file
        json_files = list(BASE_DIR.parent.glob("prophet_store*_GROCERY_I_v1.json"))
        meta_files = list(BASE_DIR.parent.glob("prophet_store*_GROCERY_I_meta_v1.joblib"))
        if not json_files or not meta_files:
            raise HTTPException(status_code=400, detail="No valid Prophet model found (.json or .pkl).")
        return {"model_path": json_files[0], "meta_path": meta_files[0]}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'")
    
# ============================================================
#  📦  Load Cached TFT Forecasts
# ============================================================
TFT_CACHE = {}

TFT_DIR = BASE_DIR.parent
for f in TFT_DIR.glob("tft_forecast_*.csv"):
    try:
        df_tft_cache = pd.read_csv(f)
        for (store, fam), grp in df_tft_cache.groupby(["store_nbr", "family"]):
            TFT_CACHE[(int(store), str(fam).strip())] = grp["prediction"].tolist()
        print(f"✅ Loaded cached TFT results from {f.name}")
    except Exception as e:
        print(f"⚠️ Failed to load TFT cache {f}: {e}")

if not TFT_CACHE:
    print("⚠️ No TFT forecast cache files found — /predict?tft will return default message")


# --- /health endpoint ---
@app.get("/health")
def health():
    return {"status": "ok", "available_models": ["lightgbm", "xgboost", "prophet"]}

# --- /predict endpoint ---
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model_name = req.model.lower()
    try:
        if model_name == "lightgbm":
            bundle = _load_model(model_name)
            model = bundle["model"]
            feature_order = bundle.get("feature_order")
            cat_cols = bundle.get("categorical_cols", [])
            df = pd.DataFrame(req.records)
            for c in cat_cols:
                if c in df.columns:
                    df[c] = df[c].astype("category")
            X = df[feature_order]
            preds = model.predict(X)
            return PredictResponse(model="lightgbm", predictions=preds.tolist())

        elif model_name == "xgboost":
            import xgboost as xgb
            bundle = _load_model(model_name)
            pre_bundle = joblib.load(bundle["pre_path"])
            pre = pre_bundle["pre"]
            feature_order = pre_bundle["feature_order"]
            df = pd.DataFrame(req.records)[feature_order]
            X_trans = pre.transform(df)
            booster = xgb.Booster()
            booster.load_model(bundle["model_path"])
            dmat = xgb.DMatrix(X_trans)
            preds = booster.predict(dmat)
            return PredictResponse(model="xgboost", predictions=preds.tolist())

        elif model_name == "prophet":
            if Prophet is None:
                raise HTTPException(status_code=500, detail="Prophet not installed.")
            bundle = _load_model("prophet")
            meta = joblib.load(bundle["meta_path"])
            model_path = bundle["model_path"]
            try:
                with open(model_path, "r") as f:
                    model = model_from_json(f.read())
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to deserialize Prophet model: {e}")
            df = pd.DataFrame(req.records)
            if "date" in df.columns:
                df = df.rename(columns={"date": "ds"})
            if "is_holiday" not in df.columns:
                df["is_holiday"] = 0
            if "onpromotion" not in df.columns:
                df["onpromotion"] = 0.0
            future = df[["ds", "is_holiday", "onpromotion"]]
            fcst = model.predict(future)
            preds = fcst["yhat"].tolist()
            return PredictResponse(model="prophet", predictions=preds)
        
        elif model_name == "tft":
            try:
                # Extract parameters
                record = req.records[0]  # ✅ Fix: comes from req, not 'records'
                store_nbr = record.get("store_nbr")
                family_k = record.get("family")

                if not store_nbr or not family_k:
                    raise ValueError("Both 'store_nbr' and 'family' must be provided for TFT model")

                # Normalize naming (spaces → underscores)
                safe_family = str(family_k).replace(" ", "_")

                # Define path relative to API folder
                tft_forecast_path = Path(__file__).resolve().parent.parent / f"tft_forecast_store{store_nbr}_{safe_family}.csv"

                # Debug print
                print(f"🔍 [TFT] Looking for cached forecast at: {tft_forecast_path}")

                # Check if file exists
                if not tft_forecast_path.exists():
                    raise FileNotFoundError(f"No cached TFT forecast found for store={store_nbr}, family='{family_k}'.")

                # Load cached forecast CSV
                df_forecast = pd.read_csv(tft_forecast_path)

                # --- Filter for the requested series (robust to underscores/spaces) ---
                df_forecast["family"] = (
                    df_forecast["family"]
                    .astype(str)
                    .str.strip()
                    .str.replace("_", " ", regex=False)
                )
                family_k_norm = str(family_k).strip().replace("_", " ")

                mask = (
                    (df_forecast["store_nbr"].astype(str) == str(store_nbr))
                    & (df_forecast["family"] == family_k_norm)
                )

                df_selected = df_forecast.loc[mask].copy()

                if df_selected.empty:
                    raise ValueError(
                        f"No matching predictions found in cached TFT file for store={store_nbr}, family='{family_k_norm}'."
    )


                # Take last 7 predictions (or all if fewer)
                preds = df_selected["prediction"].tail(7).tolist()

                # --- ✅ Inverse transform with GroupNormalizer (if available) ---
                try:
                    import torch
                    from pytorch_forecasting.data import GroupNormalizer

                    meta_path = Path(__file__).resolve().parent.parent / "tft_retail_v1.meta.joblib"
                    if meta_path.exists():
                        meta = joblib.load(meta_path)
                        dataset_params = meta.get("dataset_parameters", {})

                        if "target_normalizer" in dataset_params:
                            normalizer = dataset_params["target_normalizer"]

                            # If it’s a GroupNormalizer, we can inverse it directly
                            if isinstance(normalizer, GroupNormalizer):
                                preds_tensor = torch.tensor(preds)
                                preds_inv = normalizer.inverse_transform(
                                    preds_tensor, group=df_selected["series_id"].iloc[0]
                                )
                                preds = preds_inv.detach().cpu().numpy().tolist()
                            else:
                                # Fallback: exp() for softplus-like transforms
                                preds = (np.exp(preds) - 1).tolist()
                        else:
                            preds = (np.exp(preds) - 1).tolist()
                except Exception as e:
                    print(f"⚠️ Inverse transform skipped due to: {e}")

                return {"model": "tft", "predictions": preds}



            except Exception as e:
                raise HTTPException(status_code=400, detail=f"tft prediction failed: {e}")


    except Exception as e:
        tb = traceback.format_exc()
        print(f"❌ {model_name} prediction failed:\n{tb}")
        raise HTTPException(status_code=400, detail=f"{model_name} prediction failed: {e}")

@app.get("/tft_cache")
def list_tft_cache():
    return {"available_tft_series": [f"store={s}, family={f}" for s, f in TFT_CACHE.keys()]}


# --- global exception handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    tb = traceback.format_exc()
    print("⛔ Internal Server Error:\n", tb)
    return JSONResponse(status_code=500, content={"error": str(exc), "trace": tb})
