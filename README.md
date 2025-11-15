# 🛒 Retail Demand Forecasting System (End-to-End)

**Time-Series Forecasting • MLOps • FastAPI • Streamlit • XGBoost • Prophet • TFT**

This project implements a production-ready retail demand forecasting system inspired by real forecasting pipelines used by Amazon, Walmart, and Shopify. It includes a full ML workflow, multiple forecasting models, API deployment, and an interactive Streamlit dashboard.

---

## 🚀 Key Features

- Full ML pipeline: data → preprocessing → feature engineering → modeling → evaluation  
- Multiple forecasting models: Prophet, XGBoost, LightGBM, TFT  
- FastAPI backend with a unified `/predict` endpoint  
- Streamlit dashboard for interactive forecasting  
- Docker-ready architecture  
- Cached TFT forecasts for instant deep-learning inference  
- Model artifacts stored as `.joblib`, `.json`, `.pkl`  

---

## 📂 Project Structure

```
Demand_Forecasting_System_for_Retail/
│
├── app.py                                          # Streamlit UI
├── main.py                                         # FastAPI backend
├── prophet_refit.py                                # Prophet refitting script
│
├── Demand_Forecasting_System_for_Retail.ipynb     # Full EDA + modeling notebook
│
├── *.csv                                           # train/test/stores/oil/holidays/transactions
├── *.joblib                                        # preprocessors, LightGBM, TFT meta, Prophet meta
├── *.json                                          # XGBoost booster, Prophet serialized model
├── *.pkl                                           # Prophet CmdStanPy models
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📊 Models Implemented

### 1. **Prophet**
- Captures daily/weekly/yearly seasonality  
- One model per store–family combination  
- Saved as `.json` and `.pkl`

**Example API request:**
```json
{
  "model": "prophet",
  "records": [{"ds": "2025-11-20"}]
}
```

### 2. **XGBoost**
Uses rich, engineered features:

- Date features (day, month, year, weekend, holiday)
- Promotions and transactions
- Oil prices
- Lag features
- Rolling-window statistics
- Store and family-level aggregates

**Artifacts:**
- `xgb_preprocessor_v1.joblib`
- `xgb_booster_v1.json`
- `xgb_meta_v1.joblib`

### 3. **LightGBM**
Fast and efficient gradient boosting with categorical support.

### 4. **Temporal Fusion Transformer (TFT)**
Deep learning forecasting model. Predictions cached as CSV files:

```
tft_forecast_store1_GROCERY_I.csv
```

API loads forecasts instantly without GPU.

---

## 🧠 ML Pipeline Overview

### ✔ Data Engineering
**Merged:**
- `train.csv`
- `stores.csv`
- `oil.csv`
- `transactions.csv`
- `holidays_events.csv`

**Created:**
- Date features (day, month, week, quarter, year)
- Cyclical encodings
- Lag features (7, 14, 28 days)
- Rolling means
- Promotions & holiday flags
- Store/family-level aggregates

### ✔ Model Training
For each model:
1. Train/validation split
2. Hyperparameter tuning
3. Evaluation using RMSE & MAPE
4. Export artifacts for API

---

## ⚡ FastAPI Backend

**Run API:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8090 --reload
```

### Endpoints

#### Health Check
```bash
GET /health
```

#### Predict
```bash
POST /predict
```

**Example XGBoost request:**
```json
{
  "model": "xgboost",
  "records": [{
    "store_nbr": 1,
    "family": "GROCERY I",
    "day": 18,
    "month": 11,
    "year": 2025,
    "onpromotion": 10,
    "state": "Pichincha"
  }]
}
```

---

## 🎨 Streamlit Dashboard

**Run Dashboard:**
```bash
streamlit run app.py
```

**Includes:**
- Prophet forecasting UI
- XGBoost detailed form
- Line charts of predictions

---

## 🐳 Docker Deployment

**Build API:**
```bash
docker build -t retail-api .
```

**Run API container:**
```bash
docker run -p 8090:8090 retail-api
```

**Run Streamlit container:**
```bash
docker run -p 8501:8501 retail-ui
```

---

## 📈 Evaluation Summary

| Model     | RMSE | MAPE | Notes                        |
|-----------|------|------|------------------------------|
| Prophet   | 211.007600    | 8.927337e+06    | Good seasonal fit            |
| XGBoost   | 499.227516   | 8.357167e+08    | Best accuracy overall        |
| LightGBM  | 795.312289    | 3.475552e+07    | Fast training                |
| TFT       | 443.717682    | 1.022024e+06    | Deep learning, cached inference |

---

## 🛠 Tech Stack

- Python 3.10+
- Pandas, NumPy, Scikit-Learn
- XGBoost, LightGBM
- Prophet, CmdStanPy
- FastAPI
- Streamlit
- Docker
- Joblib / Pickle / JSON

---

## 🧪 How to Run End-to-End

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Start API:
```bash
uvicorn main:app --host 0.0.0.0 --port 8090 --reload
```

### 3. Launch Streamlit:
```bash
streamlit run app.py
```

### 4. Open in browser:
- **Streamlit UI** → http://localhost:8501
- **FastAPI Docs** → http://localhost:8090/docs

---

## 📌 Future Improvements

- Prefect/Airflow pipeline orchestration
- Model monitoring with EvidentlyAI
- AWS deployment (EC2 + S3 + Lambda)
- Online learning + auto-retraining
- Feature Store (Feast)

---

## 📝 License

MIT License.

---

## ✨ Author

**Nagarjunan Saravanan**  
MS Computer Science – Binghamton University  
Aspiring ML Engineer & Computer Vision Engineer
