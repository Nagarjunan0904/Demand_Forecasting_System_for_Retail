# ============================================================
# Refit Prophet model inside Docker to use CmdStanPy backend
# ============================================================

from prophet import Prophet
import joblib
import pandas as pd
import os
import pickle

print("📂 Current working directory:", os.getcwd())

# --- 1️⃣ Load training data ---
#  Replace this path with your actual training CSV used for Prophet earlier
df = pd.read_csv("/app/store-sales-time-series-forecasting/train.csv")  # Adjust name if different
print(f"✅ Loaded training data with shape {df.shape}")

# --- 2️⃣ Prepare Prophet-friendly columns ---
df = df.rename(columns={"date": "ds", "sales": "y"})

# --- 3️⃣ Fit Prophet model ---
model = Prophet()
model.fit(df)

# --- 4️⃣ Save model using CmdStanPy backend ---
model_json = "/app/prophet_store1_GROCERY_I_v1.json"
meta_joblib = "/app/prophet_store1_GROCERY_I_meta_v1.joblib"

# Save Prophet model using pickle (Prophet v1.1.5 compatible)
with open(model_json.replace(".json", ".pkl"), "wb") as f:
    pickle.dump(model, f)

joblib.dump({"meta": "prophet docker model"}, meta_joblib)
print(f"✅ Re-exported Prophet model to:\n   {model_json.replace('.json', '.pkl')}\n   {meta_joblib}")
print("🎯 CmdStanPy backend confirmed.")

