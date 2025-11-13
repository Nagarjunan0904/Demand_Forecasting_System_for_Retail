import streamlit as st
import pandas as pd
import requests
from datetime import date, timedelta

# ------------------------------
# App Configuration
# ------------------------------
st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")
st.title("🛍️ Retail Demand Forecasting Dashboard")

API_URL = "http://localhost:8090/predict"  # ✅ your running container port

# ------------------------------
# Model Selection
# ------------------------------
model = st.selectbox("Select model", ["prophet", "xgboost"])

if model == "prophet":
    st.subheader("📅 Prophet Forecast")
    start_date = st.date_input("Start Date", date.today())
    days = st.number_input("Days to Forecast", min_value=1, max_value=30, value=7)

    if st.button("Generate Prophet Forecast"):
        future_dates = [
            {"ds": str(start_date + timedelta(days=i))} for i in range(days)
        ]
        payload = {"model": "prophet", "records": future_dates}

        try:
            res = requests.post(API_URL, json=payload)
            if res.status_code == 200:
                preds = pd.DataFrame(res.json()["predictions"])
                st.success("✅ Prophet Forecast Generated Successfully!")
                st.line_chart(preds.set_index("ds")["yhat"])
            else:
                st.error(res.text)
        except Exception as e:
            st.error(str(e))

# ------------------------------
# XGBoost Forecast
# ------------------------------
elif model == "xgboost":
    st.subheader("📊 XGBoost Prediction")

    store = st.number_input("Store Number", min_value=1, value=1)
    family = st.text_input("Product Family", "GROCERY I")
    day = st.number_input("Day of Month", min_value=1, max_value=31, value=18)
    month = st.number_input("Month", min_value=1, max_value=12, value=11)
    year = st.number_input("Year", min_value=2020, max_value=2030, value=2025)

    if st.button("Generate XGBoost Forecast"):
        payload = {
            "model": "xgboost",
            "records": [
                {
                    "store_nbr": store,
                    "family": family,
                    "state": "Pichincha",
                    "type": "A",
                    "cluster": 13,
                    "onpromotion": 10,
                    "is_holiday": 0,
                    "weekday": 2,
                    "is_weekend": 0,
                    "day": day,
                    "weekofyear": 46,
                    "is_month_start": 0,
                    "is_month_end": 0,
                    "month": month,
                    "quarter": 4,
                    "year": year,
                    "transactions": 210,
                    "oil_price": 55.43,
                    "holiday_type": "Work Day",
                    "avg_sales_per_family": 1250.75,
                    "avg_sales_per_store_family": 1189.24,
                    "sales_vs_store_avg": 1.02,
                    "sales_vs_family_avg": 0.97,
                    "avg_sales_per_store": 1450.31
                }
            ],
        }

        try:
            res = requests.post(API_URL, json=payload)
            if res.status_code == 200:
                preds = res.json()["predictions"]
                st.success("✅ XGBoost Forecast Generated Successfully!")
                st.metric("Predicted Sales", round(preds[0]["yhat"], 2))
            else:
                st.error(res.text)
        except Exception as e:
            st.error(str(e))
