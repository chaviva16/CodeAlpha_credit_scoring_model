import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("best_credit_model.pkl")
scaler = joblib.load("scaler.pkl")

# App title
st.title("💳 Creditworthiness Prediction App")
st.write("This app predicts whether an individual is **creditworthy** based on their financial transaction data.")

# Sidebar input fields
st.sidebar.header("Input Features")
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, step=10.0)
time = st.sidebar.number_input("Transaction Time", min_value=0.0, step=100.0)

# Simulated PCA features (V1–V28) – in real use, they’d come from preprocessing
v_features = []
for i in range(1, 29):
    v = st.sidebar.slider(f"V{i}", -10.0, 10.0, 0.0)
    v_features.append(v)

# Collect all features
features = [time] + v_features + [amount]

# Scale input
features_scaled = scaler.transform([features])

# Predict button
if st.button("Predict Creditworthiness"):
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        st.error(f"🚨 Risky Transaction Detected! (Fraud Probability: {prob:.2%})")
    else:
        st.success(f"✅ Creditworthy Transaction (Fraud Probability: {prob:.2%})")
