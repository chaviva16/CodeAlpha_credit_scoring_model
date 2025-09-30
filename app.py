import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ========================
# Load model & scaler
# ========================
model = joblib.load("best_credit_model.pkl")
scaler = joblib.load("scaler.pkl")

# ========================
# App Config
# ========================
st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="wide")
st.title("💳 Credit Card Fraud Detection")
st.write(
    "This app demonstrates a **machine learning model** trained on the "
    "**Kaggle Credit Card Fraud Dataset** to detect potentially fraudulent transactions."
)

# ========================
# Sidebar Navigation
# ========================
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", ["📊 Dataset Insights", "🤖 Fraud Prediction Demo"])

# ========================
# 1. Dataset Insights
# ========================
if section == "📊 Dataset Insights":
    st.header("📊 Dataset Insights")

    # Load dataset (only sample first 20k rows for faster demo)
    @st.cache_data
    def load_data():
        data = pd.read_csv("creditcard.csv")
        return data

    data = load_data()

    st.write("Here’s a quick look at the dataset:")
    st.dataframe(data.head())

    # Class distribution
    st.subheader("🔎 Class Distribution (Fraud vs Legitimate)")
    class_counts = data["Class"].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis", ax=ax)
    ax.set_xticklabels(["Legitimate (0)", "Fraudulent (1)"])
    ax.set_ylabel("Number of Transactions")
    st.pyplot(fig)

    # Amount distribution
    st.subheader("💵 Transaction Amount Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data[data["Class"] == 0]["Amount"], bins=50, color="green", label="Legit", ax=ax, stat="density", kde=True)
    sns.histplot(data[data["Class"] == 1]["Amount"], bins=50, color="red", label="Fraud", ax=ax, stat="density", kde=True)
    ax.legend()
    ax.set_xlim(0, 500)  # zoom in for clarity
    st.pyplot(fig)

    # Fraud over time
    st.subheader("⏱️ Fraud Frequency Over Time")
    fraud_over_time = data[data["Class"] == 1].groupby("Time").size()
    fig, ax = plt.subplots()
    fraud_over_time.plot(kind="line", ax=ax, color="red")
    ax.set_xlabel("Time (seconds since first transaction)")
    ax.set_ylabel("Number of Frauds")
    st.pyplot(fig)

    st.info("Fraudulent transactions are rare and often clustered in time with smaller amounts.")

# ========================
# 2. Fraud Prediction Demo
# ========================
elif section == "🤖 Fraud Prediction Demo":
    st.header("🤖 Fraud Prediction Demo")
    st.write("Enter transaction details to check if it may be **fraudulent**.")

    # User inputs
    amount = st.number_input("💵 Transaction Amount ($)", min_value=0.0, step=10.0)
    time = st.number_input("⏰ Transaction Time (seconds since first transaction)", min_value=0.0, step=100.0)

    # Fill PCA features with 0 for demo
    v_features = [0.0] * 28
    features = [time] + v_features + [amount]
    features_scaled = scaler.transform([features])

    if st.button("🚀 Predict Transaction"):
        prediction = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][1]

        if prediction == 1:
            st.error(f"🚨 Fraudulent Transaction Detected! (Fraud Probability: {prob:.2%})")
        else:
            st.success(f"✅ Legitimate Transaction (Fraud Probability: {prob:.2%})")

    