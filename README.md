# 💳 Credit Card Fraud Detection
# 📌 Project Overview

This project focuses on building a machine learning model to detect fraudulent credit card transactions. Using the well-known Kaggle Credit Card Fraud Detection dataset, we trained and compared multiple models to identify risky transactions.

Since fraud cases are highly imbalanced (fraudulent = ~0.17% of all transactions), we used data preprocessing, scaling, and resampling techniques to improve recall while maintaining high accuracy.

The final model was deployed using Streamlit to provide an interactive demo.

# 🎯 Objectives

Preprocess the credit card transaction dataset.

Handle imbalanced data (fraud vs. non-fraud).

Compare machine learning models: Logistic Regression, Random Forest, XGBoost, LightGBM.

Optimize for recall, since missing fraud cases is costlier than false positives.

Deploy an interactive Streamlit web app for prediction.

# 📊 Dataset

Source: Kaggle - Credit Card Fraud Detection

Transactions made by European cardholders in September 2013.

Contains 284,807 transactions with 492 fraud cases (0.172%).

Features:

Time: Seconds elapsed between this transaction and the first transaction.

Amount: Transaction amount.

V1–V28: PCA-transformed features to protect sensitive data.

Class: Target variable (0 = Non-Fraud, 1 = Fraud).

# ⚙️ Methodology

Data Preprocessing

StandardScaler applied to Time and Amount.

PCA features kept as is.

Handling Imbalanced Data

Tried SMOTE oversampling and class weighting.

# Model Training

Logistic Regression

Random Forest

XGBoost

LightGBM

# Model Evaluation

Metrics: Precision, Recall, F1-score, ROC-AUC

Chose LightGBM as the final model due to:

Highest Recall (87%)

High Accuracy (>99%)

Good balance of Precision and Recall.

# 🚀 Results
     Model	        Precision Recall	F1-score	Accuracy

Logistic Regression   0.75	   0.62	   0.68     97.5%

Random Forest        	0.91	   0.83	   0.87	    99.2%

XGBoost              	0.93	   0.85    0.89   	99.4%

LightGBM (Final)    	0.94	   0.87	   0.90   	99.5%

✅ LightGBM selected as the final model.

# 📂 Project Structure
├── fraud_detection_model.ipynb   # Jupyter Notebook with full training & evaluation

├── best_credit_model.pkl         # Trained model

├── scaler.pkl                    # Scaler for preprocessing

├── requirements.txt              # Dependencies
