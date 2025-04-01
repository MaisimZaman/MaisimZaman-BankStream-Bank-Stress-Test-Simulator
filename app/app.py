import streamlit as st
import pandas as pd
import sys
import os
import joblib

# Add the root of the project (one level up from /app) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import risk metric functions
from src.risk_metrics import expected_loss, monte_carlo_var

# === 🔧 App Config ===
st.set_page_config(page_title="BankStressSim", layout="wide")
st.title("🏦 Bank Stress Testing Simulator")

# === 📉 Sidebar: Macro Stress Inputs ===
st.sidebar.header("📉 Macro Stress Inputs")
interest_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 10.0, 5.0, 0.25)
unemployment_rate = st.sidebar.slider("Unemployment Rate (%)", 0.0, 15.0, 6.0, 0.5)
inflation_rate = st.sidebar.slider("Inflation Rate (%)", 0.0, 10.0, 2.5, 0.5)
capital_requirement = st.sidebar.slider("Capital Requirement (%)", 5.0, 15.0, 10.0, 0.5)

# === 📂 Load Pretrained Model ===
model_path = os.path.join("models", "loan_default_model.pkl")
if not os.path.exists(model_path):
    st.error("❌ Trained model not found. Please make sure 'loan_default_model.pkl' exists in /models.")
    st.stop()

model = joblib.load(model_path)
feature_names = model.get_booster().feature_names

# === 📊 Upload Portfolio CSV ===
st.header("📊 Loan Portfolio Simulation")
uploaded_file = st.file_uploader("Upload your loan dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_encoded = df.copy()

    # Binary conversions
    binary_cols = ["HasMortgage", "HasDependents", "HasCoSigner"]
    for col in binary_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map({"Yes": 1, "No": 0})

    # Feature Engineering
    df_encoded["loan_to_income"] = df_encoded["LoanAmount"] / (df_encoded["Income"] + 1e-6)
    df_encoded["credit_utilization_score"] = df_encoded["DTIRatio"] * df_encoded["InterestRate"]
    df_encoded["employment_years"] = df_encoded["MonthsEmployed"] / 12
    df_encoded["credit_income_ratio"] = df_encoded["CreditScore"] / (df_encoded["Income"] + 1e-6)

    df_encoded["loan_to_income"] = df_encoded["loan_to_income"].clip(0, 10)
    df_encoded["credit_utilization_score"] = df_encoded["credit_utilization_score"].clip(0, 10)
    df_encoded["credit_income_ratio"] = df_encoded["credit_income_ratio"].clip(0, 1)

    # One-hot encoding
    cat_cols = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"]
    df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)

    # Align features
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_names]

    # === 🔮 Predict Default Probabilities ===
    df["pd"] = model.predict_proba(df_encoded)[:, 1]

    # Apply macroeconomic stress multiplier
    stress_multiplier = 1 + ((interest_rate - 3) * 0.05 + (unemployment_rate - 5) * 0.07)
    df["pd"] = (df["pd"] * stress_multiplier).clip(0, 1)

    # === 📈 Risk Metrics ===
    el = expected_loss(df, pd_col="pd", lgd=0.45)
    var_95 = monte_carlo_var(df, confidence=0.95, lgd=0.45)
    total_loan = df["LoanAmount"].sum()
    capital_held = total_loan * (capital_requirement / 100)
    car = 100 * capital_held / (el + 1e-6)

    # === 📊 Display Metrics ===
    st.subheader("📈 Portfolio Risk Metrics")
    st.metric("Expected Loss", f"${el:,.2f}")
    st.metric("Value at Risk (95%)", f"${var_95:,.2f}")
    st.metric("Capital Adequacy Ratio", f"{car:.2f}%", delta=f"{car - capital_requirement:.2f}%", delta_color="inverse")

    # Optional: show sample
    with st.expander("🔍 Show Sample Data"):
        st.dataframe(df.head(10))

else:
    st.info("Please upload a CSV file to run the simulation.")
