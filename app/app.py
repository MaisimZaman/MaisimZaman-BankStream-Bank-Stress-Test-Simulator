import streamlit as st
import pandas as pd
import sys
import os
import joblib
import matplotlib.pyplot as plt
import shap
import numpy as np

# Add the root of the project (one level up from /app) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import risk metric functions
from src.risk_metrics import expected_loss, monte_carlo_var_and_cvar


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

    df_encoded = df_encoded.astype(float)


    # === 🔮 Predict Default Probabilities ===
    df["pd"] = model.predict_proba(df_encoded)[:, 1]

    # Apply macroeconomic stress multiplier
    stress_multiplier = 1 + ((interest_rate - 3) * 0.05 + (unemployment_rate - 5) * 0.07)
    df["pd"] = (df["pd"] * stress_multiplier).clip(0, 1)
    df["RiskFlag"] = pd.cut(
        df["pd"],
        bins=[-0.01, 0.1, 0.3, 0.6, 1.0],
        labels=["✅ Low Risk", "🟡 Moderate", "🟠 High", "🔴 Critical"]
    )
    df["EL"] = df["LoanAmount"] * df["pd"] * 0.45  # Assuming fixed LGD


    # === 📈 Risk Metrics ===
    el = expected_loss(df, pd_col="pd", lgd=0.45)
    var_95, cvar_95 = monte_carlo_var_and_cvar(df, confidence=0.95, lgd=0.45)
    total_loan = df["LoanAmount"].sum()
    capital_held = total_loan * (capital_requirement / 100)
    car = 100 * capital_held / (el + 1e-6)


    #SHAP Explainer
    explainer = shap.Explainer(model, df_encoded)
    shap_values = explainer(df_encoded)

    # === 📊 Display Metrics ===
    st.subheader("📈 Portfolio Risk Metrics")
    st.metric("Expected Loss", f"${el:,.2f}")
    st.metric("Value at Risk (95%)", f"${var_95:,.2f}")
    st.metric("Capital Adequacy Ratio", f"{car:.2f}%", delta=f"{car - capital_requirement:.2f}%", delta_color="inverse")
    st.metric("Conditional VaR (95%)", f"${cvar_95:,.2f}")


    st.subheader("🚨 High Risk Borrowers")
    risk_level = st.selectbox("Select Risk Category", ["All", "🟠 High", "🔴 Critical"])

    if risk_level != "All":
        st.dataframe(df[df["RiskFlag"] == risk_level].sort_values("pd", ascending=False))
    else:
        st.dataframe(df[df["RiskFlag"].isin(["🟠 High", "🔴 Critical"])].sort_values("pd", ascending=False))

    #st.subheader("🧠 SHAP Explanation (Model Transparency)")

    borrower_index = st.slider("Select a borrower to explain", 0, len(df) - 1, 0)

    st.write("Borrower Info:")
    st.write(df.iloc[borrower_index])


    importances = model.feature_importances_
    features = feature_names  # already stored during training



    st.subheader("📄 SHAP-Based Explanation (Text Summary)")

    shap_vals_row = shap_values[borrower_index].values
    top_indices = np.argsort(np.abs(shap_vals_row))[-3:][::-1]

    st.markdown("#### 🔍 Top Risk Drivers for This Borrower:")

    for idx in top_indices:
        feature = df_encoded.columns[idx]
        shap_value = shap_vals_row[idx]
        effect = "increased" if shap_value > 0 else "decreased"
        strength = "strongly" if abs(shap_value) > 0.1 else "slightly"

    st.markdown(f"- **{feature}** {strength} **{effect}** the probability of default by **{shap_value:+.3f}**")


    st.subheader("📥 Export Full Portfolio Results")

    csv = df.to_csv(index=False)


    st.download_button(
        label="📁 Download Portfolio with PD & Risk Flags",
        data=csv,
        file_name="portfolio_results.csv",
        mime="text/csv"
    )





    # Optional: show sample
    with st.expander("🔍 Show Sample Data"):
        st.dataframe(df.head(10))

else:
    st.info("Please upload a CSV file to run the simulation.")
