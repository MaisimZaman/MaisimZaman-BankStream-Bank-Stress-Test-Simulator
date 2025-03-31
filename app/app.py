import streamlit as st
import pandas as pd

import sys
import os

# Add the root of the project (one level up from /app) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your backend logic
from src.data_loader import generate_synthetic_data
from src.model import train_default_model
from src.risk_metrics import expected_loss, monte_carlo_var


st.set_page_config(page_title="BankStressSim", layout="wide")
st.title("🏦 Bank Stress Testing Simulator")

# Sidebar for macroeconomic inputs
st.sidebar.header("📉 Macro Stress Inputs")
interest_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 10.0, 5.0, 0.25)
unemployment_rate = st.sidebar.slider("Unemployment Rate (%)", 0.0, 15.0, 6.0, 0.5)
inflation_rate = st.sidebar.slider("Inflation Rate (%)", 0.0, 10.0, 2.5, 0.5)
capital_requirement = st.sidebar.slider("Capital Requirement (%)", 5.0, 15.0, 10.0, 0.5)

# Generate synthetic portfolio
st.header("📊 Simulating Loan Portfolio")
df = generate_synthetic_data(n_customers=5000)

# Train ML model and get PDs
model, feature_names = train_default_model(df)
df_encoded = pd.get_dummies(df, columns=["loan_type"])

# Ensure all training-time features exist
for col in feature_names:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

# Reorder columns
df_encoded = df_encoded[feature_names]

# Make prediction
df["pd"] = model.predict_proba(df_encoded)[:, 1]



# Adjust PDs based on macroeconomic inputs (simplified multiplier logic)
stress_multiplier = 1 + ((interest_rate - 3) * 0.05 + (unemployment_rate - 5) * 0.07)
df["pd"] = (df["pd"] * stress_multiplier).clip(0, 1)

# Calculate risk metrics
el = expected_loss(df, pd_col="pd", lgd=0.45)
var_95 = monte_carlo_var(df, confidence=0.95, lgd=0.45)
total_loan = df["loan_amount"].sum()
capital_held = total_loan * 0.10  # assume bank holds 10% of loan book
car = 100 * capital_held / (el + 1e-6)

# Display metrics
st.subheader("📈 Portfolio Metrics")
st.metric("Expected Loss", f"${el:,.2f}")
st.metric("Value at Risk (95%)", f"${var_95:,.2f}")
st.metric("Capital Adequacy Ratio", f"{car:.2f}%", delta=f"{car - capital_requirement:.2f}%", delta_color="inverse")

# Optional: Show data table
with st.expander("🔍 Show Sample Data"):
    st.dataframe(df.head(10))
