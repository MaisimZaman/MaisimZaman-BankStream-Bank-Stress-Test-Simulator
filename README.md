Dataset used for this project: https://www.kaggle.com/datasets/nikhil1e9/loan-default


# 🏦 BankStressSim – ML-Powered Loan Portfolio Risk Simulator

BankStressSim is a fully interactive, machine learning–driven loan portfolio risk simulator designed for banks, fintechs, and analysts. It uses real-world loan data, predictive modeling, and macroeconomic stress inputs to estimate:

- 📉 Expected Loss (EL)
- 🚨 Value at Risk (VaR)
- 🔥 Conditional VaR (CVaR)
- 🏦 Capital Adequacy Ratio (CAR)
- 🧠 SHAP-based model explanations
- 📊 Borrower-level risk flags

This project simulates stress testing like real financial institutions under Basel III, OSFI, or IFRS 9 guidelines.

---

## 🚀 Features

✅ Upload a loan dataset (CSV)  
✅ Predict probability of default (PD) using a trained XGBoost model  
✅ Adjust macroeconomic sliders (Interest Rate, Unemployment, Capital Requirement)  
✅ View risk metrics: EL, VaR, CVaR, CAR  
✅ SHAP explanations for any borrower  
✅ Download risk-flagged portfolio as CSV  
✅ Analyze top drivers of default

---

## 📁 Folder Structure

```
BankStressSim/
│
├── app/                   # Streamlit frontend
│   └── app.py
├── src/                   # Backend logic
│   ├── model.py
│   ├── data_loader.py
│   ├── risk_metrics.py
│   └── utils.py
├── models/                # Saved XGBoost model (loan_default_model.pkl)
├── data/                  # Source or sample CSVs
├── tests/                 # Model evaluation scripts
└── requirements.txt       # Required Python packages
```

---

## 🧪 How to Run the App Locally

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/BankStressSim.git
cd BankStressSim
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app/app.py
```

---

## 📥 How to Use the App

1. Upload a CSV containing loan applicant data
2. Adjust macroeconomic stress inputs using sidebar sliders
3. View:
   - Predicted PD per borrower
   - Total Expected Loss and VaR
   - Capital Adequacy Ratio (CAR)
4. Select a borrower to explain using SHAP
5. Filter or download high-risk borrowers from the dashboard

---

## 📊 Sample Input Format (CSV)

```
LoanID,Age,Income,LoanAmount,CreditScore,MonthsEmployed,NumCreditLines,InterestRate,LoanTerm,DTIRatio,Education,EmploymentType,MaritalStatus,HasMortgage,HasDependents,LoanPurpose,HasCoSigner
ID001,45,65000,25000,700,72,4,7.5,36,0.35,Bachelor's,Full-time,Married,Yes,No,Auto,No
...
```

---

## 🧠 Model Training

The backend uses an XGBoost classifier trained on borrower features to estimate the probability of default. You can retrain or update it using:
```python
from src.model import train_default_model
```

Save the model:
```python
joblib.dump(model, "models/loan_default_model.pkl")
```

---

## 🧠 SHAP Explainability

SHAP plots and summaries are used to explain the top contributing factors behind each borrower’s risk score. Each prediction is fully transparent and auditable.

---

## 📌 Requirements

- Python 3.8+
- streamlit
- pandas
- numpy
- xgboost
- shap
- scikit-learn
- matplotlib
- joblib

---

## 🙌 Credit

Built for educational and institutional risk prototyping.  
Inspired by real-world Basel III and IFRS 9 frameworks.

---

## 📧 Questions?

DM me or open an issue. Want to deploy it to your team, customize it for a real credit team, or simulate 2008-style macro scenarios? Let’s talk.

