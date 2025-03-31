import pandas as pd
import numpy as np

def generate_synthetic_data(n_customers=10000, random_seed=42):
    np.random.seed(random_seed)
    data = pd.DataFrame({
        "credit_score": np.random.normal(680, 50, n_customers).clip(500, 850),
        "income": np.random.normal(60000, 20000, n_customers).clip(15000, 200000),
        "loan_amount": np.random.normal(150000, 80000, n_customers).clip(5000, 500000),
        "loan_type": np.random.choice(["mortgage", "personal", "credit_card"], n_customers),
        "term_length": np.random.choice([5, 10, 15, 20, 25], n_customers),
        "defaulted": np.random.binomial(1, 0.05, n_customers)  # synthetic target
    })
    return data
