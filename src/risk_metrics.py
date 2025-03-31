import numpy as np

def expected_loss(df, pd_col="pd", lgd=0.45):
    df["EL"] = df["loan_amount"] * df[pd_col] * lgd
    return df["EL"].sum()

def monte_carlo_var(df, simulations=10000, confidence=0.95, lgd=0.45):
    losses = []
    for _ in range(simulations):
        simulated_defaults = np.random.binomial(1, df["pd"])
        simulated_loss = (simulated_defaults * df["loan_amount"] * lgd).sum()
        losses.append(simulated_loss)
    losses.sort()
    var_index = int((1 - confidence) * len(losses))
    return losses[var_index]
