import numpy as np

def expected_loss(df, pd_col="pd", lgd=0.45):
    df["EL"] = df["LoanAmount"] * df[pd_col] * lgd
    return df["EL"].sum()

def monte_carlo_var_and_cvar(df, simulations=10000, confidence=0.95, lgd=0.45):
    import numpy as np
    losses = []
    loan_amounts = df["LoanAmount"].values
    pds = df["pd"].values

    for _ in range(simulations):
        simulated_defaults = np.random.binomial(1, pds)
        simulated_loss = np.sum(simulated_defaults * loan_amounts * lgd)
        losses.append(simulated_loss)

    losses = np.array(losses)
    var = np.percentile(losses, confidence * 100)
    cvar = losses[losses >= var].mean() if np.any(losses >= var) else 0.0

    return var, cvar

def monte_carlo_var(df, simulations=10000, confidence=0.95, lgd=0.45):
    losses = []
    for _ in range(simulations):
        simulated_defaults = np.random.binomial(1, df["pd"])
        simulated_loss = (simulated_defaults * df["LoanAmount"] * lgd).sum()
        losses.append(simulated_loss)

    return np.percentile(losses, confidence * 100)
