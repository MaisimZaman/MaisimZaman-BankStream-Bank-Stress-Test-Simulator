import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

def train_default_model(df):
    df = df.copy()
    
    # Drop irrelevant columns
    df.drop(columns=["LoanID"], inplace=True)

    # Binary conversions
    binary_cols = ["HasMortgage", "HasDependents", "HasCoSigner"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Encode categorical variables
    cat_cols = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Separate features and target
    X = df.drop(columns=["Default"])
    y = df["Default"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Resample with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Train XGBoost model
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),  # helps with imbalance
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_resampled, y_resampled)

    # Predict on original test set
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["No Default", "Default"], digits=3)

    return model, list(X.columns), report
