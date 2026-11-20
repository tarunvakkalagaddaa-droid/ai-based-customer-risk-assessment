"""Shared feature engineering used by training and prediction."""
import pandas as pd

BASE_COLUMNS = [
    "age",
    "annual_income",
    "credit_score",
    "employment_years",
    "loan_amount",
    "existing_debt",
    "late_payments",
    "num_credit_accounts",
    "account_age_years",
]

FEATURE_COLUMNS = BASE_COLUMNS + ["debt_to_income", "loan_to_income"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    income = df["annual_income"].replace(0, 1)
    df["debt_to_income"] = (df["existing_debt"] + df["loan_amount"]) / income
    df["loan_to_income"] = df["loan_amount"] / income
    return df
