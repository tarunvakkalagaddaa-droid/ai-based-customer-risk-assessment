"""Generate a synthetic customer dataset for risk assessment.

Each customer gets a risk label (Low / Medium / High) derived from a
weighted score of financial and behavioural features plus some noise,
so a model has a realistic (but learnable) pattern to pick up.
"""
import os

import numpy as np
import pandas as pd

RNG_SEED = 42


def generate_customers(n: int = 5000, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 70, n)
    annual_income = rng.normal(55000, 22000, n).clip(12000, 250000).round(0)
    credit_score = rng.normal(660, 85, n).clip(300, 850).round(0)
    employment_years = np.clip(rng.normal(7, 5, n), 0, 40).round(1)
    loan_amount = (annual_income * rng.uniform(0.1, 1.5, n)).round(0)
    existing_debt = (annual_income * rng.uniform(0.0, 1.0, n)).round(0)
    late_payments = rng.poisson(1.2, n).clip(0, 12)
    num_credit_accounts = rng.integers(1, 12, n)
    account_age_years = np.clip(rng.normal(6, 4, n), 0, 30).round(1)

    dti = (existing_debt + loan_amount) / annual_income  # debt-to-income

    # Weighted risk score (higher = riskier)
    score = (
        -0.012 * (credit_score - 650)
        + 1.4 * dti
        + 0.45 * late_payments
        - 0.06 * employment_years
        - 0.05 * account_age_years
        - 0.00001 * (annual_income - 55000)
        + 0.03 * np.maximum(0, 25 - age)
        + rng.normal(0, 0.6, n)
    )

    low_cut, high_cut = np.quantile(score, [0.45, 0.80])
    risk = np.where(score < low_cut, "Low", np.where(score < high_cut, "Medium", "High"))

    return pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income,
            "credit_score": credit_score,
            "employment_years": employment_years,
            "loan_amount": loan_amount,
            "existing_debt": existing_debt,
            "late_payments": late_payments,
            "num_credit_accounts": num_credit_accounts,
            "account_age_years": account_age_years,
            "risk_level": risk,
        }
    )


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_customers()
    df.to_csv("data/customers.csv", index=False)
    print(f"Saved data/customers.csv with {len(df)} rows")
    print(df["risk_level"].value_counts())
