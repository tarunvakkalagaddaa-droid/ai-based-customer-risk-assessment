"""Load the trained model and score a customer."""
import argparse
import os

import joblib
import pandas as pd

from features import add_features, FEATURE_COLUMNS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "risk_model.joblib")


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError("Model not found. Run `python train_model.py` first.")
    return joblib.load(path)


def assess_customer(customer: dict, model=None) -> dict:
    """Return the predicted risk level and class probabilities."""
    model = model or load_model()
    df = add_features(pd.DataFrame([customer]))
    X = df[FEATURE_COLUMNS]
    probs = model.predict_proba(X)[0]
    proba = {cls: round(float(p), 3) for cls, p in zip(model.classes_, probs)}
    level = model.classes_[probs.argmax()]
    return {"risk_level": level, "probabilities": proba}


def main():
    p = argparse.ArgumentParser(description="AI-Based Customer Risk Assessment")
    p.add_argument("--age", type=int, default=35)
    p.add_argument("--annual_income", type=float, default=60000)
    p.add_argument("--credit_score", type=int, default=680)
    p.add_argument("--employment_years", type=float, default=5)
    p.add_argument("--loan_amount", type=float, default=20000)
    p.add_argument("--existing_debt", type=float, default=15000)
    p.add_argument("--late_payments", type=int, default=1)
    p.add_argument("--num_credit_accounts", type=int, default=4)
    p.add_argument("--account_age_years", type=float, default=6)
    args = vars(p.parse_args())

    result = assess_customer(args)
    print(f"Predicted risk level: {result['risk_level']}")
    for cls, prob in result["probabilities"].items():
        print(f"  {cls:<7}{prob:.1%}")


if __name__ == "__main__":
    main()
