"""Train the customer risk model and save it to models/."""
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from generate_data import generate_customers
from features import add_features, FEATURE_COLUMNS

DATA_PATH = "data/customers.csv"
MODEL_PATH = "models/risk_model.joblib"


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = generate_customers()
        df.to_csv(DATA_PATH, index=False)
        print(f"Generated {DATA_PATH}")

    df = add_features(df)
    X = df[FEATURE_COLUMNS]
    y = df["risk_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}\n")
    print(classification_report(y_test, pred))
    labels = ["Low", "Medium", "High"]
    print("Confusion matrix (rows=actual, cols=predicted):", labels)
    print(confusion_matrix(y_test, pred, labels=labels))

    print("\nTop feature importances:")
    imp = sorted(zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda t: -t[1])
    for name, val in imp:
        print(f"  {name:<24}{val:.3f}")

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
