# AI-Based Customer Risk Assessment

A simple machine-learning project that classifies customers as **Low**, **Medium**
or **High** risk using financial and behavioural features (credit score, income,
debt-to-income ratio, late payments, etc.).

It uses a Random Forest classifier (scikit-learn) trained on a synthetic dataset,
so it runs with no external data. You can swap in a real CSV with the same columns.

## Project structure

```
customer_risk_assessment/
├── generate_data.py   # creates data/customers.csv (synthetic data)
├── features.py        # shared feature engineering (debt_to_income, loan_to_income)
├── train_model.py     # trains the model, prints metrics, saves models/risk_model.joblib
├── predict.py         # command-line prediction + assess_customer() function
├── app.py             # Streamlit web app
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Generate data (optional - training does this automatically if missing):
   ```bash
   python generate_data.py
   ```
2. Train the model:
   ```bash
   python train_model.py
   ```
3. Predict from the command line:
   ```bash
   python predict.py --credit_score 580 --annual_income 40000 --loan_amount 30000 --existing_debt 25000 --late_payments 4
   ```
4. Launch the web app:
   ```bash
   streamlit run app.py
   ```

## Using your own data

Replace `data/customers.csv` with a file containing these columns, then re-run
`train_model.py`:

`age, annual_income, credit_score, employment_years, loan_amount, existing_debt,
late_payments, num_credit_accounts, account_age_years, risk_level`

`risk_level` must be one of `Low`, `Medium`, `High`.

## Possible extensions

- Try other models (Gradient Boosting, XGBoost, Logistic Regression) and compare
- Add SHAP explanations for each prediction
- Expose a REST API with FastAPI
- Store assessments in a database
