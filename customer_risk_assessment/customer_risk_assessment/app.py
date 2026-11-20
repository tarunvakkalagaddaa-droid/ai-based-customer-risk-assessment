"""Streamlit web app for customer risk assessment.

Run with:  streamlit run app.py
"""
import streamlit as st

from predict import assess_customer, load_model

st.set_page_config(page_title="Customer Risk Assessment", page_icon="🛡️")
st.title("🛡️ AI-Based Customer Risk Assessment")
st.caption("Enter customer details to predict their risk level.")


@st.cache_resource
def get_model():
    return load_model()


try:
    model = get_model()
except FileNotFoundError:
    st.error("Model not found. Run `python train_model.py` first.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 35)
    annual_income = st.number_input("Annual income", 0, 1_000_000, 60000, step=1000)
    credit_score = st.slider("Credit score", 300, 850, 680)
    employment_years = st.number_input("Years employed", 0.0, 50.0, 5.0, step=0.5)
    late_payments = st.number_input("Late payments (last 2 yrs)", 0, 50, 1)
with col2:
    loan_amount = st.number_input("Requested loan amount", 0, 1_000_000, 20000, step=1000)
    existing_debt = st.number_input("Existing debt", 0, 1_000_000, 15000, step=1000)
    num_credit_accounts = st.number_input("Number of credit accounts", 0, 50, 4)
    account_age_years = st.number_input("Oldest account age (years)", 0.0, 60.0, 6.0, step=0.5)

if st.button("Assess risk", type="primary"):
    result = assess_customer(
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
        },
        model,
    )
    level = result["risk_level"]
    {"Low": st.success, "Medium": st.warning, "High": st.error}[level](
        f"Predicted risk level: **{level}**"
    )
    st.bar_chart(result["probabilities"])
    dti = (existing_debt + loan_amount) / max(annual_income, 1)
    st.write(f"Debt-to-income ratio (incl. new loan): **{dti:.2f}**")
