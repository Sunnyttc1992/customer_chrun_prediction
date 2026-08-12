import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from src.serving.inference import predict

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="centered",
    initial_sidebar_state="expanded",
)

PRIMARY_COLOR = "#2E2F31"
SECONDARY_COLOR = "#8D8F8F"
CARD_BG = "#F6F5F3"
TEXT_COLOR = "#1F2225"

st.markdown(
    f"""
    <style>
        .reportview-container {{ background-color: #F7F6F2; color: {TEXT_COLOR}; }}
        .stApp {{ background-color: #F7F6F2; color: {TEXT_COLOR}; }}
        .css-1d391kg {{ background-color: {CARD_BG}; }}
        .css-18e3th9 {{ color: {PRIMARY_COLOR}; }}
        .stButton>button {{ background-color: {PRIMARY_COLOR}; color: white; border-radius: 10px; }}
        .stButton>button:hover {{ background-color: #414446; }}
        .stTextInput>div>div>input {{ background-color: white; color: {TEXT_COLOR}; }}
        .stSelectbox>div>div>div>div {{ background-color: white; color: {TEXT_COLOR}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Teleco Churn Predictor")
st.write(
    "Predict customer churn risk with a minimal, calm interface built for confident decisions. "
    "All inputs validate quietly, and model confidence is shown clearly."
)

st.markdown("---")

with st.form(key="prediction_form"):
    st.subheader("Customer profile")

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=120,
        value=12,
        step=1,
        help="How long the customer has been active.",
    )
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=500.0,
        value=70.0,
        step=0.5,
    )
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=50000.0,
        value=float(max(tenure * monthly_charges, 0.0)),
        step=1.0,
    )

    contract = st.selectbox(
        "Contract type",
        ["Month-to-month", "One year", "Two year"],
    )
    payment_method = st.selectbox(
        "Payment method",
        ["Credit", "Debit", "UPI"],
    )
    internet_service = st.selectbox(
        "Internet service",
        ["DSL", "Fiber optic", "No"],
    )
    tech_support = st.selectbox(
        "Tech support",
        ["Yes", "No"],
    )
    online_security = st.selectbox(
        "Online security",
        ["Yes", "No"],
    )
    support_calls = st.number_input(
        "Customer support calls",
        min_value=0,
        max_value=20,
        value=1,
        step=1,
    )

    submit_button = st.form_submit_button("Calculate churn risk")

if submit_button:
    if total_charges < monthly_charges:
        st.warning(
            "Total charges are less than monthly charges — check tenure or billing values."
        )

    inputs = {
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract": contract,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "online_security": online_security,
        "support_calls": support_calls,
    }

    with st.spinner("Evaluating churn risk…"):
        result = predict(inputs)

    st.markdown("---")
    st.subheader("Prediction")
    st.success(result["prediction"])
    st.write(f"**Confidence:** {result['confidence'] * 100:.1f}%")
    st.write(f"**Churn probability:** {result['probability']:.3f}")
    st.info(
        "This model is tuned to highlight customers whose behavior patterns "
        "suggest they may leave soon. Use the result as a calm guide, not a final decision."
    )

    st.markdown(
        "<div style='margin-top: 1rem; padding: 1rem; background: #FFFFFF; border-radius: 14px;'>" 
        "<strong>Next step</strong>: offer proactive service or a loyalty incentive to customers with elevated churn risk.</div>",
        unsafe_allow_html=True,
    )
