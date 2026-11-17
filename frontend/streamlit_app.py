

import os
import streamlit as st
import requests


# FastAPI backend URL
API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict"
)


# --------------------------------------------------
# 1. Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Telco Customer Churn Predictor")

st.write(
    "Enter customer details below to predict "
    "whether the customer is likely to churn."
)


# --------------------------------------------------
# 2. Customer input form
# --------------------------------------------------

with st.form("customer_prediction_form"):

    st.subheader("Customer Information")

    gender = st.selectbox("Gender", ["Male", "Female"])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])

    st.subheader("Services Information")

    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])

    MultipleLines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    InternetService = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    OnlineSecurity = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    OnlineBackup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    DeviceProtection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    TechSupport = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    StreamingTV = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    StreamingMovies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    st.subheader("Billing Information")

    Contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    PaperlessBilling = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
    )

    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    st.subheader("Charges and Tenure")

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12,
        step=1
    )

    MonthlyCharges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

    TotalCharges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=600.0,
        step=10.0
    )

    submitted = st.form_submit_button(
        "Predict Customer Churn"
    )


# --------------------------------------------------
# 3. Send customer data to FastAPI
# --------------------------------------------------

if submitted:

    payload = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges)
    }

    try:
        with st.spinner("Analyzing customer information..."):

            # Send POST request to FastAPI
            response = requests.post(
                API_URL,
                json=payload,
                timeout=30
            )

            # Raise error for unsuccessful HTTP responses
            response.raise_for_status()

            # Convert API response into Python dictionary
            result = response.json()["prediction"]

        st.subheader("Prediction Result")

        if "likely to churn" in str(result).lower():
            st.error(f"⚠️ {result}")
        else:
            st.success(f"✅ {result}")

    except requests.exceptions.RequestException as e:
        st.error(f"FastAPI connection failed: {str(e)}")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
