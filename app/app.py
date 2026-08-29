import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND FEATURE NAME MAP
# ============================================================



@st.cache_resource
def load_resources():

    BASE_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )

    PROJECT_DIR = os.path.dirname(BASE_DIR)

    model_path = os.path.join(
        PROJECT_DIR,
        "models",
        "churn_model.pkl"
    )

    feature_map_path = os.path.join(
        PROJECT_DIR,
        "models",
        "feature_name_map.pkl"
    )

    model = joblib.load(model_path)

    feature_name_map = joblib.load(
        feature_map_path
    )

    return model, feature_name_map


try:
    best_lr_model, FEATURE_NAME_MAP = load_resources()

except Exception as e:

    st.error(
        f"Error loading model files: {e}"
    )

    st.stop()


# ============================================================
# CONFIGURATION
# ============================================================

THRESHOLD = 0.35


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_customer(customer_data):

    churn_probability = (
        best_lr_model.predict_proba(customer_data)[0][1]
    )

    prediction = (
        "Yes"
        if churn_probability >= THRESHOLD
        else "No"
    )

    return {
        "churn_probability": float(churn_probability),
        "threshold": THRESHOLD,
        "prediction": prediction
    }


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(churn_probability):

    if churn_probability >= 0.70:
        return "High"

    elif churn_probability >= THRESHOLD:
        return "Medium"

    else:
        return "Low"


# ============================================================
# CUSTOMER ASSESSMENT
# ============================================================

def assess_customer(customer_df):

    prediction_result = predict_customer(
        customer_df
    )

    risk_level = classify_risk(
        prediction_result["churn_probability"]
    )

    prediction_result["risk_level"] = risk_level

    return prediction_result


# ============================================================
# CUSTOMER EXPLANATION
# ============================================================

def explain_customer(customer_data, top_n=5):

    # Get preprocessing pipeline
    preprocessor = (
        best_lr_model.named_steps["preprocessor"]
    )

    # Get logistic regression model
    model = (
        best_lr_model.named_steps["model"]
    )

    # Transform customer data
    transformed_customer = (
        preprocessor.transform(customer_data)
    )

    # Get transformed feature names
    feature_names = (
        preprocessor.get_feature_names_out()
    )

    # Get coefficients
    coefficients = model.coef_[0]

    # Convert sparse matrix if necessary
    if hasattr(transformed_customer, "toarray"):

        transformed_customer = (
            transformed_customer.toarray()
        )

    # Customer values
    values = transformed_customer[0]

    # Calculate contributions
    contributions = values * coefficients

    # Create explanation dataframe
    explanation_df = pd.DataFrame({

        "Feature": feature_names,

        "Value": values,

        "Coefficient": coefficients,

        "Contribution": contributions

    })

    # Sort contributions
    explanation_df = explanation_df.sort_values(
        "Contribution",
        ascending=False
    )

    # Higher-risk factors
    higher_risk = explanation_df[
        explanation_df["Contribution"] > 0
    ].head(top_n)

    # Lower-risk factors
    lower_risk = explanation_df[
        explanation_df["Contribution"] < 0
    ].sort_values(
        "Contribution",
        ascending=True
    ).head(top_n)

    return {

        "higher_risk_factors": higher_risk,

        "lower_risk_factors": lower_risk

    }


# ============================================================
# CLEAN FEATURE NAMES
# ============================================================

def clean_feature_names(explanation):

    higher_risk = (
        explanation["higher_risk_factors"].copy()
    )

    lower_risk = (
        explanation["lower_risk_factors"].copy()
    )

    higher_risk["Feature"] = (
        higher_risk["Feature"].map(
            lambda x: FEATURE_NAME_MAP.get(x, x)
        )
    )

    lower_risk["Feature"] = (
        lower_risk["Feature"].map(
            lambda x: FEATURE_NAME_MAP.get(x, x)
        )
    )

    return {

        "higher_risk_factors": higher_risk,

        "lower_risk_factors": lower_risk

    }


# ============================================================
# STREAMLIT INTERFACE
# ============================================================

st.title("📊 Telco Customer Churn Prediction System")

st.write(
    """
    Enter customer information below to estimate churn risk
    and view the factors associated with the prediction.
    """
)

st.divider()


# ============================================================
# CUSTOMER INPUTS
# ============================================================

st.subheader("Customer Information")


col1, col2, col3 = st.columns(3)


# ----------------------------
# COLUMN 1
# ----------------------------

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )


# ----------------------------
# COLUMN 2
# ----------------------------

with col2:

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )


# ----------------------------
# COLUMN 3
# ----------------------------

with col3:

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )


st.divider()


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "Predict Customer Churn",
    type="primary"
):

    # Create customer dataframe

    customer_df = pd.DataFrame([{

        "gender": gender,

        "SeniorCitizen": senior_citizen,

        "Partner": partner,

        "Dependents": dependents,

        "tenure": tenure,

        "PhoneService": phone_service,

        "MultipleLines": multiple_lines,

        "InternetService": internet_service,

        "OnlineSecurity": online_security,

        "OnlineBackup": online_backup,

        "DeviceProtection": device_protection,

        "TechSupport": tech_support,

        "StreamingTV": streaming_tv,

        "StreamingMovies": streaming_movies,

        "Contract": contract,

        "PaperlessBilling": paperless_billing,

        "PaymentMethod": payment_method,

        "MonthlyCharges": monthly_charges,

        "TotalCharges": total_charges

    }])


    # ========================================================
    # MODEL ASSESSMENT
    # ========================================================

    assessment = assess_customer(
        customer_df
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    st.divider()

    st.subheader("Prediction Results")

    result_col1, result_col2, result_col3, result_col4 = (
        st.columns(4)
    )

    with result_col1:

        st.metric(
            "Churn Probability",
            f"{assessment['churn_probability']:.2%}"
        )


    with result_col2:

        st.metric(
            "Decision Threshold",
            f"{assessment['threshold']:.2f}"
        )


    with result_col3:

        st.metric(
            "Prediction",
            assessment["prediction"]
        )


    with result_col4:

        st.metric(
            "Risk Level",
            assessment["risk_level"]
        )


    # ========================================================
    # RISK MESSAGE
    # ========================================================

    if assessment["risk_level"] == "High":

        st.error(
            "High predicted churn risk."
        )

    elif assessment["risk_level"] == "Medium":

        st.warning(
            "Medium predicted churn risk."
        )

    else:

        st.success(
            "Low predicted churn risk."
        )


    # ========================================================
    # CUSTOMER EXPLANATION
    # ========================================================

    explanation = explain_customer(
        customer_df
    )

    clean_explanation = clean_feature_names(
        explanation
    )


    # ========================================================
    # DISPLAY RISK FACTORS
    # ========================================================

    st.divider()

    st.subheader(
        "Customer-Level Risk Factors"
    )


    factor_col1, factor_col2 = st.columns(2)


    # Higher risk factors

    with factor_col1:

        st.markdown(
            "### 🔴 Higher-Risk Factors"
        )

        higher_risk_display = (
            clean_explanation[
                "higher_risk_factors"
            ][
                ["Feature", "Contribution"]
            ]
        )

        st.dataframe(
            higher_risk_display,
            use_container_width=True,
            hide_index=True
        )


    # Lower risk factors

    with factor_col2:

        st.markdown(
            "### 🟢 Lower-Risk Factors"
        )

        lower_risk_display = (
            clean_explanation[
                "lower_risk_factors"
            ][
                ["Feature", "Contribution"]
            ]
        )

        st.dataframe(
            lower_risk_display,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        "This application provides a machine-learning "
        "prediction based on customer information. "
        "The final business decision remains with a human "
        "decision-maker."
    )
