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
def build_verified_payload(customer_data):

    # --------------------------------
    # 1. Get model assessment
    # --------------------------------

    assessment = assess_customer(customer_data)

    # --------------------------------
    # 2. Get customer explanation
    # --------------------------------

    explanation = explain_customer(customer_data)

    # --------------------------------
    # 3. Clean feature names
    # --------------------------------

    clean_explanation = clean_feature_names(
        explanation
    )

    # --------------------------------
    # 4. Build higher-risk signals
    # --------------------------------

    higher_risk = []

    for _, row in clean_explanation[
        "higher_risk_factors"
    ].iterrows():

        higher_risk.append({

            "feature": row["Feature"],

            "contribution": round(
                float(row["Contribution"]),
                3
            )

        })

    # --------------------------------
    # 5. Build lower-risk signals
    # --------------------------------

    lower_risk = []

    for _, row in clean_explanation[
        "lower_risk_factors"
    ].iterrows():

        lower_risk.append({

            "feature": row["Feature"],

            "contribution": round(
                float(row["Contribution"]),
                3
            )

        })

    # --------------------------------
    # 6. Build verified payload
    # --------------------------------

    verified_payload = {

        "churn_probability": round(
            float(assessment["churn_probability"]),
            4
        ),

        "threshold": float(
            assessment["threshold"]
        ),

        "prediction": assessment["prediction"],

        "risk_level": assessment["risk_level"],

        "higher_risk_signals": higher_risk,

        "lower_risk_signals": lower_risk
    }

    return verified_payload

def generate_business_response(verified_payload):

    higher_risk_factors = [

        item["feature"]

        for item in verified_payload[
            "higher_risk_signals"
        ]
    ]

    lower_risk_factors = [

        item["feature"]

        for item in verified_payload[
            "lower_risk_signals"
        ]
    ]


    response = {

        "risk_summary": (

            f"Churn probability: "
            f"{verified_payload['churn_probability']}, "

            f"Decision threshold: "
            f"{verified_payload['threshold']}, "

            f"Prediction: "
            f"{verified_payload['prediction']}, "

            f"Risk Level: "
            f"{verified_payload['risk_level']}"
        ),

        "strongest_factors": higher_risk_factors,

        "lower_risk_factors": lower_risk_factors,

        "business_considerations": (

            "The model identifies the listed higher-risk "
            "and lower-risk factors associated with this "
            "prediction. The final business decision "
            "remains with a human decision-maker."
        )
    }

    return response
def validate_business_response(
    response,
    verified_payload
):

    violations = []

    # --------------------------------
    # Validate model values
    # --------------------------------

    summary = response[
        "risk_summary"
    ].lower()

    probability = str(
        verified_payload["churn_probability"]
    )

    threshold = str(
        verified_payload["threshold"]
    )

    prediction = verified_payload[
        "prediction"
    ].lower()

    risk_level = verified_payload[
        "risk_level"
    ].lower()


    if probability not in summary:

        violations.append(
            "Incorrect or missing churn probability"
        )


    if threshold not in summary:

        violations.append(
            "Incorrect or missing decision threshold"
        )


    if prediction not in summary:

        violations.append(
            "Incorrect or missing prediction"
        )


    if risk_level not in summary:

        violations.append(
            "Incorrect or missing risk level"
        )


    # --------------------------------
    # Expected factors
    # --------------------------------

    expected_higher = [

        item["feature"]

        for item in verified_payload[
            "higher_risk_signals"
        ]
    ]

    expected_lower = [

        item["feature"]

        for item in verified_payload[
            "lower_risk_signals"
        ]
    ]


    # --------------------------------
    # Validate higher-risk factors
    # --------------------------------

    if response[
        "strongest_factors"
    ] != expected_higher:

        violations.append(
            "Higher-risk factor integrity check failed"
        )


    # --------------------------------
    # Validate lower-risk factors
    # --------------------------------

    if response[
        "lower_risk_factors"
    ] != expected_lower:

        violations.append(
            "Lower-risk factor integrity check failed"
        )


    # --------------------------------
    # Forbidden phrases
    # --------------------------------

    forbidden_phrases = [

        "within one year",

        "within a year",

        "will definitely churn",

        "discount",

        "offer",

        "financial loss"
    ]


    full_response = " ".join([

        response["risk_summary"],

        " ".join(
            response["strongest_factors"]
        ),

        " ".join(
            response["lower_risk_factors"]
        ),

        response["business_considerations"]

    ]).lower()


    for phrase in forbidden_phrases:

        if phrase in full_response:

            violations.append(phrase)


    return {

        "approved": len(violations) == 0,

        "violations": violations
    }


# ============================================================
# STREAMLIT INTERFACE
# ============================================================
# ========================================================
# PROJECT KNOWLEDGE BASE
# ========================================================

PROJECT_KNOWLEDGE = {

    "model_selection": """
Logistic Regression was selected as the final model because
the business objective prioritized identifying as many actual
churners as possible. At the final decision threshold of 0.35,
Logistic Regression achieved higher recall and fewer false
negatives than Random Forest in the final comparison.
""",

    "final_model": """
The final model is Logistic Regression with C=1 and
class_weight='balanced'. The classification threshold was set
to 0.35 instead of the default 0.50.
""",

    "recall": """
Recall was prioritized because missing a customer who is
actually likely to churn is considered more costly than
contacting some customers who may ultimately stay.
""",

    "threshold": """
The threshold was selected using model performance and an
illustrative business-cost analysis. A threshold of 0.35 provided
a strong balance between identifying churners and controlling
unnecessary false-positive interventions.
""",

    "model_comparison": """
In the final comparison at a threshold of 0.35, Logistic
Regression achieved approximately 90.37% recall compared with
83.96% for Random Forest. Logistic Regression also produced
fewer false negatives: 36 compared with 60 for Random Forest.

Random Forest had stronger precision and a slightly higher F1
score, but Logistic Regression was selected because the primary
business objective was minimizing missed churners.
""",

    "limitations": """
The model provides a probability-based prediction rather than
certainty. The prediction depends on the available customer
features and historical data used for training. The output should
support, rather than replace, human business decision-making.
"""
}

# ========================================================
# INTERACTIVE DECISION-SUPPORT ASSISTANT
# ========================================================

def answer_business_question(question, verified_payload):

    question_lower = question.lower()

    # ----------------------------------------------------
    # Why was Logistic Regression selected?
    # ----------------------------------------------------

    if any(word in question_lower for word in [
        "why choose",
        "why selected",
        "why logistic",
        "model chosen",
        "choose the model",
        "selected model"
    ]):

        return (
            PROJECT_KNOWLEDGE["model_selection"]
            + "\n\n"
            + PROJECT_KNOWLEDGE["final_model"]
        )


    # ----------------------------------------------------
    # Recall
    # ----------------------------------------------------

    elif any(word in question_lower for word in [
        "recall",
        "missed churner",
        "false negative",
        "fn"
    ]):

        return PROJECT_KNOWLEDGE["recall"]


    # ----------------------------------------------------
    # Threshold
    # ----------------------------------------------------

    elif any(word in question_lower for word in [
        "threshold",
        "0.35",
        "decision boundary",
        "cutoff"
    ]):

        return (
            PROJECT_KNOWLEDGE["threshold"]
            + "\n\n"
            + f"The threshold used for this prediction is "
              f"{verified_payload['threshold']}."
        )


    # ----------------------------------------------------
    # Random Forest comparison
    # ----------------------------------------------------

    elif any(word in question_lower for word in [
        "random forest",
        "why not random",
        "compare models",
        "model comparison"
    ]):

        return PROJECT_KNOWLEDGE["model_comparison"]


    # ----------------------------------------------------
    # Why is this customer high risk?
    # ----------------------------------------------------

    elif any(word in question_lower for word in [
        "why high risk",
        "why is this customer",
        "why did the customer",
        "why churn",
        "why this prediction"
    ]):

        higher_factors = [

            item["feature"]

            for item in verified_payload[
                "higher_risk_signals"
            ]
        ]

        return (
            f"This customer has a churn probability of "
            f"{verified_payload['churn_probability']:.2%}, "
            f"which is above the decision threshold of "
            f"{verified_payload['threshold']}. "
            f"The prediction is therefore "
            f"{verified_payload['prediction']} with a "
            f"{verified_payload['risk_level']} risk level.\n\n"

            f"The strongest factors associated with higher "
            f"predicted churn risk are: "
            f"{', '.join(higher_factors)}.\n\n"

            "These factors are associations identified by the "
            "model for this prediction and should not be "
            "interpreted as proving direct causation."
        )


    # ----------------------------------------------------
    # Risk factors
    # ----------------------------------------------------

    elif any(word in question_lower for word in [
        "risk factor",
        "higher risk",
        "factors",
        "explainability",
        "contribution"
    ]):

        higher_factors = [

            item["feature"]

            for item in verified_payload[
                "higher_risk_signals"
            ]
        ]

        lower_factors = [

            item["feature"]

            for item in verified_payload[
                "lower_risk_signals"
            ]
        ]

        return (
            "Factors associated with higher predicted churn "
            "risk:\n\n• "
            + "\n• ".join(higher_factors)
            + "\n\nFactors associated with lower predicted "
              "churn risk:\n\n• "
            + "\n• ".join(lower_factors)
        )


    # ----------------------------------------------------
    # Probability
    # ----------------------------------------------------

    elif any(word in question_lower for word in [
        "probability",
        "91.37",
        "what does",
        "prediction mean"
    ]):

        return (
            f"The model estimates a churn probability of "
            f"{verified_payload['churn_probability']:.2%}. "
            f"This is above the decision threshold of "
            f"{verified_payload['threshold']}, resulting in "
            f"a prediction of {verified_payload['prediction']} "
            f"and a {verified_payload['risk_level']} risk level. "
            f"The probability is an estimate, not a guarantee "
            f"that the customer will churn."
        )


    # ----------------------------------------------------
    # Model limitations
    # ----------------------------------------------------

    elif any(word in question_lower for word in [
        "limitation",
        "trust",
        "reliable",
        "accuracy",
        "limitations"
    ]):

        return PROJECT_KNOWLEDGE["limitations"]


    # ----------------------------------------------------
    # Default grounded response
    # ----------------------------------------------------

    else:

        return (
            "I can answer questions based on the verified "
            "customer prediction and the documented machine-"
            "learning decisions in this project.\n\n"

            "Try asking:\n\n"

            "• Why was Logistic Regression selected?\n"
            "• Why is recall important?\n"
            "• Why was the threshold set to 0.35?\n"
            "• Why is this customer High Risk?\n"
            "• Why not Random Forest?\n"
            "• What are the model limitations?"
        )
        
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
    # VERIFIED BUSINESS COMMUNICATION
    # ========================================================

    st.divider()

    st.subheader(
        "🤖 AI Business Communication & Safety Validation"
    )


    # --------------------------------
    # Build verified ML payload
    # --------------------------------

    verified_payload = build_verified_payload(
    customer_df
    )

    st.session_state["verified_payload"] = verified_payload


    # --------------------------------
    # Generate structured response
    # --------------------------------

    business_response = generate_business_response(
        verified_payload
    )


    # --------------------------------
    # Validate response
    # --------------------------------

    validation_result = validate_business_response(

        business_response,

        verified_payload

    )


    # --------------------------------
    # Display validation status
    # --------------------------------

    if validation_result["approved"]:

        st.success(
            "AI response approved — "
            "Verified ML output integrity check passed."
        )

    else:

        st.error(
            "AI response failed safety validation."
        )

        st.write(
            validation_result["violations"]
        )


    # --------------------------------
    # Display approved response
    # --------------------------------

    if validation_result["approved"]:

        st.markdown(
            "### Risk Summary"
        )

        st.write(
            business_response["risk_summary"]
        )


        st.markdown(
            "### Strongest Higher-Risk Factors"
        )

        for factor in business_response[
            "strongest_factors"
        ]:

            st.write(f"• {factor}")


        st.markdown(
            "### Lower-Risk Factors"
        )

        for factor in business_response[
            "lower_risk_factors"
        ]:

            st.write(f"• {factor}")


        st.markdown(
            "### Business Considerations"
        )

        st.write(
            business_response[
                "business_considerations"
            ]
        )

    # ========================================================
# INTERACTIVE DECISION-SUPPORT ASSISTANT
# ========================================================

st.divider()

st.subheader(
    "💬 Ask the Churn Decision-Support Assistant"
)

st.write(
    "Ask questions about the current prediction, model "
    "selection, threshold, recall, model comparison, or "
    "project methodology."
)

st.caption(
    "Example: Why was Logistic Regression selected?"
)


# --------------------------------------------------------
# Check whether a prediction exists
# --------------------------------------------------------

if "verified_payload" in st.session_state:

    question = st.chat_input(
        "Ask a question about the churn prediction or model..."
    )


    if question:

        # Display user question
        with st.chat_message("user"):

            st.write(question)


        # Generate grounded answer
        answer = answer_business_question(

            question,

            st.session_state["verified_payload"]

        )


        # Display assistant response
        with st.chat_message("assistant"):

            st.write(answer)


else:

    st.info(
        "Please generate a customer prediction first before "
        "asking questions."
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
