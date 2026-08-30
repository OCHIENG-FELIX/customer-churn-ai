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
# ============================================================
# PROJECT KNOWLEDGE BASE  (keep your existing PROJECT_KNOWLEDGE dict)
# ============================================================
# ... keep the big PROJECT_KNOWLEDGE dictionary exactly as you have it ...


# ============================================================
# UNIFIED PROJECT Q&A FUNCTION
# ============================================================
def answer_project_question(question, verified_payload=None):
    q = question.lower().strip()

    # ---------- Project overview ----------
    if any(w in q for w in ["what is this project", "project about", "project objective", "what does this project do"]):
        return (
            f"This project is called {PROJECT_KNOWLEDGE['project_overview']['name']}. "
            f"Its objective is to {PROJECT_KNOWLEDGE['project_overview']['objective']}"
        )

    # ---------- Business objective ----------
    if any(w in q for w in ["business problem", "business objective", "main goal", "goal of the project"]):
        return PROJECT_KNOWLEDGE["business_objective"]["primary_goal"]

    # ---------- Why Logistic Regression ----------
    if any(w in q for w in ["why logistic", "why choose logistic", "why was logistic regression selected", "why is logistic regression the final model"]):
        return PROJECT_KNOWLEDGE["final_model"]["selection_reason"]

    # ---------- Final model ----------
    if any(w in q for w in ["final model", "which model", "what model are you using", "what model was selected"]):
        return f"The final model is {PROJECT_KNOWLEDGE['final_model']['name']}."

    # ---------- Recall ----------
    if any(w in q for w in ["why recall", "why is recall important", "why prioritize recall"]):
        return PROJECT_KNOWLEDGE["business_objective"]["reason_for_recall"]

    # ---------- False negatives ----------
    if any(w in q for w in ["false negative", "what is fn", "what are false negatives"]):
        return PROJECT_KNOWLEDGE["business_objective"]["false_negative"]

    # ---------- Threshold ----------
    if any(w in q for w in ["why 0.35", "why threshold", "why did you choose the threshold", "decision threshold"]):
        return PROJECT_KNOWLEDGE["decision_threshold"]["reason"]

    if any(w in q for w in ["lower the threshold", "what happens if threshold is lower"]):
        return PROJECT_KNOWLEDGE["decision_threshold"]["lower_threshold_effect"]

    if any(w in q for w in ["increase the threshold", "higher threshold", "raise the threshold"]):
        return PROJECT_KNOWLEDGE["decision_threshold"]["higher_threshold_effect"]

    # ---------- Dataset ----------
    if any(w in q for w in ["how many customers", "dataset size", "how many records"]):
        return f"The dataset contains {PROJECT_KNOWLEDGE['dataset']['total_customers']} customer records."

    if any(w in q for w in ["imbalanced", "class imbalance", "dataset balanced"]):
        return PROJECT_KNOWLEDGE["dataset"]["class_imbalance"]

    if any(w in q for w in ["customerid", "customer id", "why remove customer id"]):
        return PROJECT_KNOWLEDGE["dataset"]["customer_id_reason"]

    # ---------- Preprocessing ----------
    if any(w in q for w in ["preprocessing", "how was the data prepared", "how did you prepare the data"]):
        return (
            "The project used One-Hot Encoding for categorical features, "
            "StandardScaler for numerical features, and a Scikit-learn Pipeline "
            "to combine preprocessing and modeling."
        )

    if any(w in q for w in ["one hot encoding", "categorical variables", "categorical features"]):
        return "One-Hot Encoding was used to convert categorical values into numerical indicator variables suitable for machine learning."

    if any(w in q for w in ["standardscaler", "scaling", "why scale"]):
        return "StandardScaler was used to standardize numerical features within the preprocessing pipeline."

    if any(w in q for w in ["totalcharges", "total charges missing", "missing values"]):
        return PROJECT_KNOWLEDGE["preprocessing"]["total_charges_handling"]

    # ---------- Train/test split ----------
    if any(w in q for w in ["train test split", "80 20", "training data", "testing data"]):
        return "The dataset was split into approximately 80% training data and 20% testing data."

    if any(w in q for w in ["stratify", "stratification"]):
        return PROJECT_KNOWLEDGE["data_split"]["reason_for_stratification"]

    if "data leakage" in q:
        return "Data leakage occurs when information that would not be available during real-world prediction improperly influences model training or evaluation."

    # ---------- Metrics ----------
    if "precision" in q:
        return PROJECT_KNOWLEDGE["metrics"]["precision"]
    if "recall" in q:
        return PROJECT_KNOWLEDGE["metrics"]["recall"]
    if "f1" in q:
        return PROJECT_KNOWLEDGE["metrics"]["f1_score"]
    if any(w in q for w in ["roc auc", "roc-auc", "auc"]):
        return PROJECT_KNOWLEDGE["metrics"]["roc_auc"]
    if "accuracy" in q:
        return PROJECT_KNOWLEDGE["metrics"]["accuracy"]
    if "confusion matrix" in q:
        return PROJECT_KNOWLEDGE["metrics"]["confusion_matrix"]

    # ---------- Explainability ----------
    if any(w in q for w in ["how do you explain", "explainability", "feature contribution"]):
        return PROJECT_KNOWLEDGE["explainability"]["method"]
    if any(w in q for w in ["positive contribution", "positive factor"]):
        return PROJECT_KNOWLEDGE["explainability"]["positive_contribution"]
    if any(w in q for w in ["negative contribution", "negative factor"]):
        return PROJECT_KNOWLEDGE["explainability"]["negative_contribution"]
    if any(w in q for w in ["causation", "cause churn", "prove cause"]):
        return PROJECT_KNOWLEDGE["explainability"]["causation_warning"]

    # ---------- Business insights ----------
    if "tenure" in q:
        return PROJECT_KNOWLEDGE["business_insights"]["tenure"]
    if any(w in q for w in ["month-to-month", "month to month", "contract"]):
        return PROJECT_KNOWLEDGE["business_insights"]["month_to_month"]
    if any(w in q for w in ["fiber optic", "internet service"]):
        return PROJECT_KNOWLEDGE["business_insights"]["fiber_optic"]
    if any(w in q for w in ["electronic check", "payment method"]):
        return PROJECT_KNOWLEDGE["business_insights"]["electronic_check"]
    if any(w in q for w in ["monthly charges", "monthlycharges"]):
        return PROJECT_KNOWLEDGE["business_insights"]["monthly_charges"]

    # ---------- Current customer questions (only if prediction exists) ----------
    if verified_payload is not None:
        if any(w in q for w in ["why is this customer", "why predicted", "why churn", "why is the customer high risk", "why high risk"]):
            factors = [item["feature"] for item in verified_payload["higher_risk_signals"]]
            return (
                f"The model estimates a churn probability of {verified_payload['churn_probability']:.2%}, "
                f"which is compared with the {verified_payload['threshold']:.2f} decision threshold. "
                f"The resulting prediction is {verified_payload['prediction']} with a {verified_payload['risk_level']} risk level. "
                f"The strongest higher-risk signals are: {', '.join(factors)}. "
                f"These represent model associations rather than proof of causation."
            )

        if any(w in q for w in ["risk factors", "strongest factors", "higher risk factors"]):
            factors = [item["feature"] for item in verified_payload["higher_risk_signals"]]
            return "The strongest higher-risk factors identified for this prediction are: " + ", ".join(factors) + "."

        if any(w in q for w in ["lower risk factors", "lower-risk factors", "protective factors"]):
            factors = [item["feature"] for item in verified_payload["lower_risk_signals"]]
            return "The lower-risk factors identified for this prediction are: " + ", ".join(factors) + "."

        if any(w in q for w in ["what is the probability", "churn probability", "prediction probability"]):
            return f"The verified churn probability for the current customer is {verified_payload['churn_probability']:.2%}."

    # ---------- Limitations & Safety ----------
    if any(w in q for w in ["definitely churn", "guarantee churn", "certain to churn"]):
        return PROJECT_KNOWLEDGE["limitations"]["prediction_not_guarantee"]
    if any(w in q for w in ["final decision", "can ai decide", "can the model decide"]):
        return PROJECT_KNOWLEDGE["limitations"]["human_decision"]
    if any(w in q for w in ["can ai change", "change prediction", "alter prediction"]):
        return PROJECT_KNOWLEDGE["ai_safety"]["ml_integrity"]

    # ---------- Fallback ----------
    return (
        "I can answer questions about this Telco Customer Churn project, including the dataset, "
        "preprocessing, exploratory analysis, models tested, Logistic Regression selection, recall, "
        "threshold selection, evaluation metrics, explainability, customer predictions, business insights, "
        "limitations, and AI safety. Please ask your question in relation to one of these areas."
    )


# ============================================================
# STREAMLIT UI – PREDICTION + ASSISTANT
# ============================================================

st.title("📊 Telco Customer Churn Prediction System")
st.write("Enter customer information below to estimate churn risk and view the factors associated with the prediction.")
st.divider()

# ---- Customer inputs (keep your existing col1, col2, col3 code) ----
# ... your existing input widgets ...

st.divider()

if st.button("Predict Customer Churn", type="primary"):
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

    assessment = assess_customer(customer_df)

    st.divider()
    st.subheader("Prediction Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churn Probability", f"{assessment['churn_probability']:.2%}")
    c2.metric("Decision Threshold", f"{assessment['threshold']:.2f}")
    c3.metric("Prediction", assessment["prediction"])
    c4.metric("Risk Level", assessment["risk_level"])

    if assessment["risk_level"] == "High":
        st.error("High predicted churn risk.")
    elif assessment["risk_level"] == "Medium":
        st.warning("Medium predicted churn risk.")
    else:
        st.success("Low predicted churn risk.")

    # Risk factors
    explanation = explain_customer(customer_df)
    clean_explanation = clean_feature_names(explanation)

    st.divider()
    st.subheader("Customer-Level Risk Factors")
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("### 🔴 Higher-Risk Factors")
        st.dataframe(clean_explanation["higher_risk_factors"][["Feature", "Contribution"]], use_container_width=True, hide_index=True)
    with f2:
        st.markdown("### 🟢 Lower-Risk Factors")
        st.dataframe(clean_explanation["lower_risk_factors"][["Feature", "Contribution"]], use_container_width=True, hide_index=True)

    # Build and store verified payload
    verified_payload = build_verified_payload(customer_df)
    st.session_state["verified_payload"] = verified_payload

    # Optional: show the business response + validation (keep your existing code if you want)
    # ...


# ============================================================
# AI ASSISTANT SECTION
# ============================================================
st.markdown("---")
st.subheader("💬 Ask About This Project")

st.write(
    "Ask any question about the Telco Customer Churn project "
    "(dataset, models, threshold, explainability, current prediction, limitations, etc.)."
)

user_question = st.text_input(
    "Type your question here",
    placeholder="Example: Why was Logistic Regression selected?  or  Why is this customer high risk?"
)

if user_question:
    payload = st.session_state.get("verified_payload", None)
    answer = answer_project_question(user_question, verified_payload=payload)
    st.info(answer)

# Disclaimer
st.divider()
st.caption(
    "This application provides a machine-learning prediction based on customer information. "
    "The final business decision remains with a human decision-maker."
)
