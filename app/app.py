import streamlit as st
import pandas as pd
import joblib
import os
from io import BytesIO
from fpdf import FPDF
from datetime import datetime
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
import seaborn as sns

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
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(BASE_DIR)
    model_path = os.path.join(PROJECT_DIR, "models", "churn_model.pkl")
    feature_map_path = os.path.join(PROJECT_DIR, "models", "feature_name_map.pkl")
    model = joblib.load(model_path)
    feature_name_map = joblib.load(feature_map_path)
    return model, feature_name_map

try:
    best_lr_model, FEATURE_NAME_MAP = load_resources()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ============================================================
# CONFIGURATION
# ============================================================
THRESHOLD = 0.35

# ============================================================
# PREDICTION FUNCTIONS
# ============================================================
def predict_customer(customer_data):
    churn_probability = best_lr_model.predict_proba(customer_data)[0][1]
    prediction = "Yes" if churn_probability >= THRESHOLD else "No"
    return {
        "churn_probability": float(churn_probability),
        "threshold": THRESHOLD,
        "prediction": prediction
    }

def classify_risk(churn_probability):
    if churn_probability >= 0.70:
        return "High"
    elif churn_probability >= THRESHOLD:
        return "Medium"
    else:
        return "Low"

def assess_customer(customer_df):
    prediction_result = predict_customer(customer_df)
    risk_level = classify_risk(prediction_result["churn_probability"])
    prediction_result["risk_level"] = risk_level
    return prediction_result

def explain_customer(customer_data, top_n=5):
    preprocessor = best_lr_model.named_steps["preprocessor"]
    model = best_lr_model.named_steps["model"]
    transformed_customer = preprocessor.transform(customer_data)
    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_[0]

    if hasattr(transformed_customer, "toarray"):
        transformed_customer = transformed_customer.toarray()

    values = transformed_customer[0]
    contributions = values * coefficients

    explanation_df = pd.DataFrame({
        "Feature": feature_names,
        "Value": values,
        "Coefficient": coefficients,
        "Contribution": contributions
    })

    explanation_df = explanation_df.sort_values("Contribution", ascending=False)
    higher_risk = explanation_df[explanation_df["Contribution"] > 0].head(top_n)
    lower_risk = explanation_df[explanation_df["Contribution"] < 0].sort_values(
        "Contribution", ascending=True
    ).head(top_n)

    return {
        "higher_risk_factors": higher_risk,
        "lower_risk_factors": lower_risk
    }

def clean_feature_names(explanation):
    higher_risk = explanation["higher_risk_factors"].copy()
    lower_risk = explanation["lower_risk_factors"].copy()
    higher_risk["Feature"] = higher_risk["Feature"].map(lambda x: FEATURE_NAME_MAP.get(x, x))
    lower_risk["Feature"] = lower_risk["Feature"].map(lambda x: FEATURE_NAME_MAP.get(x, x))
    return {
        "higher_risk_factors": higher_risk,
        "lower_risk_factors": lower_risk
    }

def build_verified_payload(customer_data):
    assessment = assess_customer(customer_data)
    explanation = explain_customer(customer_data)
    clean_explanation = clean_feature_names(explanation)

    higher_risk = []
    for _, row in clean_explanation["higher_risk_factors"].iterrows():
        higher_risk.append({
            "feature": row["Feature"],
            "contribution": round(float(row["Contribution"]), 3)
        })

    lower_risk = []
    for _, row in clean_explanation["lower_risk_factors"].iterrows():
        lower_risk.append({
            "feature": row["Feature"],
            "contribution": round(float(row["Contribution"]), 3)
        })

    return {
        "churn_probability": round(float(assessment["churn_probability"]), 4),
        "threshold": float(assessment["threshold"]),
        "prediction": assessment["prediction"],
        "risk_level": assessment["risk_level"],
        "higher_risk_signals": higher_risk,
        "lower_risk_signals": lower_risk
    }

def process_batch(df):
    required_columns = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges"
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    input_df = df[required_columns].copy()
    probabilities = best_lr_model.predict_proba(input_df)[:, 1]
    predictions = ["Yes" if p >= THRESHOLD else "No" for p in probabilities]
    risk_levels = [classify_risk(p) for p in probabilities]

    results = df.copy()
    results["Churn_Probability"] = [round(p, 4) for p in probabilities]
    results["Prediction"] = predictions
    results["Risk_Level"] = risk_levels
    return results

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Batch Predictions")
    output.seek(0)
    return output

def create_excel_report(assessment, clean_explanation, customer_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_data = {
            "Metric": ["Churn Probability", "Decision Threshold", "Prediction", "Risk Level", "Report Generated"],
            "Value": [
                f"{assessment['churn_probability']:.2%}",
                f"{assessment['threshold']:.2f}",
                assessment["prediction"],
                assessment["risk_level"],
                datetime.now(ZoneInfo("Africa/Nairobi")).strftime("%Y-%m-%d %H:%M:%S")
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Prediction Summary", index=False)
        customer_df.T.reset_index().rename(columns={"index": "Feature", 0: "Value"}).to_excel(
            writer, sheet_name="Customer Inputs", index=False
        )
        clean_explanation["higher_risk_factors"][["Feature", "Contribution"]].to_excel(
            writer, sheet_name="Higher Risk Factors", index=False
        )
        clean_explanation["lower_risk_factors"][["Feature", "Contribution"]].to_excel(
            writer, sheet_name="Lower Risk Factors", index=False
        )
    output.seek(0)
    return output

def create_pdf_report(assessment, clean_explanation, customer_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Telco Customer Churn Prediction Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated on: {datetime.now(ZoneInfo('Africa/Nairobi')).strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Prediction Results", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Churn Probability : {assessment['churn_probability']:.2%}", ln=True)
    pdf.cell(0, 8, f"Decision Threshold: {assessment['threshold']:.2f}", ln=True)
    pdf.cell(0, 8, f"Prediction        : {assessment['prediction']}", ln=True)
    pdf.cell(0, 8, f"Risk Level        : {assessment['risk_level']}", ln=True)
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Higher-Risk Factors", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for _, row in clean_explanation["higher_risk_factors"].iterrows():
        pdf.cell(0, 7, f"- {row['Feature']}: {row['Contribution']:.3f}", ln=True)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Lower-Risk Factors", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for _, row in clean_explanation["lower_risk_factors"].iterrows():
        pdf.cell(0, 7, f"- {row['Feature']}: {row['Contribution']:.3f}", ln=True)

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 6,
        "Disclaimer: This report is generated by a machine learning model. "
        "The final business decision remains with a human decision-maker."
    )
    return bytes(pdf.output())

# ============================================================
# PROJECT KNOWLEDGE BASE (kept as-is)
# ============================================================
PROJECT_KNOWLEDGE = {
    "project_overview": {
        "name": "Telco Customer Churn AI",
        "objective": "Predict whether a telecom customer is likely to churn using machine learning and provide interpretable customer-level explanations.",
        "target": "Churn",
        "problem_type": "Binary Classification",
        "classes": ["Yes", "No"],
        "positive_class": "Yes"
    },
    "dataset": {
        "name": "Telco Customer Churn Dataset",
        "total_customers": 7043,
        "original_features": 21,
        "predictor_features": 19,
        "customer_id_used": False,
        "customer_id_reason": "CustomerID was excluded because it is an identifier rather than a meaningful predictive feature.",
        "churn_distribution": {"No": 5174, "Yes": 1869},
        "churn_percentage": 26.537,
        "non_churn_percentage": 73.463,
        "class_imbalance": "The dataset is imbalanced because non-churners are more common than churners."
    },
    "business_objective": {
        "primary_goal": "Identify customers at risk of churn.",
        "priority_metric": "Recall",
        "reason_for_recall": "The business objective prioritizes identifying actual churners because missing a customer who is likely to churn was considered more costly than unnecessarily flagging some customers.",
        "false_negative": "A false negative occurs when the model predicts that a customer will not churn when the customer actually churns."
    },
    "preprocessing": {
        "numeric_features": ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"],
        "categorical_features": [
            "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"
        ],
        "categorical_encoding": "One-Hot Encoding",
        "scaling": "StandardScaler",
        "total_charges_handling": "TotalCharges was converted to a numeric variable. Blank values associated with new customers were treated as zero accumulated charges.",
        "pipeline": "Preprocessing and modeling were combined using a Scikit-learn Pipeline to support consistent transformations and reduce the risk of data leakage."
    },
    "data_split": {
        "training_percentage": 80,
        "testing_percentage": 20,
        "stratification": True,
        "reason_for_stratification": "Stratification was used to preserve the churn class distribution across the training and testing datasets.",
        "training_shape": "(5634, 19)",
        "testing_shape": "(1409, 19)"
    },
    "models_evaluated": ["Logistic Regression", "Decision Tree", "Random Forest", "K-Nearest Neighbors"],
    "baseline_models": {
        "Logistic Regression": {"accuracy": 0.8055, "precision": 0.6572, "recall": 0.5588, "f1_score": 0.6040, "roc_auc": 0.8421},
        "Decision Tree": {"accuracy": 0.7211, "precision": 0.4755, "recall": 0.4920, "f1_score": 0.4836, "roc_auc": 0.6477},
        "Random Forest": {"accuracy": 0.7864, "precision": 0.6254, "recall": 0.4866, "f1_score": 0.5474, "roc_auc": 0.8185},
        "KNN": {"accuracy": 0.7637, "precision": 0.5527, "recall": 0.5749, "f1_score": 0.5636, "roc_auc": 0.7896}
    },
    "final_model": {
        "name": "Logistic Regression",
        "parameters": {"C": 1, "class_weight": "balanced", "max_iter": 1000, "random_state": 42},
        "selection_reason": "Logistic Regression was selected because it aligned well with the business objective of prioritizing recall, provided strong discrimination performance, and offered clear coefficient-based interpretability for customer-level explanations.",
        "interpretability": "Logistic Regression coefficients support analysis of how transformed customer features contribute to predictions."
    },
    "decision_threshold": {
        "selected_threshold": 0.35,
        "default_threshold": 0.50,
        "reason": "Threshold analysis was performed because the default 0.50 threshold was not automatically assumed to be optimal for the business objective. The selected threshold balanced the goal of identifying more potential churners with the trade-off of additional false positives.",
        "lower_threshold_effect": "Lowering the threshold generally increases the number of customers classified as potential churners, which can increase recall while reducing precision.",
        "higher_threshold_effect": "Increasing the threshold generally makes the model more conservative, which may improve precision while reducing recall."
    },
    "metrics": {
        "accuracy": "Accuracy measures the overall proportion of correct predictions.",
        "precision": "Precision measures how many customers predicted as churners were actually churners.",
        "recall": "Recall measures how many actual churners were correctly identified by the model.",
        "f1_score": "The F1 score combines precision and recall into a single metric.",
        "roc_auc": "ROC-AUC measures the model's ability to distinguish between churners and non-churners across different thresholds.",
        "confusion_matrix": "A confusion matrix summarizes predictions into true positives, true negatives, false positives, and false negatives."
    },
    "explainability": {
        "method": "Customer-level explanations are generated using the fitted Logistic Regression coefficients and the customer's transformed feature values.",
        "positive_contribution": "A positive contribution is associated with increasing the model's predicted churn risk for that customer.",
        "negative_contribution": "A negative contribution is associated with lowering the model's predicted churn risk for that customer.",
        "causation_warning": "Feature contributions represent model associations and do not prove direct causation."
    },
    "business_insights": {
        "tenure": "Customers with longer tenure generally showed lower churn, while shorter-tenure customers were more associated with churn.",
        "monthly_charges": "Higher MonthlyCharges were associated with higher churn patterns in the exploratory analysis.",
        "month_to_month": "Month-to-month contracts showed higher observed churn than longer-term contracts.",
        "fiber_optic": "Fiber optic customers showed the highest observed churn rate among the InternetService categories.",
        "electronic_check": "Electronic check had the highest observed churn rate among the payment methods analyzed.",
        "paperless_billing": "Customers using PaperlessBilling showed a higher observed churn rate than customers not using PaperlessBilling."
    },
    "limitations": {
        "prediction_not_guarantee": "The model provides probability-based predictions and cannot guarantee future customer behavior.",
        "association_not_causation": "Patterns and feature contributions represent associations identified by the model and should not be interpreted as proof of causation.",
        "human_decision": "The application is a decision-support system. Final business decisions remain with human decision-makers.",
        "data_dependency": "Model performance depends on the quality, relevance, and representativeness of the available training data."
    },
    "ai_safety": {
        "ml_integrity": "The AI communication layer must not alter verified ML outputs such as churn probability, threshold, prediction, risk level, or verified customer-level factors.",
        "no_invention": "The AI should not invent customer information, model results, or undocumented business facts.",
        "human_oversight": "The final business decision remains with a human decision-maker."
    }
}

# ============================================================
# AI ASSISTANT
# ============================================================
def answer_project_question(question, verified_payload=None):
    q = question.lower().strip()

    if any(w in q for w in ["what is this project", "project about", "project objective", "what does this project do"]):
        return f"This project is called {PROJECT_KNOWLEDGE['project_overview']['name']}. Its objective is to {PROJECT_KNOWLEDGE['project_overview']['objective']}"

    if any(w in q for w in ["business problem", "business objective", "main goal", "goal of the project"]):
        return PROJECT_KNOWLEDGE["business_objective"]["primary_goal"]

    if any(w in q for w in ["why logistic", "why choose logistic", "why was logistic regression selected", "why is logistic regression the final model"]):
        return PROJECT_KNOWLEDGE["final_model"]["selection_reason"]

    if any(w in q for w in ["final model", "which model", "what model are you using", "what model was selected"]):
        return f"The final model is {PROJECT_KNOWLEDGE['final_model']['name']}."

    if any(w in q for w in ["why recall", "why is recall important", "why prioritize recall"]):
        return PROJECT_KNOWLEDGE["business_objective"]["reason_for_recall"]

    if any(w in q for w in ["false negative", "what is fn", "what are false negatives"]):
        return PROJECT_KNOWLEDGE["business_objective"]["false_negative"]

    if any(w in q for w in ["why 0.35", "why threshold", "why did you choose the threshold", "decision threshold"]):
        return PROJECT_KNOWLEDGE["decision_threshold"]["reason"]
    if any(w in q for w in ["lower the threshold", "what happens if threshold is lower"]):
        return PROJECT_KNOWLEDGE["decision_threshold"]["lower_threshold_effect"]
    if any(w in q for w in ["increase the threshold", "higher threshold", "raise the threshold"]):
        return PROJECT_KNOWLEDGE["decision_threshold"]["higher_threshold_effect"]

    if any(w in q for w in ["how many customers", "dataset size", "how many records"]):
        return f"The dataset contains {PROJECT_KNOWLEDGE['dataset']['total_customers']} customer records."
    if any(w in q for w in ["imbalanced", "class imbalance", "dataset balanced"]):
        return PROJECT_KNOWLEDGE["dataset"]["class_imbalance"]
    if any(w in q for w in ["customerid", "customer id", "why remove customer id"]):
        return PROJECT_KNOWLEDGE["dataset"]["customer_id_reason"]

    if any(w in q for w in ["preprocessing", "how was the data prepared", "how did you prepare the data"]):
        return "The project used One-Hot Encoding for categorical features, StandardScaler for numerical features, and a Scikit-learn Pipeline to combine preprocessing and modeling."
    if any(w in q for w in ["one hot encoding", "categorical variables", "categorical features"]):
        return "One-Hot Encoding was used to convert categorical values into numerical indicator variables suitable for machine learning."
    if any(w in q for w in ["standardscaler", "scaling", "why scale"]):
        return "StandardScaler was used to standardize numerical features within the preprocessing pipeline."
    if any(w in q for w in ["totalcharges", "total charges missing", "missing values"]):
        return PROJECT_KNOWLEDGE["preprocessing"]["total_charges_handling"]

    if any(w in q for w in ["train test split", "80 20", "training data", "testing data"]):
        return "The dataset was split into approximately 80% training data and 20% testing data."
    if any(w in q for w in ["stratify", "stratification"]):
        return PROJECT_KNOWLEDGE["data_split"]["reason_for_stratification"]
    if "data leakage" in q:
        return "Data leakage occurs when information that would not be available during real-world prediction improperly influences model training or evaluation."

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

    if any(w in q for w in ["how do you explain", "explainability", "feature contribution"]):
        return PROJECT_KNOWLEDGE["explainability"]["method"]
    if any(w in q for w in ["positive contribution", "positive factor"]):
        return PROJECT_KNOWLEDGE["explainability"]["positive_contribution"]
    if any(w in q for w in ["negative contribution", "negative factor"]):
        return PROJECT_KNOWLEDGE["explainability"]["negative_contribution"]
    if any(w in q for w in ["causation", "cause churn", "prove cause"]):
        return PROJECT_KNOWLEDGE["explainability"]["causation_warning"]

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

    if any(w in q for w in ["definitely churn", "guarantee churn", "certain to churn"]):
        return PROJECT_KNOWLEDGE["limitations"]["prediction_not_guarantee"]
    if any(w in q for w in ["final decision", "can ai decide", "can the model decide"]):
        return PROJECT_KNOWLEDGE["limitations"]["human_decision"]
    if any(w in q for w in ["can ai change", "change prediction", "alter prediction"]):
        return PROJECT_KNOWLEDGE["ai_safety"]["ml_integrity"]

    return (
        "I can answer questions about this Telco Customer Churn project, including the dataset, "
        "preprocessing, models, threshold, evaluation metrics, explainability, customer predictions, "
        "business insights, limitations, and AI safety. Please ask your question in relation to one of these areas."
    )

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("📊 Telco Churn AI")
    st.markdown("---")
    st.markdown("### About")
    st.write(
        "Predict telecom customer churn using Logistic Regression "
        "and get clear, interpretable explanations."
    )
    st.markdown("---")
    st.markdown("### Sections")
    st.markdown("""
    - 🔍 Single Prediction  
    - 📂 Batch Prediction  
    - 📈 Model Performance  
    - 💬 AI Assistant  
    """)
    st.markdown("---")
    st.caption("Final decisions should always be made by a human.")

# ============================================================
# MAIN TITLE
# ============================================================
st.title("📊 Telco Customer Churn Prediction System")
st.write("Enter customer information below to estimate churn risk and view the factors associated with the prediction.")

# ============================================================
# 1. SINGLE CUSTOMER PREDICTION
# ============================================================
st.markdown("---")
st.header("🔍 Single Customer Prediction")

st.subheader("Customer Information")
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12, help="How long the customer has been with the company")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

with col2:
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

with col3:
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

st.markdown("")
if st.button("🚀 Predict Customer Churn", type="primary", use_container_width=True):
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
    explanation = explain_customer(customer_df)
    clean_explanation = clean_feature_names(explanation)

    st.markdown("---")
    st.subheader("Prediction Results")

    # Big probability + progress
    prob = assessment["churn_probability"]
    st.markdown(f"### Churn Probability: `{prob:.1%}`")
    st.progress(min(prob, 1.0))

    c1, c2, c3 = st.columns(3)
    c1.metric("Decision Threshold", f"{assessment['threshold']:.2f}")
    c2.metric("Prediction", assessment["prediction"])
    c3.metric("Risk Level", assessment["risk_level"])

    if assessment["risk_level"] == "High":
        st.error("🔴 High predicted churn risk")
    elif assessment["risk_level"] == "Medium":
        st.warning("🟠 Medium predicted churn risk")
    else:
        st.success("🟢 Low predicted churn risk")

    # Risk factors
    st.markdown("---")
    st.subheader("Customer-Level Risk Factors")
    f1, f2 = st.columns(2)

    with f1:
        st.markdown("#### 🔴 Higher-Risk Factors")
        higher = clean_explanation["higher_risk_factors"][["Feature", "Contribution"]].copy()
        higher["Contribution"] = higher["Contribution"].round(3)
        st.dataframe(higher, use_container_width=True, hide_index=True)

    with f2:
        st.markdown("#### 🟢 Lower-Risk Factors")
        lower = clean_explanation["lower_risk_factors"][["Feature", "Contribution"]].copy()
        lower["Contribution"] = lower["Contribution"].round(3)
        st.dataframe(lower, use_container_width=True, hide_index=True)

    # Store for AI assistant
    st.session_state["verified_payload"] = build_verified_payload(customer_df)

    # Download reports
    st.markdown("---")
    st.subheader("📥 Download Report")
    col_excel, col_pdf = st.columns(2)

    with col_excel:
        excel_file = create_excel_report(assessment, clean_explanation, customer_df)
        st.download_button(
            label="Download Excel Report",
            data=excel_file,
            file_name=f"churn_report_{datetime.now(ZoneInfo('Africa/Nairobi')).strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with col_pdf:
        pdf_bytes = create_pdf_report(assessment, clean_explanation, customer_df)
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"churn_report_{datetime.now(ZoneInfo('Africa/Nairobi')).strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )

# ============================================================
# 2. BATCH PREDICTION
# ============================================================
st.markdown("---")
st.header("📂 Batch Prediction")

st.write("Upload a CSV file containing multiple customers to get predictions for all of them at once.")

sample_csv = """gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges
Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85
Male,0,No,No,34,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,No,Mailed check,56.95,1889.5
Male,0,No,No,2,Yes,No,DSL,Yes,Yes,No,No,No,No,Month-to-month,Yes,Mailed check,53.85,108.15
Male,0,No,No,45,No,No phone service,DSL,Yes,No,Yes,Yes,No,No,One year,No,Bank transfer (automatic),42.3,1840.75
Female,0,No,No,2,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,70.7,151.65
"""

st.download_button(
    label="📥 Download Sample CSV Template",
    data=sample_csv,
    file_name="sample_customers.csv",
    mime="text/csv"
)

with st.expander("View required columns"):
    st.code(
        "gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, "
        "MultipleLines, InternetService, OnlineSecurity, OnlineBackup, "
        "DeviceProtection, TechSupport, StreamingTV, StreamingMovies, "
        "Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges"
    )

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.write("### Preview of uploaded data")
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("Run Batch Prediction", type="primary"):
            with st.spinner("Running predictions..."):
                results_df = process_batch(batch_df)

            st.success(f"Successfully processed {len(results_df)} customers!")
            st.write("### Prediction Results")
            st.dataframe(results_df, use_container_width=True)

            st.write("### Summary")
            s1, s2, s3 = st.columns(3)
            s1.metric("Total Customers", len(results_df))
            s2.metric("Predicted Churners", (results_df["Prediction"] == "Yes").sum())
            s3.metric("High Risk Customers", (results_df["Risk_Level"] == "High").sum())

            excel_data = convert_df_to_excel(results_df)
            st.download_button(
                label="📥 Download Batch Results (Excel)",
                data=excel_data,
                file_name=f"batch_churn_predictions_{datetime.now(ZoneInfo('Africa/Nairobi')).strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except Exception as e:
        st.error(f"Error processing file: {e}")

# ============================================================
# 3. MODEL PERFORMANCE DASHBOARD
# ============================================================
st.markdown("---")
st.header("📈 Model Performance Dashboard")

st.write(
    "Performance of the final **Logistic Regression** model on the held-out test set "
    "(after hyperparameter tuning and threshold selection)."
)

st.markdown("### Key Metrics")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", "73.81%")
m2.metric("Precision", "50.43%")
m3.metric("Recall", "78.34%")
m4.metric("F1-Score", "61.36%")
m5.metric("ROC-AUC", "84.16%")

st.caption("Final model (C=1, class_weight='balanced') evaluated with threshold = 0.35")

st.markdown("### Confusion Matrix (Test Set)")

cm_data = pd.DataFrame(
    [[850, 183], [81, 295]],
    index=["Actual: No", "Actual: Yes"],
    columns=["Predicted: No", "Predicted: Yes"]
)

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
ax.set_title("Confusion Matrix - Logistic Regression")
st.pyplot(fig)
plt.close()

st.markdown("""
**How to read it:**
- **True Negatives (850)**: Correctly predicted non-churners  
- **False Positives (183)**: Predicted churn but customer stayed  
- **False Negatives (81)**: Missed actual churners  
- **True Positives (295)**: Correctly predicted churners  
""")

with st.expander("What do these metrics mean?"):
    st.markdown("""
    - **Accuracy**: Overall percentage of correct predictions  
    - **Precision**: Of all customers predicted as churners, how many actually churned  
    - **Recall**: Of all actual churners, how many the model successfully identified  
    - **F1-Score**: Balance between Precision and Recall  
    - **ROC-AUC**: Ability of the model to distinguish between churners and non-churners  

    **Why we prioritized Recall:**  
    Missing a customer who is about to churn (False Negative) is more costly than incorrectly flagging a loyal customer.
    """)

st.markdown("### Model Comparison (Baseline Results)")

comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "KNN"],
    "Accuracy": [0.8055, 0.7211, 0.7864, 0.7637],
    "Precision": [0.6572, 0.4755, 0.6254, 0.5527],
    "Recall": [0.5588, 0.4920, 0.4866, 0.5749],
    "F1-Score": [0.6040, 0.4836, 0.5474, 0.5636],
    "ROC-AUC": [0.8421, 0.6477, 0.8185, 0.7896]
})

st.dataframe(
    comparison_df.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1-Score": "{:.2%}",
        "ROC-AUC": "{:.2%}"
    }),
    use_container_width=True,
    hide_index=True
)

st.info(
    "After hyperparameter tuning, **Logistic Regression** was selected as the final model "
    "because it offered the best combination of high Recall, strong ROC-AUC, and interpretability."
)

# ============================================================
# 4. AI ASSISTANT
# ============================================================
st.markdown("---")
st.header("💬 Ask About This Project")

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
    st.markdown("#### Answer")
    st.success(answer)

# ============================================================
# DISCLAIMER
# ============================================================
st.markdown("---")
st.caption(
    "This application provides a machine-learning prediction based on customer information. "
    "The final business decision remains with a human decision-maker."
)
