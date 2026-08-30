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

    # ============================================================
    # PROJECT OVERVIEW
    # ============================================================

    "project_overview": {
        "name": "Telco Customer Churn AI",
        "objective": (
            "Predict whether a telecom customer is likely to churn "
            "using machine learning and provide interpretable "
            "customer-level explanations."
        ),
        "target": "Churn",
        "problem_type": "Binary Classification",
        "classes": ["Yes", "No"],
        "positive_class": "Yes"
    },


    # ============================================================
    # DATASET
    # ============================================================

    "dataset": {
        "name": "Telco Customer Churn Dataset",
        "total_customers": 7043,
        "original_features": 21,
        "predictor_features": 19,

        "customer_id_used": False,

        "customer_id_reason": (
            "CustomerID was excluded because it is an identifier "
            "rather than a meaningful predictive feature."
        ),

        "churn_distribution": {
            "No": 5174,
            "Yes": 1869
        },

        "churn_percentage": 26.537,
        "non_churn_percentage": 73.463,

        "class_imbalance": (
            "The dataset is imbalanced because non-churners are "
            "more common than churners."
        )
    },


    # ============================================================
    # BUSINESS OBJECTIVE
    # ============================================================

    "business_objective": {
        "primary_goal": (
            "Identify customers at risk of churn."
        ),

        "priority_metric": "Recall",

        "reason_for_recall": (
            "The business objective prioritizes identifying actual "
            "churners because missing a customer who is likely to "
            "churn was considered more costly than unnecessarily "
            "flagging some customers."
        ),

        "false_negative": (
            "A false negative occurs when the model predicts that "
            "a customer will not churn when the customer actually churns."
        )
    },


    # ============================================================
    # DATA PREPROCESSING
    # ============================================================

    "preprocessing": {

        "numeric_features": [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ],

        "categorical_features": [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod"
        ],

        "categorical_encoding": "One-Hot Encoding",

        "scaling": "StandardScaler",

        "total_charges_handling": (
            "TotalCharges was converted to a numeric variable. "
            "Blank values associated with new customers were treated "
            "as zero accumulated charges."
        ),

        "pipeline": (
            "Preprocessing and modeling were combined using a "
            "Scikit-learn Pipeline to support consistent transformations "
            "and reduce the risk of data leakage."
        )
    },


    # ============================================================
    # TRAIN / TEST SPLIT
    # ============================================================

    "data_split": {
        "training_percentage": 80,
        "testing_percentage": 20,

        "stratification": True,

        "reason_for_stratification": (
            "Stratification was used to preserve the churn class "
            "distribution across the training and testing datasets."
        ),

        "training_shape": "(5634, 19)",
        "testing_shape": "(1409, 19)"
    },


    # ============================================================
    # MODELS EVALUATED
    # ============================================================

    "models_evaluated": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "K-Nearest Neighbors"
    ],


    # ============================================================
    # BASELINE MODEL RESULTS
    # ============================================================

    "baseline_models": {

        "Logistic Regression": {
            "accuracy": 0.8055,
            "precision": 0.6572,
            "recall": 0.5588,
            "f1_score": 0.6040,
            "roc_auc": 0.8421
        },

        "Decision Tree": {
            "accuracy": 0.7211,
            "precision": 0.4755,
            "recall": 0.4920,
            "f1_score": 0.4836,
            "roc_auc": 0.6477
        },

        "Random Forest": {
            "accuracy": 0.7864,
            "precision": 0.6254,
            "recall": 0.4866,
            "f1_score": 0.5474,
            "roc_auc": 0.8185
        },

        "KNN": {
            "accuracy": 0.7637,
            "precision": 0.5527,
            "recall": 0.5749,
            "f1_score": 0.5636,
            "roc_auc": 0.7896
        }
    },


    # ============================================================
    # MODEL VALIDATION
    # ============================================================

    "cross_validation": {
        "method": "5-Fold Cross-Validation",

        "purpose": (
            "Cross-validation was used to evaluate model performance "
            "across multiple splits of the training data."
        )
    },


    # ============================================================
    # HYPERPARAMETER TUNING
    # ============================================================

    "tuning": {

        "Logistic Regression": {
            "best_parameters": {
                "C": 1,
                "class_weight": "balanced"
            },

            "best_cv_recall": 0.8040,

            "test_results": {
                "accuracy": 0.7381,
                "precision": 0.5043,
                "recall": 0.7834,
                "f1_score": 0.6136,
                "roc_auc": 0.8416
            }
        },

        "KNN": {
            "best_parameters": {
                "n_neighbors": 21,
                "p": 1,
                "weights": "uniform"
            },

            "best_cv_recall": 0.5940
        },

        "Random Forest": {
            "best_parameters": {
                "class_weight": "balanced",
                "max_depth": 10,
                "min_samples_split": 10,
                "n_estimators": 100
            },

            "best_cv_recall": 0.7324
        }
    },


    # ============================================================
    # FINAL MODEL
    # ============================================================

    "final_model": {

        "name": "Logistic Regression",

        "parameters": {
            "C": 1,
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42
        },

        "selection_reason": (
            "Logistic Regression was selected because it aligned "
            "well with the business objective of prioritizing recall, "
            "provided strong discrimination performance, and offered "
            "clear coefficient-based interpretability for customer-level "
            "explanations."
        ),

        "interpretability": (
            "Logistic Regression coefficients support analysis of "
            "how transformed customer features contribute to predictions."
        )
    },


    # ============================================================
    # THRESHOLD
    # ============================================================

    "decision_threshold": {

        "selected_threshold": 0.35,

        "default_threshold": 0.50,

        "reason": (
            "Threshold analysis was performed because the default "
            "0.50 threshold was not automatically assumed to be optimal "
            "for the business objective. The selected threshold balanced "
            "the goal of identifying more potential churners with the "
            "trade-off of additional false positives."
        ),

        "lower_threshold_effect": (
            "Lowering the threshold generally increases the number "
            "of customers classified as potential churners, which can "
            "increase recall while reducing precision."
        ),

        "higher_threshold_effect": (
            "Increasing the threshold generally makes the model more "
            "conservative, which may improve precision while reducing recall."
        )
    },


    # ============================================================
    # EVALUATION METRICS
    # ============================================================

    "metrics": {

        "accuracy": (
            "Accuracy measures the overall proportion of correct predictions."
        ),

        "precision": (
            "Precision measures how many customers predicted as churners "
            "were actually churners."
        ),

        "recall": (
            "Recall measures how many actual churners were correctly "
            "identified by the model."
        ),

        "f1_score": (
            "The F1 score combines precision and recall into a single metric."
        ),

        "roc_auc": (
            "ROC-AUC measures the model's ability to distinguish between "
            "churners and non-churners across different thresholds."
        ),

        "confusion_matrix": (
            "A confusion matrix summarizes predictions into true positives, "
            "true negatives, false positives, and false negatives."
        )
    },


    # ============================================================
    # EXPLAINABILITY
    # ============================================================

    "explainability": {

        "method": (
            "Customer-level explanations are generated using the fitted "
            "Logistic Regression coefficients and the customer's transformed "
            "feature values."
        ),

        "positive_contribution": (
            "A positive contribution is associated with increasing the "
            "model's predicted churn risk for that customer."
        ),

        "negative_contribution": (
            "A negative contribution is associated with lowering the "
            "model's predicted churn risk for that customer."
        ),

        "causation_warning": (
            "Feature contributions represent model associations and "
            "do not prove direct causation."
        )
    },


    # ============================================================
    # KEY EXPLORATORY INSIGHTS
    # ============================================================

    "business_insights": {

        "tenure": (
            "Customers with longer tenure generally showed lower churn, "
            "while shorter-tenure customers were more associated with churn."
        ),

        "monthly_charges": (
            "Higher MonthlyCharges were associated with higher churn "
            "patterns in the exploratory analysis."
        ),

        "month_to_month": (
            "Month-to-month contracts showed higher observed churn "
            "than longer-term contracts."
        ),

        "fiber_optic": (
            "Fiber optic customers showed the highest observed churn "
            "rate among the InternetService categories."
        ),

        "fiber_optic_churn_rate": "41.89%",

        "dsl_churn_rate": "18.96%",

        "no_internet_churn_rate": "7.40%",

        "electronic_check": (
            "Electronic check had the highest observed churn rate "
            "among the payment methods analyzed."
        ),

        "electronic_check_churn_rate": "45.29%",

        "paperless_billing": (
            "Customers using PaperlessBilling showed a higher observed "
            "churn rate than customers not using PaperlessBilling."
        ),

        "paperless_billing_yes_rate": "33.57%",

        "paperless_billing_no_rate": "16.33%"
    },


    # ============================================================
    # LIMITATIONS
    # ============================================================

    "limitations": {

        "prediction_not_guarantee": (
            "The model provides probability-based predictions and "
            "cannot guarantee future customer behavior."
        ),

        "association_not_causation": (
            "Patterns and feature contributions represent associations "
            "identified by the model and should not be interpreted as "
            "proof of causation."
        ),

        "human_decision": (
            "The application is a decision-support system. Final business "
            "decisions remain with human decision-makers."
        ),

        "data_dependency": (
            "Model performance depends on the quality, relevance, and "
            "representativeness of the available training data."
        )
    },


    # ============================================================
    # AI SAFETY
    # ============================================================

    "ai_safety": {

        "ml_integrity": (
            "The AI communication layer must not alter verified ML outputs "
            "such as churn probability, threshold, prediction, risk level, "
            "or verified customer-level factors."
        ),

        "no_invention": (
            "The AI should not invent customer information, model results, "
            "or undocumented business facts."
        ),

        "human_oversight": (
            "The final business decision remains with a human decision-maker."
        )
    }
}

def answer_project_question(question, verified_payload=None):

    question_lower = question.lower().strip()

    # ============================================================
    # PROJECT OVERVIEW
    # ============================================================

    if any(word in question_lower for word in [
        "what is this project",
        "project about",
        "project objective",
        "what does this project do"
    ]):

        return (
            f"This project is called "
            f"{PROJECT_KNOWLEDGE['project_overview']['name']}. "
            f"Its objective is to "
            f"{PROJECT_KNOWLEDGE['project_overview']['objective']}"
        )


    # ============================================================
    # BUSINESS PROBLEM
    # ============================================================

    if any(word in question_lower for word in [
        "business problem",
        "business objective",
        "main goal",
        "goal of the project"
    ]):

        return (
            PROJECT_KNOWLEDGE["business_objective"]["primary_goal"]
        )


    # ============================================================
    # WHY LOGISTIC REGRESSION
    # ============================================================

    if any(word in question_lower for word in [
        "why logistic regression",
        "why did you choose logistic",
        "why choose logistic",
        "why was logistic regression selected",
        "why is logistic regression the final model"
    ]):

        return (
            PROJECT_KNOWLEDGE["final_model"]["selection_reason"]
        )


    # ============================================================
    # FINAL MODEL
    # ============================================================

    if any(word in question_lower for word in [
        "final model",
        "which model",
        "what model are you using",
        "what model was selected"
    ]):

        return (
            f"The final model is "
            f"{PROJECT_KNOWLEDGE['final_model']['name']}."
        )


    # ============================================================
    # RECALL
    # ============================================================

    if any(word in question_lower for word in [
        "why recall",
        "why is recall important",
        "why prioritize recall"
    ]):

        return (
            PROJECT_KNOWLEDGE["business_objective"]["reason_for_recall"]
        )


    # ============================================================
    # FALSE NEGATIVES
    # ============================================================

    if any(word in question_lower for word in [
        "false negative",
        "what is fn",
        "what are false negatives"
    ]):

        return (
            PROJECT_KNOWLEDGE["business_objective"]["false_negative"]
        )


    # ============================================================
    # THRESHOLD
    # ============================================================

    if any(word in question_lower for word in [
        "why 0.35",
        "why threshold",
        "why did you choose the threshold",
        "decision threshold"
    ]):

        return (
            PROJECT_KNOWLEDGE["decision_threshold"]["reason"]
        )


    if any(word in question_lower for word in [
        "lower the threshold",
        "what happens if threshold is lower"
    ]):

        return (
            PROJECT_KNOWLEDGE["decision_threshold"]
            ["lower_threshold_effect"]
        )


    if any(word in question_lower for word in [
        "increase the threshold",
        "higher threshold",
        "raise the threshold"
    ]):

        return (
            PROJECT_KNOWLEDGE["decision_threshold"]
            ["higher_threshold_effect"]
        )


    # ============================================================
    # DATASET SIZE
    # ============================================================

    if any(word in question_lower for word in [
        "how many customers",
        "dataset size",
        "how many records"
    ]):

        return (
            f"The dataset contains "
            f"{PROJECT_KNOWLEDGE['dataset']['total_customers']} "
            f"customer records."
        )


    # ============================================================
    # CLASS IMBALANCE
    # ============================================================

    if any(word in question_lower for word in [
        "imbalanced",
        "class imbalance",
        "dataset balanced"
    ]):

        return (
            PROJECT_KNOWLEDGE["dataset"]["class_imbalance"]
        )


    # ============================================================
    # CUSTOMER ID
    # ============================================================

    if any(word in question_lower for word in [
        "customerid",
        "customer id",
        "why remove customer id"
    ]):

        return (
            PROJECT_KNOWLEDGE["dataset"]["customer_id_reason"]
        )


    # ============================================================
    # PREPROCESSING
    # ============================================================

    if any(word in question_lower for word in [
        "preprocessing",
        "how was the data prepared",
        "how did you prepare the data"
    ]):

        return (
            "The project used One-Hot Encoding for categorical "
            "features, StandardScaler for numerical features, and "
            "a Scikit-learn Pipeline to combine preprocessing and modeling."
        )


    # ============================================================
    # ONE HOT ENCODING
    # ============================================================

    if any(word in question_lower for word in [
        "one hot encoding",
        "categorical variables",
        "categorical features"
    ]):

        return (
            "One-Hot Encoding was used to convert categorical values "
            "into numerical indicator variables suitable for machine learning."
        )


    # ============================================================
    # SCALING
    # ============================================================

    if any(word in question_lower for word in [
        "standardscaler",
        "scaling",
        "why scale"
    ]):

        return (
            "StandardScaler was used to standardize numerical features "
            "within the preprocessing pipeline."
        )


    # ============================================================
    # TOTAL CHARGES
    # ============================================================

    if any(word in question_lower for word in [
        "totalcharges",
        "total charges missing",
        "missing values"
    ]):

        return (
            PROJECT_KNOWLEDGE["preprocessing"]
            ["total_charges_handling"]
        )


    # ============================================================
    # TRAIN TEST SPLIT
    # ============================================================

    if any(word in question_lower for word in [
        "train test split",
        "80 20",
        "training data",
        "testing data"
    ]):

        return (
            "The dataset was split into approximately 80% training "
            "data and 20% testing data."
        )


    # ============================================================
    # STRATIFICATION
    # ============================================================

    if any(word in question_lower for word in [
        "stratify",
        "stratification"
    ]):

        return (
            PROJECT_KNOWLEDGE["data_split"]
            ["reason_for_stratification"]
        )


    # ============================================================
    # DATA LEAKAGE
    # ============================================================

    if "data leakage" in question_lower:

        return (
            "Data leakage occurs when information that would not be "
            "available during real-world prediction improperly influences "
            "model training or evaluation."
        )


    # ============================================================
    # METRICS
    # ============================================================

    if "precision" in question_lower:

        return PROJECT_KNOWLEDGE["metrics"]["precision"]

    if "recall" in question_lower:

        return PROJECT_KNOWLEDGE["metrics"]["recall"]

    if "f1" in question_lower:

        return PROJECT_KNOWLEDGE["metrics"]["f1_score"]

    if any(word in question_lower for word in [
        "roc auc",
        "roc-auc",
        "auc"
    ]):

        return PROJECT_KNOWLEDGE["metrics"]["roc_auc"]

    if "accuracy" in question_lower:

        return PROJECT_KNOWLEDGE["metrics"]["accuracy"]

    if "confusion matrix" in question_lower:

        return PROJECT_KNOWLEDGE["metrics"]["confusion_matrix"]


    # ============================================================
    # EXPLAINABILITY
    # ============================================================

    if any(word in question_lower for word in [
        "how do you explain",
        "explainability",
        "feature contribution"
    ]):

        return (
            PROJECT_KNOWLEDGE["explainability"]["method"]
        )


    if any(word in question_lower for word in [
        "positive contribution",
        "positive factor"
    ]):

        return (
            PROJECT_KNOWLEDGE["explainability"]
            ["positive_contribution"]
        )


    if any(word in question_lower for word in [
        "negative contribution",
        "negative factor"
    ]):

        return (
            PROJECT_KNOWLEDGE["explainability"]
            ["negative_contribution"]
        )


    if any(word in question_lower for word in [
        "causation",
        "cause churn",
        "prove cause"
    ]):

        return (
            PROJECT_KNOWLEDGE["explainability"]
            ["causation_warning"]
        )


    # ============================================================
    # BUSINESS INSIGHTS
    # ============================================================

    if "tenure" in question_lower:

        return PROJECT_KNOWLEDGE["business_insights"]["tenure"]

    if any(word in question_lower for word in [
        "month-to-month",
        "month to month",
        "contract"
    ]):

        return PROJECT_KNOWLEDGE["business_insights"]["month_to_month"]

    if any(word in question_lower for word in [
        "fiber optic",
        "internet service"
    ]):

        return (
            PROJECT_KNOWLEDGE["business_insights"]["fiber_optic"]
        )

    if any(word in question_lower for word in [
        "electronic check",
        "payment method"
    ]):

        return (
            PROJECT_KNOWLEDGE["business_insights"]["electronic_check"]
        )

    if any(word in question_lower for word in [
        "monthly charges",
        "monthlycharges"
    ]):

        return (
            PROJECT_KNOWLEDGE["business_insights"]["monthly_charges"]
        )


    # ============================================================
    # CURRENT CUSTOMER QUESTIONS
    # ============================================================

    if verified_payload is not None:

        if any(word in question_lower for word in [
            "why is this customer",
            "why predicted",
            "why churn",
            "why is the customer high risk"
        ]):

            probability = verified_payload["churn_probability"]
            threshold = verified_payload["threshold"]
            prediction = verified_payload["prediction"]
            risk_level = verified_payload["risk_level"]

            factors = [
                item["feature"]
                for item in verified_payload["higher_risk_signals"]
            ]

            return (
                f"The model estimates a churn probability of "
                f"{probability:.2%}, which is compared with the "
                f"{threshold:.2f} decision threshold. The resulting "
                f"prediction is {prediction} with a {risk_level} risk "
                f"level. The strongest higher-risk signals are: "
                f"{', '.join(factors)}. These represent model "
                f"associations rather than proof of causation."
            )


        if any(word in question_lower for word in [
            "risk factors",
            "strongest factors",
            "higher risk factors"
        ]):

            factors = [
                item["feature"]
                for item in verified_payload["higher_risk_signals"]
            ]

            return (
                "The strongest higher-risk factors identified for "
                "this prediction are: " + ", ".join(factors) + "."
            )


        if any(word in question_lower for word in [
            "lower risk factors",
            "lower-risk factors",
            "protective factors"
        ]):

            factors = [
                item["feature"]
                for item in verified_payload["lower_risk_signals"]
            ]

            return (
                "The lower-risk factors identified for this prediction "
                "are: " + ", ".join(factors) + "."
            )


        if any(word in question_lower for word in [
            "what is the probability",
            "churn probability",
            "prediction probability"
        ]):

            return (
                f"The verified churn probability for the current "
                f"customer is "
                f"{verified_payload['churn_probability']:.2%}."
            )


    # ============================================================
    # LIMITATIONS AND SAFETY
    # ============================================================

    if any(word in question_lower for word in [
        "definitely churn",
        "guarantee churn",
        "certain to churn"
    ]):

        return (
            PROJECT_KNOWLEDGE["limitations"]
            ["prediction_not_guarantee"]
        )


    if any(word in question_lower for word in [
        "final decision",
        "can ai decide",
        "can the model decide"
    ]):

        return (
            PROJECT_KNOWLEDGE["limitations"]["human_decision"]
        )


    if any(word in question_lower for word in [
        "can ai change",
        "change prediction",
        "alter prediction"
    ]):

        return (
            PROJECT_KNOWLEDGE["ai_safety"]["ml_integrity"]
        )


    # ============================================================
    # FALLBACK
    # ============================================================

    return (
        "I can answer questions about this Telco Customer Churn project, "
        "including the dataset, preprocessing, exploratory analysis, "
        "models tested, Logistic Regression selection, recall, threshold "
        "selection, evaluation metrics, explainability, customer predictions, "
        "business insights, limitations, and AI safety. "
        "Please ask your question in relation to one of these areas."
    )
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

st.markdown("---")
st.subheader("💬 Ask About This Project")

st.write(
    "Ask questions about the Telco Customer Churn project, "
    "including the dataset, preprocessing, models, evaluation, "
    "threshold selection, explainability, AI safety, and business reasoning."
)

user_question = st.text_input(
    "Ask a question about the project",
    placeholder="Example: Why was Logistic Regression selected?"
)

if user_question:

    answer = answer_project_question(
        user_question,
        verified_payload=verified_payload
    )

    st.write(answer)

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
