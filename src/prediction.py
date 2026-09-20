# ============================================================
# prediction.py
# Purpose: The single entry point that ties everything together -
#          the original Unit 1-3 loan approval model PLUS the new
#          Financial Health Score, Risk Analysis, Recommendations,
#          Anomaly Detection, and Clustering modules.
#
# app.py (the Streamlit dashboard) calls run_full_prediction()
# once and gets back one complete result dictionary - it doesn't
# need to know about scalers, encoders, or individual model files.
# This keeps the complex wiring in one well-tested place.
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os

from feature_engineering import compute_financial_features, compute_financial_health_score
from risk_analysis import assess_risk_level, generate_early_warnings
from recommendation_engine import generate_recommendations
from anomaly_detection import check_anomaly
from clustering import assign_cluster

# ------------------------------------------------------------
# Load the ORIGINAL trained loan model artifacts once, at import
# time, so they aren't reloaded from disk on every prediction.
# ------------------------------------------------------------
_scaler = joblib.load(os.path.join("models", "scaler.pkl"))
_label_encoders = joblib.load(os.path.join("models", "label_encoders.pkl"))
_loan_model = joblib.load(os.path.join("models", "random_forest.pkl"))


def _predict_loan_approval(applicant: dict) -> dict:
    """
    Runs the ORIGINAL Unit 1-3 pipeline: builds the model's input
    row exactly as predict.py did, applies the same encoders and
    scaler used during training, and returns the approval
    prediction, probability, and top contributing factors
    (explainability via Random Forest feature importance).
    """
    input_dict = {
        "Gender": applicant["gender"],
        "Married": applicant["married"],
        "Dependents": str(applicant["dependents"]).replace("3+", "3"),
        "Education": applicant["education"],
        "Self_Employed": applicant["self_employed"],
        "Loan_Amount_Term": applicant["loan_amount_term"],
        "Credit_History": applicant["credit_history"],
    }
    df_input = pd.DataFrame([input_dict])

    for col in ["Gender", "Married", "Education", "Self_Employed", "Dependents"]:
        le = _label_encoders[col]
        df_input[col] = le.transform(df_input[col])

    df_input["Property_Area_Semiurban"] = 1 if applicant["property_area"] == "Semiurban" else 0
    df_input["Property_Area_Urban"] = 1 if applicant["property_area"] == "Urban" else 0

    total_income = applicant["applicant_income"] + applicant["coapplicant_income"]
    df_input["TotalIncome_log"] = np.log1p(total_income)
    df_input["LoanAmount_log"] = np.log1p(applicant["loan_amount"])

    feature_order = _scaler.feature_names_in_
    df_input = df_input[feature_order]

    X_scaled = _scaler.transform(df_input)

    prediction = _loan_model.predict(X_scaled)[0]
    probabilities = _loan_model.predict_proba(X_scaled)[0]

    # Explainability: top 5 features by importance, for this applicant
    importances = _loan_model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_order, "Importance": importances
    }).sort_values("Importance", ascending=False).head(5)

    return {
        "loan_prediction": "Approved" if prediction == 1 else "Rejected",
        "loan_approval_probability": round(float(probabilities[1]), 4),
        "loan_rejection_probability": round(float(probabilities[0]), 4),
        "top_factors": importance_df.to_dict(orient="records"),
    }


def run_full_prediction(applicant: dict) -> dict:
    """
    The main entry point. 'applicant' is a dict expected to contain:
        gender, married, dependents, education, self_employed,
        employment_status, applicant_income, coapplicant_income,
        loan_amount, loan_amount_term, credit_history, property_area,
        monthly_expenses, monthly_savings, existing_emi, other_debt

    Returns a single dict with every result needed by the dashboard:
    loan prediction, health score, risk level, warnings,
    recommendations, cluster, and anomaly check.
    """

    # --------------------------------------------------------
    # 1. Original Unit 1-3 loan approval prediction
    # --------------------------------------------------------
    loan_result = _predict_loan_approval(applicant)

    # --------------------------------------------------------
    # 2. Derived financial features (shared by everything below)
    # --------------------------------------------------------
    features = compute_financial_features(
        monthly_income=applicant["applicant_income"] + applicant["coapplicant_income"],
        monthly_expenses=applicant["monthly_expenses"],
        monthly_savings=applicant["monthly_savings"],
        existing_emi=applicant["existing_emi"],
        other_debt=applicant["other_debt"],
        loan_amount=applicant["loan_amount"],
        loan_tenure_months=applicant["loan_amount_term"],
        credit_history=applicant["credit_history"],
        dependents=int(str(applicant["dependents"]).replace("3+", "3")),
    )

    # --------------------------------------------------------
    # 3. Financial Health Score
    # --------------------------------------------------------
    health_result = compute_financial_health_score(features)

    # --------------------------------------------------------
    # 4. Risk Level + Early Warnings
    # --------------------------------------------------------
    risk_result = assess_risk_level(features)
    warnings = generate_early_warnings(features)

    # --------------------------------------------------------
    # 5. Personalized Recommendations
    # --------------------------------------------------------
    recommendations = generate_recommendations(features, risk_result)

    # --------------------------------------------------------
    # 6. Anomaly Detection
    # --------------------------------------------------------
    anomaly_result = check_anomaly(features)

    # --------------------------------------------------------
    # 7. Financial Behavior Clustering
    # --------------------------------------------------------
    cluster_result = assign_cluster(features)

    # --------------------------------------------------------
    # 8. Employment advisory (Student/Unemployed rule-based note,
    #    same reasoning as the original predict.py)
    # --------------------------------------------------------
    employment_advisory = None
    if applicant.get("employment_status") in ["Student", "Unemployed"]:
        notes = []
        if applicant["coapplicant_income"] == 0:
            notes.append("No co-applicant/guarantor income provided.")
        if applicant["credit_history"] == 0:
            notes.append("Poor or no credit history.")
        employment_advisory = {
            "message": "Training data has no dedicated Student/Unemployed category; "
                       "treat the loan prediction with caution for this applicant type.",
            "notes": notes,
        }

    # --------------------------------------------------------
    # Combine everything into one result dictionary
    # --------------------------------------------------------
    return {
        "features": features,
        "loan": loan_result,
        "health": health_result,
        "risk": risk_result,
        "warnings": warnings,
        "recommendations": recommendations,
        "anomaly": anomaly_result,
        "cluster": cluster_result,
        "employment_advisory": employment_advisory,
    }


# ------------------------------------------------------------
# Quick manual test - only runs when this file is executed
# directly (python src/prediction.py)
# ------------------------------------------------------------
if __name__ == "__main__":
    sample_applicant = {
        "gender": "Male", "married": "Yes", "dependents": "0",
        "education": "Graduate", "self_employed": "No",
        "employment_status": "Salaried",
        "applicant_income": 6000, "coapplicant_income": 2000,
        "loan_amount": 150, "loan_amount_term": 360,
        "credit_history": 1.0, "property_area": "Semiurban",
        "monthly_expenses": 4000, "monthly_savings": 1500,
        "existing_emi": 1000, "other_debt": 0,
    }

    result = run_full_prediction(sample_applicant)

    print("===== Full Prediction Result =====")
    print(f"Loan: {result['loan']['loan_prediction']} "
          f"({result['loan']['loan_approval_probability']*100:.1f}% approval probability)")
    print(f"Health Score: {result['health']['financial_health_score']} "
          f"({result['health']['category']})")
    print(f"Risk Level: {result['risk']['risk_level']} "
          f"({result['risk']['risk_probability']}%)")
    print(f"Cluster: {result['cluster']['cluster_label']}")
    print(f"Anomaly: {result['anomaly']['is_anomaly']}")
    print(f"Warnings: {len(result['warnings'])} triggered")
    for w in result["warnings"]:
        print(f"  {w}")
    print(f"Recommendations: {len(result['recommendations'])} generated")
    for r in result["recommendations"]:
        print(f"  - {r}")
