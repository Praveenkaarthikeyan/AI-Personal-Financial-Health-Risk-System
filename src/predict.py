import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

os.makedirs("outputs", exist_ok=True)

# Load the saved scaler, label encoders, and trained model
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
model = joblib.load("models/random_forest.pkl")  # best performing model so far

print("===== Loan Eligibility Predictor =====")
print("Enter applicant details below:\n")

# ------------------------------------------------------------
# STEP 1: Collect input from the user
# .strip().capitalize() auto-fixes casing so "no", "NO", "No" all work
# ------------------------------------------------------------
gender = input("Gender (Male/Female): ").strip().capitalize()
married = input("Married (Yes/No): ").strip().capitalize()
dependents = input("Dependents (0/1/2/3+): ").strip()
education_raw = input("Education (Graduate/Not Graduate): ").strip().lower()
education = "Graduate" if education_raw == "graduate" else "Not Graduate"


print("\nEmployment Status options: Salaried / Self-Employed / Business / Student / Unemployed")
employment_status = input("Employment Status: ").strip().capitalize()

if employment_status in ["Self-Employed", "Business"]:
    self_employed = "Yes"
else:
    self_employed = "No"   # Salaried, Student, Unemployed all map to "No" for the model

applicant_income = float(input("Applicant Income: "))
coapplicant_income = float(input("Coapplicant Income: "))
loan_amount = float(input("Loan Amount (in thousands): "))
loan_amount_term = float(input("Loan Amount Term (in days, e.g. 360): "))
credit_history = float(input("Credit History (1.0 = good, 0.0 = bad): "))
property_area = input("Property Area (Urban/Semiurban/Rural): ").strip().capitalize()

# ------------------------------------------------------------
# STEP 2: Build a single-row dataframe matching the training data format
# ------------------------------------------------------------
input_dict = {
    "Gender": gender,
    "Married": married,
    "Dependents": dependents.replace("3+", "3"),
    "Education": education,
    "Self_Employed": self_employed,
    "Loan_Amount_Term": loan_amount_term,
    "Credit_History": credit_history,
}
df_input = pd.DataFrame([input_dict])

for col in ["Gender", "Married", "Education", "Self_Employed", "Dependents"]:
    le = label_encoders[col]
    df_input[col] = le.transform(df_input[col])

df_input["Property_Area_Semiurban"] = 1 if property_area == "Semiurban" else 0
df_input["Property_Area_Urban"] = 1 if property_area == "Urban" else 0

total_income = applicant_income + coapplicant_income
df_input["TotalIncome_log"] = np.log1p(total_income)
df_input["LoanAmount_log"] = np.log1p(loan_amount)

feature_order = scaler.feature_names_in_
df_input = df_input[feature_order]

X_scaled = scaler.transform(df_input)

# ------------------------------------------------------------
# STEP 3: Predict
# ------------------------------------------------------------
prediction = model.predict(X_scaled)[0]
probabilities = model.predict_proba(X_scaled)[0]
prob_rejected = probabilities[0] * 100
prob_approved = probabilities[1] * 100

# ------------------------------------------------------------
# STEP 4: Print applicant summary
# ------------------------------------------------------------
print("\n===== Applicant Summary =====")
print(f"Gender                : {gender}")
print(f"Married                : {married}")
print(f"Dependents             : {dependents}")
print(f"Education              : {education}")
print(f"Employment Status      : {employment_status}")
print(f"Applicant Income       : {applicant_income}")
print(f"Coapplicant Income     : {coapplicant_income}")
print(f"Total Income           : {total_income}")
print(f"Loan Amount (000s)     : {loan_amount}")
print(f"Loan Amount Term       : {loan_amount_term} days")
print(f"Credit History         : {credit_history}")
print(f"Property Area          : {property_area}")

print("\n===== Prediction Result =====")
if prediction == 1:
    print("✅ Loan Status: APPROVED")
else:
    print("❌ Loan Status: REJECTED")

print(f"\nProbability of Approval : {prob_approved:.1f}%")
print(f"Probability of Rejection: {prob_rejected:.1f}%")

# ------------------------------------------------------------
# STEP 5: Employment-based advisory (Student / Unemployed)
# Rule-based layer, separate from the ML model, because the training
# data has no real examples of students or unemployed applicants.
# ------------------------------------------------------------
if employment_status in ["Student", "Unemployed"]:
    print("\n===== Employment Advisory =====")
    print("Note: the training data has no dedicated 'Student' or 'Unemployed'")
    print("category, so the model treats this applicant like a salaried person")
    print("with the income you entered. In practice, banks usually require:")
    if coapplicant_income == 0:
        print(" - A co-applicant / guarantor with steady income (currently: none provided)")
    if credit_history == 0:
        print(" - A good credit history, or an alternative guarantee (currently: poor/no credit history)")
    print(" - Treat the model's prediction above with caution for this applicant type.")

# ------------------------------------------------------------
# STEP 6: Explainability - WHY the model decided this way
# We use the Random Forest's built-in feature_importances_, which
# ranks how much each feature influenced predictions overall. We
# then show the applicant's own value for the top features, and
# whether it worked in their favor or against them, using simple,
# well-known directional rules for this dataset domain.
# ------------------------------------------------------------
importances = model.feature_importances_
feature_names = feature_order

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

top_features = importance_df.head(5)

print("\n===== Why this decision? (Top factors considered) =====")

# Simple directional explanations for the most common important features.
# These reflect well-known patterns in this dataset, used to make the
# explanation readable for a non-technical evaluator.
explanations = {
    "Credit_History": lambda: (
        "good credit history - a strong positive factor" if credit_history == 1.0
        else "poor/no credit history - a strong negative factor"
    ),
    "TotalIncome_log": lambda: (
        f"total income of {total_income:.0f} - "
        + ("relatively strong" if total_income >= 5000 else "relatively low")
    ),
    "LoanAmount_log": lambda: (
        f"requested loan amount of {loan_amount:.0f} (in thousands) - "
        + ("high relative to typical applicants" if loan_amount >= 150 else "reasonable relative to typical applicants")
    ),
    "Loan_Amount_Term": lambda: f"loan term of {loan_amount_term:.0f} days",
    "Married": lambda: f"marital status: {married}",
    "Education": lambda: f"education level: {education}",
    "Dependents": lambda: f"number of dependents: {dependents}",
    "Self_Employed": lambda: f"employment type: {employment_status}",
    "Property_Area_Urban": lambda: f"property located in: {property_area}",
    "Property_Area_Semiurban": lambda: f"property located in: {property_area}",
    "Gender": lambda: f"gender: {gender}",
}

for _, row in top_features.iterrows():
    feat = row["Feature"]
    weight = row["Importance"] * 100
    detail = explanations.get(feat, lambda: "factor considered by the model")()
    print(f" - {feat} (influence: {weight:.1f}%): {detail}")

# ------------------------------------------------------------
# STEP 7: Graphs - probability bar chart + feature importance chart
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Probability chart
bars = axes[0].bar(
    ["Rejected", "Approved"],
    [prob_rejected, prob_approved],
    color=["#C44E52", "#55A868"]
)
axes[0].set_ylim(0, 100)
axes[0].set_ylabel("Probability (%)")
axes[0].set_title("Loan Eligibility Prediction")
for bar, value in zip(bars, [prob_rejected, prob_approved]):
    axes[0].text(bar.get_x() + bar.get_width() / 2, value + 2,
                  f"{value:.1f}%", ha="center", fontweight="bold")

# Feature importance chart (top 5, explainability)
axes[1].barh(top_features["Feature"][::-1], top_features["Importance"][::-1], color="#4C72B0")
axes[1].set_xlabel("Importance")
axes[1].set_title("Top Factors Influencing Decision")

plt.tight_layout()
plt.savefig("outputs/prediction_result_and_explanation.png")
plt.close()

print("\nSaved: outputs/prediction_result_and_explanation.png")