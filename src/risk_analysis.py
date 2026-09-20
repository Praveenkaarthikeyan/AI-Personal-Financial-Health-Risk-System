# ============================================================
# risk_analysis.py
# Purpose: Derive a Low/Medium/High Risk classification and
#          generate specific early-warning messages, using the
#          ratios already computed by feature_engineering.py.
#
# Like the Financial Health Score, this is a TRANSPARENT
# RULE-BASED system, not a trained ML model - the dataset has no
# historical "risk level" labels to learn from. Every threshold
# below is explicit and explainable for your project review.
# ============================================================


def assess_risk_level(features: dict) -> dict:
    """
    Classifies overall financial risk as Low / Medium / High,
    based on the most risk-relevant ratios: EMI burden, savings
    rate, expense ratio, debt burden, and credit history.

    Returns risk level, a 0-100 "risk probability" (higher =
    riskier), and which factors contributed most.
    """

    # Each ratio contributes "risk points" if it crosses a
    # concerning threshold. More points = higher risk.
    risk_points = 0
    max_points = 0
    contributing_factors = []

    # --- EMI-to-Income: the single biggest risk indicator ---
    max_points += 30
    if features["emi_to_income_ratio"] > 0.50:
        risk_points += 30
        contributing_factors.append("Very high EMI burden (over 50% of income)")
    elif features["emi_to_income_ratio"] > 0.35:
        risk_points += 18
        contributing_factors.append("High EMI burden (over 35% of income)")
    elif features["emi_to_income_ratio"] > 0.20:
        risk_points += 8

    # --- Savings Rate: low/negative savings is a red flag ---
    max_points += 20
    if features["savings_rate"] <= 0:
        risk_points += 20
        contributing_factors.append("No savings, or spending more than earned")
    elif features["savings_rate"] < 0.10:
        risk_points += 12
        contributing_factors.append("Low savings rate (under 10% of income)")

    # --- Expense-to-Income: high expenses leave little room ---
    max_points += 20
    if features["expense_to_income_ratio"] > 0.80:
        risk_points += 20
        contributing_factors.append("Expenses consume over 80% of income")
    elif features["expense_to_income_ratio"] > 0.60:
        risk_points += 10

    # --- Credit History: strong signal on its own ---
    max_points += 20
    if features["credit_history"] == 0.0:
        risk_points += 20
        contributing_factors.append("Poor or no credit history")

    # --- Debt Burden: overall debt load including other debts ---
    max_points += 10
    if features["debt_burden_ratio"] > 0.60:
        risk_points += 10
        contributing_factors.append("High overall debt burden")

    # Convert risk points into a 0-100 "risk probability"
    risk_probability = round((risk_points / max_points) * 100, 1)

    # Classify into 3 bands
    if risk_probability >= 60:
        risk_level = "High Risk"
    elif risk_probability >= 30:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    if not contributing_factors:
        contributing_factors.append("No major risk factors detected")

    return {
        "risk_level": risk_level,
        "risk_probability": risk_probability,
        "contributing_factors": contributing_factors,
    }


def generate_early_warnings(features: dict) -> list:
    """
    Checks the financial features against specific danger
    thresholds and returns a list of clear warning messages.
    Each warning is specific and actionable, not generic -
    directly reflects the actual numbers involved.
    """
    warnings = []

    if features["emi_to_income_ratio"] > 0.40:
        pct = features["emi_to_income_ratio"] * 100
        warnings.append(
            f"⚠ High EMI burden detected: your EMIs consume {pct:.0f}% of your "
            f"monthly income (recommended limit is under 40%)."
        )

    if features["disposable_income"] < 0:
        warnings.append(
            f"⚠ Your expenses and EMIs exceed your income by "
            f"₹{abs(features['disposable_income']):.0f}/month. This is unsustainable."
        )
    elif features["expense_to_income_ratio"] > 0.70:
        pct = features["expense_to_income_ratio"] * 100
        warnings.append(
            f"⚠ Your current expenses leave very little disposable income "
            f"(expenses are {pct:.0f}% of your income)."
        )

    if features["savings_rate"] < 0.10:
        pct = features["savings_rate"] * 100
        warnings.append(
            f"⚠ Low savings rate detected: you're saving only {pct:.0f}% of your "
            f"income. Financial experts recommend at least 20%."
        )

    if features["loan_to_income_ratio"] > 3.0:
        warnings.append(
            f"⚠ The requested loan amount is high relative to your annual income "
            f"(loan-to-income ratio: {features['loan_to_income_ratio']:.1f}x)."
        )

    if features["debt_burden_ratio"] > 0.50:
        pct = features["debt_burden_ratio"] * 100
        warnings.append(
            f"⚠ Increasing debt may reduce your financial stability: total debt "
            f"obligations are {pct:.0f}% of your income."
        )

    if features["credit_history"] == 0.0:
        warnings.append(
            "⚠ Poor or missing credit history significantly increases risk and "
            "may affect loan approval regardless of income."
        )

    return warnings


# ------------------------------------------------------------
# Quick manual test - only runs when this file is executed
# directly (python src/risk_analysis.py), not when imported.
# ------------------------------------------------------------
if __name__ == "__main__":
    # Import here (not at top) so this test block works standalone
    # without requiring feature_engineering.py to be run first
    from feature_engineering import compute_financial_features

    # A deliberately risky profile to demonstrate the warnings firing
    risky_features = compute_financial_features(
        monthly_income=25000,
        monthly_expenses=20000,
        monthly_savings=500,
        existing_emi=5000,
        other_debt=2000,
        loan_amount=300,
        loan_tenure_months=36,
        credit_history=0.0,
        dependents=3
    )

    print("Derived features (risky profile):")
    for k, v in risky_features.items():
        print(f"  {k}: {v}")

    risk_result = assess_risk_level(risky_features)
    print("\nRisk Assessment:")
    print(f"  Risk Level: {risk_result['risk_level']}")
    print(f"  Risk Probability: {risk_result['risk_probability']}%")
    print(f"  Contributing Factors: {risk_result['contributing_factors']}")

    print("\nEarly Warnings:")
    for w in generate_early_warnings(risky_features):
        print(f"  {w}")