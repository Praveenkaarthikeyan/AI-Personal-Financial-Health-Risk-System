# ============================================================
# recommendation_engine.py
# Purpose: Generate specific, personalized recommendations based
#          on the user's actual financial ratios - not generic
#          text. Reuses the same features dict produced by
#          feature_engineering.py.
#
# Like risk_analysis.py, this is rule-based logic, not ML -
# recommendations need to be explainable and directly traceable
# to the numbers involved, which a black-box model wouldn't give us.
# ============================================================


def generate_recommendations(features: dict, risk_result: dict) -> list:
    """
    Returns a list of personalized recommendation strings, each
    referencing the applicant's own numbers wherever possible.
    Recommendations are ordered roughly by priority/impact.
    """
    recommendations = []

    # --------------------------------------------------------
    # 1. EMI burden too high -> suggest reducing loan amount or
    #    extending tenure to lower the monthly EMI
    # --------------------------------------------------------
    if features["emi_to_income_ratio"] > 0.40:
        # Estimate what loan amount WOULD bring EMI ratio down to a safe 35%
        safe_emi_target = 0.35 * features["monthly_income"]
        # Rough scaling since EMI is roughly proportional to principal
        if features["total_emi"] > 0:
            scale_factor = safe_emi_target / features["total_emi"]
        else:
            scale_factor = 1.0
        recommendations.append(
            f"Consider reducing your requested loan amount, or increasing the loan "
            f"tenure, to bring your EMI-to-income ratio closer to a safer 35% "
            f"(it's currently {features['emi_to_income_ratio']*100:.0f}%)."
        )

    # --------------------------------------------------------
    # 2. Low savings rate -> suggest a target savings amount
    # --------------------------------------------------------
    if features["savings_rate"] < 0.20:
        target_savings = round(0.20 * features["monthly_income"])
        gap = round(target_savings - features["monthly_savings"])
        if gap > 0:
            recommendations.append(
                f"Try to increase monthly savings by about ₹{gap} to reach the "
                f"recommended 20% savings rate (you're currently saving "
                f"{features['savings_rate']*100:.0f}%)."
            )

    # --------------------------------------------------------
    # 3. High expense ratio -> suggest a specific reduction target
    # --------------------------------------------------------
    if features["expense_to_income_ratio"] > 0.60:
        target_expense = round(0.50 * features["monthly_income"])
        reduction = round(features["monthly_expenses"] - target_expense)
        if reduction > 0:
            recommendations.append(
                f"Reduce discretionary/monthly expenses by roughly ₹{reduction} "
                f"to bring your expense ratio down to a healthier 50% of income."
            )

    # --------------------------------------------------------
    # 4. Negative or very low disposable income -> urgent action
    # --------------------------------------------------------
    if features["disposable_income"] < 0:
        recommendations.append(
            "Your current obligations exceed your income - avoid taking on any "
            "additional debt until expenses and EMIs are brought under control."
        )
    elif features["disposable_income"] < 0.10 * features["monthly_income"]:
        recommendations.append(
            "Your disposable income is very thin. Building a small emergency "
            "fund (even 1-2 months of expenses) should be a priority before "
            "taking on new debt."
        )

    # --------------------------------------------------------
    # 5. High loan-to-income ratio -> suggest a smaller loan
    # --------------------------------------------------------
    if features["loan_to_income_ratio"] > 3.0:
        recommendations.append(
            "The requested loan amount is large relative to your annual income. "
            "Consider requesting a smaller amount, or adding a co-applicant to "
            "strengthen the application."
        )

    # --------------------------------------------------------
    # 6. Poor credit history -> suggest rebuilding it
    # --------------------------------------------------------
    if features["credit_history"] == 0.0:
        recommendations.append(
            "Focus on rebuilding your credit history with small, on-time "
            "repayments (e.g. a credit card or small loan) before applying "
            "for a larger loan - this is one of the strongest factors lenders consider."
        )

    # --------------------------------------------------------
    # 7. High debt burden -> avoid new debt, consider consolidation
    # --------------------------------------------------------
    if features["debt_burden_ratio"] > 0.50 and features["other_debt"] > 0:
        recommendations.append(
            "Consider consolidating existing debts or paying down high-interest "
            "debt first, since your total debt burden is already significant."
        )

    # --------------------------------------------------------
    # 8. If everything looks healthy, say so explicitly
    # --------------------------------------------------------
    if not recommendations:
        recommendations.append(
            "Your financial profile looks healthy across all key indicators. "
            "Maintaining your current savings rate and avoiding unnecessary new "
            "debt will help sustain this."
        )

    return recommendations


# ------------------------------------------------------------
# Quick manual test - only runs when this file is executed
# directly (python src/recommendation_engine.py)
# ------------------------------------------------------------
if __name__ == "__main__":
    from feature_engineering import compute_financial_features
    from risk_analysis import assess_risk_level

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

    risk_result = assess_risk_level(risky_features)
    recommendations = generate_recommendations(risky_features, risk_result)

    print("Personalized Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")