# ============================================================
# feature_engineering.py
# Purpose: Take raw financial inputs (income, expenses, EMI,
#          savings, debt, etc.) and derive the ratios used across
#          the whole application - Financial Health Score, Risk
#          Analysis, Anomaly Detection, and Clustering all reuse
#          these same functions instead of recalculating them.
#
# This module has NO console input() calls and NO file I/O -
# it's a pure calculation layer, so it can be safely imported by
# predict.py, app.py, anomaly_detection.py, clustering.py, etc.
# ============================================================

def compute_financial_features(
    monthly_income: float,
    monthly_expenses: float,
    monthly_savings: float,
    existing_emi: float,
    other_debt: float,
    loan_amount: float,          # in thousands, matches original dataset convention
    loan_tenure_months: int,
    credit_history: float,       # 1.0 = good, 0.0 = bad
    dependents: int,
    interest_rate_annual: float = 10.0   # assumed rate if user doesn't provide one
) -> dict:
    """
    Computes all derived financial ratios from raw inputs.
    Returns a dictionary of every derived feature, used
    downstream by the health score, risk analysis, and warnings.
    """

    # --------------------------------------------------------
    # Guard against divide-by-zero: if income is 0, ratios that
    # divide by income would crash. We treat 0 income as a
    # worst-case scenario instead of crashing.
    # --------------------------------------------------------
    safe_income = monthly_income if monthly_income > 0 else 1

    # --------------------------------------------------------
    # Estimated Monthly EMI for the NEW loan being requested
    # Uses the standard EMI formula:
    #   EMI = P * r * (1+r)^n / ((1+r)^n - 1)
    # where P = principal, r = monthly interest rate, n = months
    # This estimates what the new loan itself would cost per
    # month, separate from any EMI the person is already paying.
    # --------------------------------------------------------
    principal = loan_amount * 1000  # convert from thousands to actual currency
    monthly_rate = (interest_rate_annual / 100) / 12
    n = max(loan_tenure_months, 1)

    if monthly_rate > 0:
        new_loan_emi = principal * monthly_rate * (1 + monthly_rate) ** n / \
                       ((1 + monthly_rate) ** n - 1)
    else:
        new_loan_emi = principal / n  # fallback for 0% interest edge case

    # --------------------------------------------------------
    # Total EMI burden = existing EMI (already being paid) +
    # the new loan's estimated EMI (if this loan gets approved)
    # --------------------------------------------------------
    total_emi = existing_emi + new_loan_emi

    # --------------------------------------------------------
    # Core ratios
    # --------------------------------------------------------
    emi_to_income_ratio = total_emi / safe_income
    loan_to_income_ratio = principal / (safe_income * 12)   # loan vs ANNUAL income
    savings_rate = monthly_savings / safe_income
    expense_to_income_ratio = monthly_expenses / safe_income
    income_per_dependent = monthly_income / (dependents + 1)  # +1 accounts for the applicant themself
    debt_burden_ratio = (total_emi + other_debt) / safe_income

    # --------------------------------------------------------
    # Disposable Income: what's actually left over each month
    # after all fixed obligations are paid
    # --------------------------------------------------------
    disposable_income = monthly_income - monthly_expenses - total_emi - other_debt

    return {
        "monthly_income": monthly_income,
        "monthly_expenses": monthly_expenses,
        "monthly_savings": monthly_savings,
        "existing_emi": existing_emi,
        "other_debt": other_debt,
        "new_loan_emi": round(new_loan_emi, 2),
        "total_emi": round(total_emi, 2),
        "emi_to_income_ratio": round(emi_to_income_ratio, 4),
        "loan_to_income_ratio": round(loan_to_income_ratio, 4),
        "savings_rate": round(savings_rate, 4),
        "expense_to_income_ratio": round(expense_to_income_ratio, 4),
        "income_per_dependent": round(income_per_dependent, 2),
        "debt_burden_ratio": round(debt_burden_ratio, 4),
        "disposable_income": round(disposable_income, 2),
        "credit_history": credit_history,
        "dependents": dependents,
    }


def _score_from_ratio(ratio: float, good_at: float, bad_at: float) -> float:
    """
    Helper: converts a ratio into a 0-100 sub-score.
    'good_at' = ratio value that earns a full 100 score
    'bad_at'  = ratio value that earns a 0 score
    Works for both "lower is better" (bad_at > good_at) and
    "higher is better" (good_at > bad_at) ratios - handles both
    directions with the same formula.
    """
    if good_at == bad_at:
        return 100.0

    # Linear interpolation between bad_at (0) and good_at (100)
    score = 100 * (ratio - bad_at) / (good_at - bad_at)
    return max(0.0, min(100.0, score))   # clip to the 0-100 range


def compute_financial_health_score(features: dict) -> dict:
    """
    Computes the 0-100 Financial Health Score from derived features.
    This is a TRANSPARENT, RULE-BASED formula (not a trained ML
    model) because the dataset has no historical health-score
    labels to train a model on. Every weight and threshold below
    is explicit and explainable - useful for your project review.
    """

    # Each ratio -> a 0-100 sub-score, using reasonable real-world
    # thresholds (these mirror common personal-finance guidelines,
    # e.g. the "50/30/20 rule" and standard EMI safety limits)
    savings_score = _score_from_ratio(features["savings_rate"], good_at=0.30, bad_at=0.0)
    emi_score = _score_from_ratio(features["emi_to_income_ratio"], good_at=0.0, bad_at=0.60)
    expense_score = _score_from_ratio(features["expense_to_income_ratio"], good_at=0.30, bad_at=0.90)
    credit_score = 100.0 if features["credit_history"] == 1.0 else 30.0
    debt_score = _score_from_ratio(features["debt_burden_ratio"], good_at=0.0, bad_at=0.70)
    loan_score = _score_from_ratio(features["loan_to_income_ratio"], good_at=0.0, bad_at=5.0)

    # --------------------------------------------------------
    # Weighted combination - weights sum to 1.0 (100%)
    # Savings, EMI burden, Expense ratio, and Credit History are
    # weighted highest (20% each) since they most directly reflect
    # day-to-day financial resilience. Debt burden and loan size
    # are weighted lower (10% each) since they're partially
    # captured already within EMI/expense ratios.
    # --------------------------------------------------------
    weights = {
        "savings": 0.20,
        "emi": 0.20,
        "expense": 0.20,
        "credit": 0.20,
        "debt": 0.10,
        "loan": 0.10,
    }

    final_score = (
        savings_score * weights["savings"] +
        emi_score * weights["emi"] +
        expense_score * weights["expense"] +
        credit_score * weights["credit"] +
        debt_score * weights["debt"] +
        loan_score * weights["loan"]
    )
    final_score = round(final_score, 1)

    # --------------------------------------------------------
    # Classify into the 4 bands you specified
    # --------------------------------------------------------
    if final_score >= 90:
        category = "Excellent"
    elif final_score >= 70:
        category = "Good"
    elif final_score >= 50:
        category = "Moderate"
    else:
        category = "High Risk"

    return {
        "financial_health_score": final_score,
        "category": category,
        "sub_scores": {
            "savings_score": round(savings_score, 1),
            "emi_score": round(emi_score, 1),
            "expense_score": round(expense_score, 1),
            "credit_score": round(credit_score, 1),
            "debt_score": round(debt_score, 1),
            "loan_score": round(loan_score, 1),
        },
        "weights_used": weights,
    }


# ------------------------------------------------------------
# Quick manual test - only runs if you execute this file directly
# (python src/feature_engineering.py), NOT when imported elsewhere.
# Useful to sanity-check the formula without needing the full app.
# ------------------------------------------------------------
if __name__ == "__main__":
    sample_features = compute_financial_features(
        monthly_income=40000,
        monthly_expenses=20000,
        monthly_savings=5000,
        existing_emi=3000,
        other_debt=0,
        loan_amount=200,       # 200 thousand
        loan_tenure_months=60,
        credit_history=1.0,
        dependents=2
    )
    print("Derived Financial Features:")
    for k, v in sample_features.items():
        print(f"  {k}: {v}")

    health = compute_financial_health_score(sample_features)
    print("\nFinancial Health Score Result:")
    print(f"  Score: {health['financial_health_score']} ({health['category']})")
    print(f"  Sub-scores: {health['sub_scores']}")