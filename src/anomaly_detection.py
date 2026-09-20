# ============================================================
# anomaly_detection.py
# Purpose: Detect financial profiles that look unusual compared
#          to the overall population, using Isolation Forest.
#
# This is a NEW ML concept beyond Units 1-3 (which were all
# supervised learning). Isolation Forest is UNSUPERVISED - it
# has no "correct answer" to learn from; it learns purely from
# the shape/spread of the data itself.
#
# HOW IT WORKS (high level):
# It builds many random trees that repeatedly split data on
# random features at random thresholds. Typical points sit in
# dense clusters and take many splits to isolate. Anomalies sit
# far from everything else and get isolated in very few splits.
# Each point gets an "anomaly score" based on how quickly it was
# isolated - the faster, the more anomalous.
#
# LIMITATION (be upfront about this in your review): the Kaggle
# loan dataset has no expense/savings/EMI history to train on, so
# this model trains on synthetic baseline profiles representing
# "typical" financial behavior, combined with any real user
# profiles saved in the database over time. As more real users
# are added, the model becomes more representative of your
# actual user base.
# ============================================================

import numpy as np
import pandas as pd
import joblib
import os

from sklearn.ensemble import IsolationForest

MODEL_PATH = os.path.join("models", "isolation_forest.pkl")

# Features used to judge "typical" vs "unusual" financial behavior
FEATURE_COLUMNS = [
    "monthly_income", "monthly_expenses", "monthly_savings",
    "emi_to_income_ratio", "expense_to_income_ratio",
    "loan_to_income_ratio", "debt_burden_ratio"
]


def _generate_synthetic_baseline(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic 'typical' financial profiles to give the
    Isolation Forest a reasonable baseline of normal behavior,
    since we don't have real historical financial-behavior data.
    Values are sampled from realistic, plausible ranges.
    """
    rng = np.random.default_rng(seed)

    income = rng.normal(35000, 15000, n_samples).clip(5000, 150000)
    expense_ratio = rng.normal(0.5, 0.15, n_samples).clip(0.1, 1.2)
    savings_ratio = rng.normal(0.15, 0.10, n_samples).clip(-0.1, 0.5)
    emi_ratio = rng.normal(0.25, 0.12, n_samples).clip(0.0, 0.8)
    loan_to_income = rng.normal(1.5, 1.0, n_samples).clip(0.0, 6.0)
    debt_burden = (emi_ratio + rng.normal(0.05, 0.05, n_samples)).clip(0.0, 1.0)

    expenses = income * expense_ratio
    savings = income * savings_ratio

    return pd.DataFrame({
        "monthly_income": income,
        "monthly_expenses": expenses,
        "monthly_savings": savings,
        "emi_to_income_ratio": emi_ratio,
        "expense_to_income_ratio": expense_ratio,
        "loan_to_income_ratio": loan_to_income,
        "debt_burden_ratio": debt_burden,
    })


def train_anomaly_model(real_profiles: pd.DataFrame = None, contamination: float = 0.05):
    """
    Trains the Isolation Forest model. Combines synthetic baseline
    data with any real user profiles passed in (e.g. from the
    database), so the model improves as more real users are added.

    contamination: the expected proportion of anomalies in the
    data (5% is a common, reasonable default).
    """
    os.makedirs("models", exist_ok=True)

    baseline = _generate_synthetic_baseline()

    if real_profiles is not None and len(real_profiles) > 0:
        training_data = pd.concat(
            [baseline, real_profiles[FEATURE_COLUMNS]], ignore_index=True
        )
    else:
        training_data = baseline

    model = IsolationForest(
        n_estimators=100,      # number of random trees, more = more stable scores
        contamination=contamination,
        random_state=42
    )
    model.fit(training_data[FEATURE_COLUMNS])

    joblib.dump(model, MODEL_PATH)
    print(f"Isolation Forest trained on {len(training_data)} profiles "
          f"({len(baseline)} synthetic + {len(training_data) - len(baseline)} real).")
    return model


def load_anomaly_model():
    """Loads the trained model, or trains a fresh one (synthetic
    baseline only) if none exists yet."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return train_anomaly_model()


def check_anomaly(features: dict, model=None) -> dict:
    """
    Checks a single financial profile (from feature_engineering.py's
    output dict) against the trained model. Returns whether it's
    flagged as an anomaly, plus a human-readable anomaly score.

    'model' can be passed in by a caller that wants to cache/reuse
    an already-loaded model across many calls (e.g. a Streamlit app
    using @st.cache_resource) instead of reloading it from disk every
    time. If omitted, the model is loaded fresh as before - console
    usage (python src/anomaly_detection.py) is unaffected.
    """
    if model is None:
        model = load_anomaly_model()

    row = pd.DataFrame([{col: features[col] for col in FEATURE_COLUMNS}])

    # predict() returns -1 for anomalies, 1 for normal points
    raw_prediction = model.predict(row)[0]
    is_anomaly = raw_prediction == -1

    # decision_function() gives a continuous anomaly score - more
    # negative means MORE anomalous, positive means more typical.
    # We flip and rescale it into an intuitive 0-100 "unusualness" score.
    raw_score = model.decision_function(row)[0]
    unusualness_score = round(max(0, min(100, (0.5 - raw_score) * 100)), 1)

    return {
        "is_anomaly": bool(is_anomaly),
        "unusualness_score": unusualness_score,
        "message": (
            "This financial profile is unusual compared to typical patterns - "
            "worth a closer manual review."
            if is_anomaly else
            "This financial profile looks typical/consistent with common patterns."
        )
    }


# ------------------------------------------------------------
# Quick manual test - only runs when this file is executed
# directly (python src/anomaly_detection.py)
# ------------------------------------------------------------
if __name__ == "__main__":
    from feature_engineering import compute_financial_features

    # Train (or load) the model first
    train_anomaly_model()

    # Test 1: a typical, unremarkable profile
    typical = compute_financial_features(
        monthly_income=35000, monthly_expenses=18000, monthly_savings=5000,
        existing_emi=4000, other_debt=0, loan_amount=150,
        loan_tenure_months=60, credit_history=1.0, dependents=1
    )
    result1 = check_anomaly(typical)
    print("\nTypical profile result:", result1)

    # Test 2: a deliberately unusual profile (huge loan vs tiny income)
    unusual = compute_financial_features(
        monthly_income=8000, monthly_expenses=7500, monthly_savings=0,
        existing_emi=0, other_debt=0, loan_amount=900,
        loan_tenure_months=360, credit_history=0.0, dependents=0
    )
    result2 = check_anomaly(unusual)
    print("Unusual profile result:", result2)