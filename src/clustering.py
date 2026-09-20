# ============================================================
# clustering.py
# Purpose: Group financial profiles into behavior clusters
#          (Healthy Saver, Balanced, High Spender, Debt Heavy)
#          using K-Means, an UNSUPERVISED learning technique.
#
# HOW K-MEANS WORKS (high level):
# 1. Pick K starting points ("centroids") - here K=4.
# 2. Assign every data point to its nearest centroid.
# 3. Recalculate each centroid as the average of its assigned points.
# 4. Repeat steps 2-3 until the groups stop changing.
#
# IMPORTANT: clustering is DESCRIPTIVE, not predictive. It finds
# natural groupings in the data - it does NOT know in advance
# what "Healthy Saver" or "Debt Heavy" means. We compute each
# cluster's average characteristics AFTER clustering and assign
# the label that best matches what that group actually looks like.
#
# Like anomaly_detection.py, this trains on a synthetic baseline
# combined with real user profiles from the database, since we
# don't have historical financial-behavior data to start from.
# ============================================================

import numpy as np
import pandas as pd
import joblib
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

MODEL_PATH = os.path.join("models", "kmeans_cluster.pkl")
SCALER_PATH = os.path.join("models", "kmeans_scaler.pkl")
LABELS_PATH = os.path.join("models", "kmeans_labels.pkl")

# Features used to determine spending/saving behavior
CLUSTER_FEATURES = [
    "savings_rate", "expense_to_income_ratio",
    "emi_to_income_ratio", "debt_burden_ratio"
]


def _generate_synthetic_baseline(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic financial profiles as FOUR distinct
    sub-populations, each centered around one of the target
    behavior types (Healthy Saver, Balanced, High Spender, Debt
    Heavy). This gives K-Means well-separated groups to discover,
    so the resulting clusters actually align with meaningful,
    interpretable financial behaviors - rather than one big random
    blob, which produces mathematically valid but confusing clusters.
    """
    rng = np.random.default_rng(seed)
    per_group = n_samples // 4

    def make_group(savings_mu, expense_mu, emi_mu, debt_mu, n):
        savings = rng.normal(savings_mu, 0.05, n)
        expense = rng.normal(expense_mu, 0.08, n)
        emi = rng.normal(emi_mu, 0.08, n).clip(0.0, 1.0)
        debt = (emi + rng.normal(debt_mu - emi_mu, 0.04, n)).clip(0.0, 1.1)
        return savings, expense, emi, debt

    # Healthy Saver: high savings, moderate expenses, low debt
    s1, e1, m1, d1 = make_group(0.30, 0.45, 0.10, 0.12, per_group)
    # Balanced: moderate everything
    s2, e2, m2, d2 = make_group(0.15, 0.55, 0.20, 0.25, per_group)
    # High Spender: low savings, high expenses, moderate debt
    s3, e3, m3, d3 = make_group(0.05, 0.75, 0.20, 0.28, per_group)
    # Debt Heavy: low/negative savings, high EMI and debt burden
    s4, e4, m4, d4 = make_group(-0.02, 0.55, 0.45, 0.60, n_samples - 3 * per_group)

    savings_rate = np.concatenate([s1, s2, s3, s4]).clip(-0.2, 0.6)
    expense_ratio = np.concatenate([e1, e2, e3, e4]).clip(0.1, 1.3)
    emi_ratio = np.concatenate([m1, m2, m3, m4]).clip(0.0, 0.95)
    debt_burden = np.concatenate([d1, d2, d3, d4]).clip(0.0, 1.1)

    return pd.DataFrame({
        "savings_rate": savings_rate,
        "expense_to_income_ratio": expense_ratio,
        "emi_to_income_ratio": emi_ratio,
        "debt_burden_ratio": debt_burden,
    })


def _label_clusters(model: KMeans, scaler: StandardScaler) -> dict:
    """
    Looks at each cluster's centroid (in original, unscaled units)
    and assigns a human-readable label based on its characteristics.
    This is the "interpretation" step - K-Means itself has no
    concept of these labels.
    """
    # Convert centroids back from scaled space to original ratio units
    centroids = scaler.inverse_transform(model.cluster_centers_)
    centroid_df = pd.DataFrame(centroids, columns=CLUSTER_FEATURES)

    labels = {}
    for i, row in centroid_df.iterrows():
        savings = row["savings_rate"]
        expense = row["expense_to_income_ratio"]
        debt = row["debt_burden_ratio"]

        # Simple rule-based interpretation of each cluster's centroid
        if debt > 0.45:
            labels[i] = "Debt Heavy"
        elif expense > 0.65:
            labels[i] = "High Spender"
        elif savings > 0.20:
            labels[i] = "Healthy Saver"
        else:
            labels[i] = "Balanced"

    return labels


def train_cluster_model(real_profiles: pd.DataFrame = None, n_clusters: int = 4):
    """Trains the K-Means model, fits a scaler, and derives
    human-readable labels for each resulting cluster."""
    os.makedirs("models", exist_ok=True)

    baseline = _generate_synthetic_baseline()
    if real_profiles is not None and len(real_profiles) > 0:
        training_data = pd.concat(
            [baseline, real_profiles[CLUSTER_FEATURES]], ignore_index=True
        )
    else:
        training_data = baseline

    # Scale features first - K-Means uses distance calculations,
    # so features need to be on comparable scales (same reasoning
    # as Min-Max Scaling in Unit 1, just a different scaler here).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(training_data[CLUSTER_FEATURES])

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(X_scaled)

    labels = _label_clusters(model, scaler)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(labels, LABELS_PATH)

    print(f"K-Means trained on {len(training_data)} profiles into {n_clusters} clusters.")
    print("Cluster interpretations:", labels)
    return model, scaler, labels


def load_cluster_model():
    """Loads the trained model/scaler/labels, or trains fresh ones
    if none exist yet."""
    if all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, LABELS_PATH]):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        labels = joblib.load(LABELS_PATH)
        return model, scaler, labels
    return train_cluster_model()


def assign_cluster(features: dict, cached_artifacts: tuple = None) -> dict:
    """
    Assigns a single financial profile to its nearest cluster and
    returns the human-readable behavior label.

    'cached_artifacts' can be a pre-loaded (model, scaler, labels)
    tuple, passed in by a caller that caches it (e.g. a Streamlit
    app using @st.cache_resource) instead of reloading from disk on
    every call. If omitted, artifacts are loaded fresh as before.
    """
    if cached_artifacts is not None:
        model, scaler, labels = cached_artifacts
    else:
        model, scaler, labels = load_cluster_model()

    row = pd.DataFrame([{col: features[col] for col in CLUSTER_FEATURES}])
    row_scaled = scaler.transform(row)

    cluster_id = model.predict(row_scaled)[0]
    cluster_label = labels[cluster_id]

    return {
        "cluster_id": int(cluster_id),
        "cluster_label": cluster_label,
    }


# ------------------------------------------------------------
# Quick manual test - only runs when this file is executed
# directly (python src/clustering.py)
# ------------------------------------------------------------
if __name__ == "__main__":
    from feature_engineering import compute_financial_features

    train_cluster_model()

    # Test 1: a strong saver profile
    saver = compute_financial_features(
        monthly_income=50000, monthly_expenses=20000, monthly_savings=15000,
        existing_emi=3000, other_debt=0, loan_amount=100,
        loan_tenure_months=60, credit_history=1.0, dependents=1
    )
    print("\nSaver profile cluster:", assign_cluster(saver))

    # Test 2: a debt-heavy profile
    debt_heavy = compute_financial_features(
        monthly_income=30000, monthly_expenses=15000, monthly_savings=500,
        existing_emi=10000, other_debt=5000, loan_amount=200,
        loan_tenure_months=36, credit_history=0.0, dependents=2
    )
    print("Debt-heavy profile cluster:", assign_cluster(debt_heavy))