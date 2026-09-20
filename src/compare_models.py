import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("outputs/cleaned_data_full.csv")

target = "Loan_Status"

common_cols = [c for c in df.columns if c not in
               ["ApplicantIncome", "CoapplicantIncome", "TotalIncome",
                "TotalIncome_log", "LoanAmount", "LoanAmount_log", target]]


def build_and_evaluate(feature_cols, label):
    """Trains a fresh Logistic Regression using given feature columns,
    returns accuracy and F1-score. Used to compare before vs after log."""
    X = df[feature_cols]
    y = df[target]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"{label}: Accuracy = {acc:.4f}, F1-score = {f1:.4f}")
    return {"Version": label, "Accuracy": acc, "F1-score": f1}


# Experiment A: BEFORE log transformation (raw income/loan values)
before_cols = common_cols + ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
before_result = build_and_evaluate(before_cols, "Before Log Transform")

# Experiment B: AFTER log transformation
after_cols = common_cols + ["TotalIncome_log", "LoanAmount_log"]
after_result = build_and_evaluate(after_cols, "After Log Transform")

# Save comparison table and bar chart
comparison_df = pd.DataFrame([before_result, after_result])
comparison_df.to_csv("outputs/log_transform_comparison.csv", index=False)

print("\nLog Transformation Comparison Table:")
print(comparison_df)

comparison_df.set_index("Version")[["Accuracy", "F1-score"]].plot(
    kind="bar", figsize=(6, 5), color=["#4C72B0", "#DD8452"]
)
plt.title("Effect of Log Transformation on Model Performance")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/log_transform_comparison_chart.png")
plt.close()

# Visual proof: income distribution before vs after log
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(df["TotalIncome"], bins=40, color="salmon")
axes[0].set_title("TotalIncome (Before Log) - Skewed")
axes[1].hist(df["TotalIncome_log"], bins=40, color="seagreen")
axes[1].set_title("TotalIncome (After Log) - Closer to Normal")
plt.tight_layout()
plt.savefig("outputs/income_distribution_before_after_log.png")
plt.close()

print("\n✅ Log transformation comparison complete.")
print("Saved: outputs/log_transform_comparison.csv")
print("Saved: outputs/log_transform_comparison_chart.png")
print("Saved: outputs/income_distribution_before_after_log.png")