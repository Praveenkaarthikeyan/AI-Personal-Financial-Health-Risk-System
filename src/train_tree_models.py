import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

X_train = pd.read_csv("outputs/X_train.csv")
X_test = pd.read_csv("outputs/X_test.csv")
y_train = pd.read_csv("outputs/y_train.csv").values.ravel()
y_test = pd.read_csv("outputs/y_test.csv").values.ravel()


def evaluate_model(name, model, X_test, y_test):
    """Predicts, prints metrics, saves confusion matrix image,
    and returns a dict of scores. Reused for both models below."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n===== {name} Results =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"outputs/confusion_matrix_{safe_name}.png")
    plt.close()

    return {"Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1-score": f1}


# Decision Tree Classifier
# max_depth=5 keeps the tree simple and reduces overfitting risk
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
joblib.dump(dt_model, "models/decision_tree.pkl")
dt_scores = evaluate_model("Decision Tree", dt_model, X_test, y_test)

# Random Forest Classifier (default settings, before tuning)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Hyperparameter tuning using GridSearchCV
# Tries every combination in param_grid, picks the best via 5-fold cross-validation
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("\nBest parameters found by GridSearchCV:")
print(grid_search.best_params_)

best_rf_model = grid_search.best_estimator_
joblib.dump(best_rf_model, "models/random_forest.pkl")

rf_scores = evaluate_model("Random Forest (Tuned)", best_rf_model, X_test, y_test)

# Compare Decision Tree vs Random Forest: table + bar chart
results_df = pd.DataFrame([dt_scores, rf_scores])
results_df.to_csv("outputs/unit2_model_comparison.csv", index=False)

print("\nModel Comparison Table:")
print(results_df)

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
results_df.set_index("Model")[metrics].plot(kind="bar", figsize=(8, 5))
plt.title("Decision Tree vs Random Forest - Metric Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/unit2_model_comparison_chart.png")
plt.close()

print("\n✅ Unit 2 training complete.")
print("Saved: models/decision_tree.pkl, models/random_forest.pkl")
print("Saved: outputs/unit2_model_comparison.csv")
print("Saved: outputs/unit2_model_comparison_chart.png")