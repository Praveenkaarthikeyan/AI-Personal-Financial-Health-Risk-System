import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.linear_model import LogisticRegression
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

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "models/logistic_regression.pkl")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("===== Logistic Regression Results =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print("\nFull classification report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rejected", "Approved"],
            yticklabels=["Rejected", "Approved"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_logistic_regression.png")
plt.close()

with open("outputs/logistic_regression_results.txt", "w") as f:
    f.write("Logistic Regression Results\n============================\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1-score : {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))

print("\n✅ Logistic Regression training complete.")
print("Saved: models/logistic_regression.pkl")
print("Saved: outputs/confusion_matrix_logistic_regression.png")
print("Saved: outputs/logistic_regression_results.txt")