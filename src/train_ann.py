import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

tf.random.set_seed(42)
np.random.seed(42)

X_train = pd.read_csv("outputs/X_train.csv")
X_test = pd.read_csv("outputs/X_test.csv")
y_train = pd.read_csv("outputs/y_train.csv").values.ravel()
y_test = pd.read_csv("outputs/y_test.csv").values.ravel()

input_dim = X_train.shape[1]

# Architecture: Input -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Sigmoid)
model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

model.save("models/ann_model.keras")

# Plot Accuracy and Loss curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history.history["accuracy"], label="Train Accuracy")
axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
axes[0].set_title("ANN Accuracy over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Validation Loss")
axes[1].set_title("ANN Loss over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/ann_accuracy_loss_curves.png")
plt.close()

# Evaluate ANN on test set
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).ravel()

ann_acc = accuracy_score(y_test, y_pred)
ann_prec = precision_score(y_test, y_pred)
ann_rec = recall_score(y_test, y_pred)
ann_f1 = f1_score(y_test, y_pred)

print("\n===== ANN Results =====")
print(f"Accuracy : {ann_acc:.4f}")
print(f"Precision: {ann_prec:.4f}")
print(f"Recall   : {ann_rec:.4f}")
print(f"F1-score : {ann_f1:.4f}")

# Compare ANN with Random Forest (from Unit 2)
rf_model = joblib.load("models/random_forest.pkl")
rf_pred = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred)
rf_rec = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

comparison_df = pd.DataFrame([
    {"Model": "Random Forest (Tuned)", "Accuracy": rf_acc, "Precision": rf_prec,
     "Recall": rf_rec, "F1-score": rf_f1},
    {"Model": "ANN", "Accuracy": ann_acc, "Precision": ann_prec,
     "Recall": ann_rec, "F1-score": ann_f1},
])

comparison_df.to_csv("outputs/unit3_ann_vs_rf_comparison.csv", index=False)
print("\nANN vs Random Forest Comparison:")
print(comparison_df)

comparison_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
    kind="bar", figsize=(8, 5)
)
plt.title("ANN vs Random Forest - Metric Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/unit3_ann_vs_rf_chart.png")
plt.close()

print("\n✅ Unit 3 (ANN) training complete.")
print("Saved: models/ann_model.keras")
print("Saved: outputs/ann_accuracy_loss_curves.png")
print("Saved: outputs/unit3_ann_vs_rf_comparison.csv")
print("Saved: outputs/unit3_ann_vs_rf_chart.png")