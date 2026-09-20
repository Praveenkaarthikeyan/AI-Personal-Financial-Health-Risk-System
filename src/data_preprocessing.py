import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/train.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Drop Loan_ID - it is just a unique identifier, has no predictive value
df = df.drop("Loan_ID", axis=1)

# Fill missing values in categorical columns with the mode (most frequent value)
categorical_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill missing values in numeric columns with the median (robust to outliers)
numeric_cols = ["LoanAmount", "Loan_Amount_Term"]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing values AFTER cleaning (should all be 0):")
print(df.isnull().sum())
# Fix 'Dependents' column - it has a text value '3+' which we convert to '3'
df["Dependents"] = df["Dependents"].replace("3+", "3")

# Label Encode binary/simple categorical columns
label_cols = ["Gender", "Married", "Education", "Self_Employed", "Dependents"]
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# One-Hot Encode Property_Area (no natural order between Urban/Semiurban/Rural)
df = pd.get_dummies(df, columns=["Property_Area"], drop_first=True)

# Encode target column: Y -> 1, N -> 0
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Save the label encoders so we can reuse them later if needed
joblib.dump(label_encoders, "models/label_encoders.pkl")

print("\nData after encoding:")
print(df.head())
print(df.dtypes)
# Feature Engineering: combine both incomes into one column
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

# Log Transformation (novelty): income and loan amount are heavily
# right-skewed (a few very high earners distort the distribution).
# log1p = log(1 + x), which safely handles any zero values.
df["TotalIncome_log"] = np.log1p(df["TotalIncome"])
df["LoanAmount_log"] = np.log1p(df["LoanAmount"])

# Save the full cleaned dataset (with both raw and log columns) for later
# comparison experiments (before vs after log transform)
df.to_csv("outputs/cleaned_data_full.csv", index=False)

print("\nData after feature engineering and log transform:")
print(df[["ApplicantIncome", "CoapplicantIncome", "TotalIncome", "TotalIncome_log", "LoanAmount", "LoanAmount_log"]].head())
# Build final feature set: drop raw income/loan columns, keep log versions
features_to_drop = ["ApplicantIncome", "CoapplicantIncome", "TotalIncome", "LoanAmount"]
df_model = df.drop(columns=features_to_drop)

X = df_model.drop("Loan_Status", axis=1)
y = df_model["Loan_Status"]

# Min-Max Scaling: scales every numeric feature to a 0-1 range so that
# large-range columns (like income) don't dominate small-range columns
# (like Credit_History, which is just 0 or 1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Save the scaler so the SAME scaling can be reused later on new data
joblib.dump(scaler, "models/scaler.pkl")

print("\nScaled features preview:")
print(X_scaled.head())
# Train-Test Split (80:20): 80% trains the model, 20% tests it on
# unseen data. random_state=42 makes the split reproducible every run.
# stratify=y keeps the same approve/reject ratio in both splits.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Save the splits so every other script (Logistic Regression, Decision
# Tree, Random Forest, ANN) can reuse the exact same train/test data
X_train.to_csv("outputs/X_train.csv", index=False)
X_test.to_csv("outputs/X_test.csv", index=False)
y_train.to_csv("outputs/y_train.csv", index=False)
y_test.to_csv("outputs/y_test.csv", index=False)

print("\n✅ Preprocessing complete.")
print(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")
print("Saved: outputs/X_train.csv, X_test.csv, y_train.csv, y_test.csv")
print("Saved: models/scaler.pkl, models/label_encoders.pkl")
