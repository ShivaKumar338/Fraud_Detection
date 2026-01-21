import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("Fraud.csv")

# Drop useless columns
df = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud", "step"])

# One-hot encode transaction type
df = pd.get_dummies(df, columns=["type"], drop_first=True)

# Features & target
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, model.predict(X_test)))

# Save model
joblib.dump(model, "fraud_model.joblib")
joblib.dump(X.columns.tolist(), "model_columns.joblib")

print("Model saved!")
