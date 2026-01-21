from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Fraud Detection API")

# Load model & columns
model = joblib.load("fraud_model.joblib")
model_columns = joblib.load("model_columns.joblib")

class Transaction(BaseModel):
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrg: float
    oldbalanceDest: float
    newbalanceDest: float

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(txn: Transaction):
    data = {
        "amount": txn.amount,
        "oldbalanceOrg": txn.oldbalanceOrg,
        "newbalanceOrg": txn.newbalanceOrg,
        "oldbalanceDest": txn.oldbalanceDest,
        "newbalanceDest": txn.newbalanceDest,
        f"type_{txn.type}": 1
    }

    row = pd.DataFrame([data])

    for col in model_columns:
        if col not in row:
            row[col] = 0

    row = row[model_columns]

    prediction = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]

    return {
        "fraud": bool(prediction),
        "probability": round(float(prob), 4)
    }
