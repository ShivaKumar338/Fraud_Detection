from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title="Fraud Detection API")

# ---- Load trained model (when you save it from notebook) ----
# model = joblib.load("model.pkl")

class Transaction(BaseModel):
    amount: float
    feature1: float
    feature2: float
    feature3: float

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(data: Transaction):
    """
    Dummy logic for now.
    Replace with real ML model prediction later.
    """
    features = np.array([[data.amount, data.feature1, data.feature2, data.feature3]])

    # When you have a real model:
    # prediction = model.predict(features)[0]

    prediction = 0  # 0 = Not Fraud, 1 = Fraud (dummy)

    return {
        "fraud": bool(prediction),
        "prediction": int(prediction)
    }
