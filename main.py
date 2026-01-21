from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model.joblib")

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
def predict(data: Transaction):
    df = pd.DataFrame([data.dict()])

    df["type"] = df["type"].map({
        "PAYMENT": 0,
        "TRANSFER": 1,
        "CASH_OUT": 2,
        "DEBIT": 3
    })

    prediction = model.predict(df)[0]

    return {"fraud": bool(prediction)}
