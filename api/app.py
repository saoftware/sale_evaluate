from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
import os

app = FastAPI(title="ML Models API")

# -----------------------
# Chargement des modèles
# -----------------------
BASE_DIR = "../model"

try:
    sales_model = joblib.load(os.path.join(BASE_DIR, "sales_model.pkl"))
    fraud_model = joblib.load(os.path.join(BASE_DIR, "fraud_model.pkl"))
except Exception as e:
    raise RuntimeError(f"Erreur chargement modèles: {e}")


# -----------------------
# Schémas d'entrée
# -----------------------
class SalesInput(BaseModel):
    features: list[float]


class FraudInput(BaseModel):
    features: list[float]


# -----------------------
# Utils sécurité
# -----------------------
def safe_predict(model, features):
    try:
        data = np.array(features).reshape(1, -1)
        pred = model.predict(data)
        return pred[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -----------------------
# Routes
# -----------------------
@app.get("/")
def home():
    return {"message": "API ML OK 🚀"}


@app.post("/predict/sales")
def predict_sales(input: SalesInput):
    prediction = safe_predict(sales_model, input.features)
    return {"sales_prediction": float(prediction)}


@app.post("/predict/fraud")
def predict_fraud(input: FraudInput):
    prediction = safe_predict(fraud_model, input.features)
    return {"fraud_prediction": int(prediction)}