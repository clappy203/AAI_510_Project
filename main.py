from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained XGBoost model
model = joblib.load("xgb_model.pkl")

# Define FastAPI app
app = FastAPI(title="XGBoost Readmission Risk API")

# Input schema for prediction
class ReadmissionInput(BaseModel):
    feature_vector: list[float]

# Mapping for human-readable results
label_mapping = {
    0: "No readmission within 30 days (low risk)",
    1: "Readmission within 30 days (high risk)"
}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "XGBoost model is live. Use /predict to get readmission risk."}

# Prediction endpoint
@app.post("/predict")
def predict_readmission(input_data: ReadmissionInput):
    try:
        features = np.array([input_data.feature_vector])
        prediction = int(model.predict(features)[0])
        probability = model.predict_proba(features)[0].tolist()
        
        return {
            "prediction": prediction,
            "label": label_mapping[prediction],
            "probability": {
                "No readmission (<30 days)": round(probability[0], 4),
                "Readmission (<30 days)": round(probability[1], 4)
            }
        }
    except Exception as e:
        return {"error": str(e)}
