# src/model.py

import joblib

def load_model():
    return joblib.load("models/risk_model.pkl")

def predict_risk(model, features):
    return model.predict([features])[0]
