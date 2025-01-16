import uvicorn
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import joblib
import pickle
import pandas as pd
import numpy as np
#create the app object
app=FastAPI()
model = joblib.load("model.pkl","rb")
# Define the input data model, including amount, time, and V1 to V28
class TransactionInput(BaseModel):
    time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    amount: float

@app.post("/predict")
async def predict_fraud(input_data: TransactionInput):
    # Create a DataFrame for the model input with the additional features
    user_input = pd.DataFrame({
        'Time': [input_data.time],
        'V1': [input_data.V1],
        'V2': [input_data.V2],
        'V3': [input_data.V3],
        'V4': [input_data.V4],
        'V5': [input_data.V5],
        'V6': [input_data.V6],
        'V7': [input_data.V7],
        'V8': [input_data.V8],
        'V9': [input_data.V9],
        'V10': [input_data.V10],
        'V11': [input_data.V11],
        'V12': [input_data.V12],
        'V13': [input_data.V13],
        'V14': [input_data.V14],
        'V15': [input_data.V15],
        'V16': [input_data.V16],
        'V17': [input_data.V17],
        'V18': [input_data.V18],
        'V19': [input_data.V19],
        'V20': [input_data.V20],
        'V21': [input_data.V21],
        'V22': [input_data.V22],
        'V23': [input_data.V23],
        'V24': [input_data.V24],
        'V25': [input_data.V25],
        'V26': [input_data.V26],
        'V27': [input_data.V27],
        'V28': [input_data.V28],
        'Amount': [input_data.amount]
    })

    # Predict using the trained model
    prediction = model.predict(user_input)[0]

    # Generate response
    if prediction == 0:
        return {"result": "The transaction is legitimate."}
    else:
        return {"result": "The transaction might be fraudulent."}

risk_model = joblib.load('credit_risk.pkl')  # Logistic Regression risk_model
scaler = joblib.load('scaler.pkl')           # StandardScaler

# Define the input schema
class CreditRiskInput(BaseModel):
    age: float
    ed: int
    employ: float
    address: float
    income: float
    debtinc: float
    creddebt: float
    othdebt: float

# API endpoint for risk assessment
@app.post("/credit-risk")
def assess_credit_risk(input_data: CreditRiskInput):
    # Convert input to numpy array
    features = np.array([
        input_data.age, input_data.ed, input_data.employ, input_data.address,
        input_data.income, input_data.debtinc, input_data.creddebt, input_data.othdebt
    ]).reshape(1, -1)
    
    # Scale the input features
    scaled_features = scaler.transform(features)
    
    # Make prediction
    prediction = risk_model.predict(scaled_features)
    risk_score = "Low Risk" if prediction[0] == 0 else "High Risk"

    # Suggestions for improvement
    suggestions = []
    if risk_score == "High Risk":
        suggestions.append("Reduce your debt-to-income ratio.")
        suggestions.append("Pay off some of your outstanding debts.")
        suggestions.append("Increase your monthly income if possible.")
    else:
        suggestions.append("Maintain your current financial behavior.")

    return {
        "risk_score": risk_score,
        "suggestions": suggestions
    }
