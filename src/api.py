import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Literal
import shap

app = FastAPI(title="Churn Prediction API")
BASE = Path(__file__).parent
model = joblib.load(BASE / "churn_model.joblib")
columns = joblib.load(BASE / "model_columns.joblib")
explainer = shap.TreeExplainer(model)

SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


YesNo = Literal["Yes", "No"]
NetService = Literal["Yes", "No", "No internet service"]

class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=72)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: NetService
    OnlineBackup: NetService
    DeviceProtection: NetService
    TechSupport: NetService
    StreamingTV: NetService
    StreamingMovies: NetService
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

def prepare(customer: Customer) -> pd.DataFrame:
    df = pd.DataFrame([customer.model_dump()])
    df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                                labels=["0-12", "13-24", "25-48", "49+"]).astype(str)
    df["avg_monthly_spend"] = np.where(df["tenure"] > 0,
                                       df["TotalCharges"] / df["tenure"].replace(0, 1),
                                       df["MonthlyCharges"])
    df["num_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    df = pd.get_dummies(df)
    return df.reindex(columns=columns, fill_value=0).astype(float)


@app.post("/predict")
def predict(customer: Customer):
    X = prepare(customer)
    prob = float(model.predict_proba(X)[0, 1])

    contrib = pd.Series(explainer.shap_values(X)[0], index=X.columns).sort_values()
    reasons_to_leave = [{"feature": k, "impact": round(float(v), 3)}
                        for k, v in contrib.iloc[::-1].head(3).items() if v > 0]
    reasons_to_stay = [{"feature": k, "impact": round(float(v), 3)}
                       for k, v in contrib.head(2).items() if v < 0]

    return {
        "churn_probability": round(prob, 3),
        "risk": "high" if prob >= 0.5 else "low",
        "reasons_to_leave": reasons_to_leave,
        "reasons_to_stay": reasons_to_stay,
    }