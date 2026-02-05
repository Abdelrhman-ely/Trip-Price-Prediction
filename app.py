from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field

app= FastAPI(title= "Trip Price Prediction", docs_url="/docs")

model= joblib.load("best_xgb.pkl")
preprocessor= joblib.load("preprocessor.pkl")


class TripFeatures(BaseModel):
    distance: float = Field(gt= 0)
    cab_type: str
    source: str
    destination: str
    surge_multiplier: float = Field(ge= 1)
    name: str
    hour: int = Field(ge= 0, le= 23)
    temp: float
    clouds: float
    pressure: float
    humidity: float
    wind: float
    month: int = Field(ge= 1, le= 12)
    day: int = Field(ge= 1, le= 31)


@app.get("/")
def root():
    return{"status": "API is running"}

@app.post("/predict")
def price_predict(trip: TripFeatures):
    try:
        row= trip.model_dump()
        X= pd.DataFrame([row])
        pred= model.predict(X)
        return{"Predicted Price": float(pred[0])}

    except Exception as e:
        raise HTTPException(status_code=400, detail= str(e))
