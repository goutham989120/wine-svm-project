from fastapi import FastAPI
from app.schema import WineInput
from app.model import predict_wine

app = FastAPI(title="Wine Classification API")

@app.get("/")
def home():
    return {"message": "Wine SVM API is running"}

@app.post("/predict")
def predict(data: WineInput):
    result = predict_wine(data.features)
    return result