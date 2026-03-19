import logging
from fastapi import FastAPI
from app.schema import WineInput
from app.model import predict_wine

# logging setup
LOG_FILE = "wine_api.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=LOG_FILE,
    filemode="a",  # append
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger("wine-svm-api")

app = FastAPI(title="Wine Classification API")

@app.on_event("startup")
def startup():
    logger.info("Starting Wine SVM API")

@app.get("/")
def home():
    logger.info("Health check called")
    return {"message": "Wine SVM API is running"}

@app.post("/predict")
def predict(data: WineInput):
    logger.info("Predict called with features=%s", data.features)
    result = predict_wine(data.features)
    logger.info("Prediction result: %s", result)
    return result

@app.on_event("shutdown")
def shutdown():
    logger.info("Shutting down Wine SVM API")