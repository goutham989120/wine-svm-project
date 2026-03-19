import logging
import pickle
import numpy as np

logger = logging.getLogger("wine-svm-api.model")

model = pickle.load(open("../svm_model.pkl", "rb"))
scaler = pickle.load(open("../scaler.pkl", "rb"))

def predict_wine(features: list[float]):
    logger.debug("Original features: %s", features)
    data = np.array(features).reshape(1, -1)

    try:
        data_scaled = scaler.transform(data)
    except Exception as e:
        logger.exception("Scaler transform failed")
        raise

    try:
        prediction = model.predict(data_scaled)
        probabilities = model.predict_proba(data_scaled)
    except Exception as e:
        logger.exception("Model prediction failed")
        raise

    result = {
        "prediction": int(prediction[0]),
        "probabilities": probabilities[0].tolist(),
    }
    logger.info("predict_wine result: %s", result)
    return result