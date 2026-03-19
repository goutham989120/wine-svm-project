import numpy as np
import pickle

# Load model once
model = pickle.load(open("model/svm_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

def predict_wine(features):
    try:
        data = np.array(features).reshape(1, -1)
        data = scaler.transform(data)

        pred = model.predict(data)
        prob = model.predict_proba(data)

        return {
            "prediction": int(pred[0]),
            "confidence": float(max(prob[0]))
        }
    except Exception as e:
        return {"error": str(e)}