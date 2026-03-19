import streamlit as st
import requests

st.title("🍷 Wine Class Predictor (Production)")

labels = [
    "Alcohol", "Malic Acid", "Ash", "Alcalinity", "Magnesium",
    "Total Phenols", "Flavanoids", "Nonflavanoid Phenols",
    "Proanthocyanins", "Color Intensity", "Hue",
    "OD280/OD315", "Proline"
]

inputs = []

for label in labels:
    val = st.number_input(label, value=0.0)
    inputs.append(val)

if st.button("Predict"):
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"features": inputs}
    )

    result = response.json()

    if "error" in result:
        st.error(result["error"])
    else:
        st.success(f"Prediction: {result['prediction']}")
        st.write(f"Confidence: {result['confidence']:.2f}")