import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load model
model = pickle.load(open("svm_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🍷 Wine Classification using SVM")

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
    data = np.array(inputs).reshape(1, -1)
    data = scaler.transform(data)
    
    prediction = model.predict(data)
    probability = model.predict_proba(data)

    st.success(f"Predicted Class: {prediction[0]}")
    st.write(f"Confidence: {max(probability[0]):.2f}")

# --- SVM decision boundary plot (2D example) ---
st.header("SVM Decision Boundary (2D sample)")

wine = load_wine()
X = wine.data[:, :2]  # Alcohol, Malic Acid
y = wine.target

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler2 = StandardScaler().fit(X_train2)
X_train2s = scaler2.transform(X_train2)
X_test2s = scaler2.transform(X_test2)

svm2d = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm2d.fit(X_train2s, y_train2)
score2d = svm2d.score(X_test2s, y_test2)

x_min, x_max = X_train2s[:, 0].min() - 1, X_train2s[:, 0].max() + 1
y_min, y_max = X_train2s[:, 1].min() - 1, X_train2s[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = svm2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
scatter = ax.scatter(X_train2s[:, 0], X_train2s[:, 1], c=y_train2, s=40, edgecolor='k', cmap='coolwarm')
ax.set_xlabel('Alcohol (scaled)')
ax.set_ylabel('Malic Acid (scaled)')
ax.set_title(f'Wine SVM decision boundary (2D), test accuracy {score2d:.3f}')

st.pyplot(fig)

st.write("Using features: Alcohol and Malic Acid (scaled, 2D slice).")

# --- PCA and t-SNE visualization for 13D data -> 2D ---
st.header("PCA and t-SNE Projection (13D → 2D)")

wine = load_wine()
X_full = wine.data
y_full = wine.target
feature_names = wine.feature_names

# Standardize full feature set
scaler_full = StandardScaler().fit(X_full)
X_full_s = scaler_full.transform(X_full)

# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_full_s)

fig, ax = plt.subplots(figsize=(8, 6))
for class_id, class_name in enumerate(wine.target_names):
    idx = y_full == class_id
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], label=class_name, alpha=0.7, s=40)
ax.set_title('Wine dataset PCA (2D)')
ax.set_xlabel('PCA-1')
ax.set_ylabel('PCA-2')
ax.legend()
st.pyplot(fig)

# t-SNE
tsne = TSNE(n_components=2, random_state=42, learning_rate='auto', init='pca')
X_tsne = tsne.fit_transform(X_full_s)

fig, ax = plt.subplots(figsize=(8, 6))
for class_id, class_name in enumerate(wine.target_names):
    idx = y_full == class_id
    ax.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=class_name, alpha=0.7, s=40)
ax.set_title('Wine dataset t-SNE (2D)')
ax.set_xlabel('t-SNE-1')
ax.set_ylabel('t-SNE-2')
ax.legend()
st.pyplot(fig)

st.write('PCA and t-SNE show a 2D projection of the full 13-feature wine dataset.')
