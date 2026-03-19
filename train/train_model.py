from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle
import os

os.makedirs("model", exist_ok=True)

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)

pickle.dump(model, open("model/svm_model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

print("Model trained & saved!")