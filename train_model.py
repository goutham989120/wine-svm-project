from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

data = load_wine()
X = data.data
y = data.target
# print(y)
# print(X)

#Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Scaling
scaler = StandardScaler()

# keep pre-scaling row for inspection
X_train_before = X_train[0].copy()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# print only first row before and after scaling
print("Row 0 of X_train before scaling:\n", X_train_before)
print("Row 0 of X_train after scaling:\n", X_train[0])

#Train model
model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)

#Save model
pickle.dump(model, open("svm_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model trained successfully")