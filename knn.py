import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data       # Features (NumPy array)
y = iris.target     # Labels (NumPy array)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# KNN Prediction without pandas
def knn_predict(train_data, train_labels, test_point, k=3):
    # Calculate Euclidean distances between the test point and all training points.
    distances = np.linalg.norm(train_data - test_point, axis=1)
    # Identify indices of the k closest neighbors.
    nearest_indices = distances.argsort()[:k]
    # Get the labels of these neighbors and return the most common one.
    return np.bincount(train_labels[nearest_indices]).argmax()

# Predictions & Accuracy: iterate directly over X_test (a NumPy array)
predictions = [knn_predict(X_train, y_train, row) for row in X_test]
print("Accuracy:", accuracy_score(y_test, predictions))

# Get new data input from the user
sepal_len = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_len = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

# Predict on new data
new_data = np.array([sepal_len, sepal_width, petal_len, petal_width])
print("Prediction for new data:", knn_predict(X_train, y_train, new_data))

'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load data
df = pd.read_csv("iris_csv.csv")
X = df.iloc[:, :4]  # Features
y = df.iloc[:, -1]  # Labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)

# Predict on new data
new_data = [[6.4, 2.7, 5.3, 1.9]]  # New data point for prediction
new_prediction = knn.predict(new_data)
print("Prediction for new data:", new_prediction[0])
'''