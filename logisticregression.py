from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
def logistic_regression(X, y, num_iterations=200, learning_rate=0.001):
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add bias term
    weights = np.zeros(X.shape[1])  # Initialize weights
    for _ in range(num_iterations):
        z = np.dot(X, weights)  # z = X * weights
        h = sigmoid(z)  # Sigmoid of z
        gradient = np.dot(X.T, (h - y)) / y.shape[0]  # Gradient of loss
        weights -= learning_rate * gradient  # Update weights
    return weights
iris = load_iris()
X = iris.data  # All features
y = iris.target  # All labels
mask = (y == 0) | (y == 1)  # Only keep samples for Setosa and Versicolor
X = X[mask]
y = y[mask]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=9)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
weights = logistic_regression(X_train_std, y_train)
y_pred = sigmoid(np.dot(np.hstack([np.ones((X_test_std.shape[0], 1)), X_test_std]), weights)) > 0.5
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy on test data: {accuracy:.4f}")
sepal_len = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_len = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))
user_input = np.array([sepal_len, sepal_width, petal_len, petal_width]).reshape(1, -1)
user_input_std = sc.transform(user_input)  # Standardize the user input
user_input_std = np.hstack([np.ones((user_input_std.shape[0], 1)), user_input_std])
prediction = sigmoid(np.dot(user_input_std, weights)) > 0.5
print(f"Prediction: {'Setosa' if prediction[0] == 0 else 'Versicolor'}")