from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Logistic regression implementation
def logistic_regression(X, y, num_iterations=200, learning_rate=0.001):
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add bias term
    weights = np.zeros(X.shape[1])  # Initialize weights
    for _ in range(num_iterations):
        z = np.dot(X, weights)  # Linear combination
        h = sigmoid(z)  # Sigmoid activation
        gradient = np.dot(X.T, (h - y)) / y.shape[0]  # Compute gradient
        weights -= learning_rate * gradient  # Update weights
    return weights

# Load dataset
iris = load_iris()
X = iris.data  # All features
y = iris.target  # All labels

# Filter dataset for Setosa and Versicolor only
mask = (y == 0) | (y == 1)
X = X[mask]
y = y[mask]

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=9)

# Train logistic regression model
weights = logistic_regression(X_train, y_train)

# Add bias term to test data and make predictions
X_test_with_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
y_pred = sigmoid(np.dot(X_test_with_bias, weights)) > 0.5

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy on test data: {accuracy:.4f}")

# User input for prediction
sepal_len = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_len = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))
user_input = np.array([sepal_len, sepal_width, petal_len, petal_width]).reshape(1, -1)
user_input_with_bias = np.hstack([np.ones((user_input.shape[0], 1)), user_input])
prediction = sigmoid(np.dot(user_input_with_bias, weights)) > 0.5
print(f"Prediction: {'Setosa' if not prediction[0] else 'Versicolor'}")
