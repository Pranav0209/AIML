import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("iris_csv.csv")
X = df.iloc[:, :4]
y = df.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# KNN Prediction
def knn_predict(train_data, train_labels, test_point, k=3):
    distances = np.linalg.norm(train_data - test_point, axis=1)
    return train_labels.iloc[distances.argsort()[:k]].mode()[0]

# Predictions & Accuracy
predictions = [knn_predict(X_train, y_train, row) for _, row in X_test.iterrows()]
print("Accuracy:", accuracy_score(y_test, predictions))

# Predict on new data
new_data = [12.2,8.4,10.1,2]
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