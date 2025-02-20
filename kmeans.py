import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:, :2]
def kmeans(X, k, max_iterations):
    random_num = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[random_num]
    for _ in range(max_iterations):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return centroids, labels
k = 3
centroids, labels = kmeans(X, k, max_iterations=100)
colors = ['r', 'g', 'b']
for i in range(k):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], c=colors[i], label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='black', label='Centroids')
input_data = np.array([[6, 2]])
input_distances = np.linalg.norm(input_data - centroids, axis=1)
input_cluster = np.argmin(input_distances)
plt.scatter(input_data[0, 0], input_data[0, 1], marker='s', c=colors[input_cluster], label='Input Data')
plt.title('K-means Clustering on Iris Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()
