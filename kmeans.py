import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris=load_iris()
X=iris.data[:,:2]
def kmeans(X,k,num_iter):
    random_num=np.random.choice(X.shape[0],k,replace=False)
    centroids=X[random_num]
    for _ in range (num_iter):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels=np.argmin(distances,axis=1)
        centroids=np.array([X[labels==i].mean(axis=0) for i in range(k)])
    return centroids,labels
k=3
centroids,labels=kmeans(X,k,num_iter=100)
colors=['r','g','b']
for i in range(k):
    plt.scatter(X[labels==i,0],X[labels==i,1],c=colors[i],label=f'Cluster{i+1}')
sepal_len=float(input("Enter sepal length: "))
sepal_width=float(input("Enter sepal width: "))
new_data=np.array([sepal_len,sepal_width]).reshape(1,-1)
input_distances=np.linalg.norm(new_data-centroids,axis=1)
input_cluster=np.argmin(input_distances)
plt.scatter(new_data[0][0],new_data[0][1],marker='s',c=colors[input_cluster],label='Input data')
plt.scatter(centroids[:,0],centroids[:,1],c='black',marker='X',label='Centroids')
plt.title("Kmeans on IRIS dataset")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.legend()
plt.show()