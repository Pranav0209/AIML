from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def logistic_regression(X,y,max_iter=2000,learning_rate=0.001):
    X=np.hstack([np.ones((X.shape[0],1)),X])
    weights=np.zeros(X.shape[1])
    for _ in range(max_iter):
        z=np.dot(X,weights)
        h=sigmoid(z)
        gradient=np.dot(X.T,(h-y))/y.shape[0]
        weights-=learning_rate*gradient
    return weights
iris=load_iris()
X=iris.data
y=iris.target
mask=(y==0) | (y==1)
X=X[mask]
y=y[mask]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
sc=StandardScaler()
X_train_std=sc.fit_transform(X_train)
X_test_std=sc.transform(X_test)
weights=logistic_regression(X_train_std,y_train)
y_pred=sigmoid(np.dot(np.hstack([np.ones((X_test.shape[0],1)),X_test_std]),weights))>0.5
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy: ",accuracy)
sepal_len=float(input("Enter sepal length: "))
sepal_width=float(input("Enter sepal width: "))
petal_len=float(input("Enter petal length: "))
petal_width=float(input("Enter petal width: "))
user_input=np.array([sepal_len,sepal_width,petal_len,petal_width]).reshape(1,-1)
user_input_std=sc.transform(user_input)
user_input_std=np.hstack([np.ones((user_input_std.shape[0],1)),user_input_std])
prediction=sigmoid(np.dot(user_input_std,weights))>0.5
print(f"Prediction: {'Setosa' if prediction[0]==0 else 'Versicolor'}")