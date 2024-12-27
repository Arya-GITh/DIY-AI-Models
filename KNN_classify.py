import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
digits = load_digits()

X = pd.DataFrame(digits.data)
Y = pd.Series(digits.target)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

X_train = X_train.values
X_test = X_test.values
Y_train = Y_train.values
Y_test = Y_test.values


#helpers
def euclidean(X_train,X_test):
    
    X_train_2 = np.sum(np.square(X_train),axis=1)
    X_test_2 = np.sum(np.square(X_test),axis=1)
    cross = np.dot(X_test,X_train.T)
    dist_2 = X_test_2[:,np.newaxis] + X_train_2 - 2*cross
    dist =  np.sqrt(np.maximum(dist_2,0))
    return dist

#KNN Classifcation
def knn_classify(X_train, X_test, Y_train,k):

    dist = euclidean(X_train,X_test)
    indices = np.argsort(dist,axis=1)[:,:k]
    y_k = Y_train[indices]
    Y_predict = mode(y_k, axis=1).mode.flatten()
    
    return Y_predict

def score(Y_predict,Y_test):
    score = np.mean(Y_predict == Y_test)
    return score

Y_predict = knn_classify(X_train,X_test,Y_train,10)

print(score(Y_predict,Y_test))

k_val = range(1,20)

pred_val = []
sci_val = []

for k in k_val:
    pred_val.append(score(knn_classify(X_train, X_test, Y_train,k), Y_test))
    sci_model = KNeighborsClassifier(k)
    sci_model.fit(X_train,Y_train)
    sci_val.append(accuracy_score(Y_test,sci_model.predict(X_test)))

plt.plot(k_val, sci_val, label =  "scikit")
plt.plot(k_val, pred_val, label = "own", linestyle = "dotted")
plt.ylabel("accuracy")
plt.xlabel("k-value")
plt.title("finding the optimal k")
plt.legend()
plt.show()
