import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
digits = load_digits()

X = pd.DataFrame(digits.data)
Y = pd.Series(digits.target)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

X_train = X_train.values
X_test = X_test.values



#helpers
def euclidean(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

#KNN Classifcation
def knn_classify(X_train, X_test, Y_train,k):
    Y_predict = []
    for x in X_test:
        dist = []
        for z in X_train:
            dist.append(euclidean(x,z))
        dist = sorted(list(enumerate(dist)), key = lambda a:a[1])
        k_indices = [pair[0] for pair in dist[:k]]
        y_k = Y_train.iloc[k_indices]
        Y_predict.append(y_k.mode().iloc[0])
    
    return Y_predict

def score(Y_predict,Y_test):
    sum = 0
    for x,y in zip(Y_predict,Y_test.values.ravel()):
        if x == y:
            sum += 1
    score = sum/len(Y_test)
    return score

Y_predict = knn_classify(X_train,X_test,Y_train,10)

print(score(Y_predict,Y_test))

k_val = [8,9,10,11,12]

pred_val = []
for k in k_val:
    pred_val.append(score(knn_classify(X_train, X_test, Y_train,k), Y_test))

plt.scatter(k_val,pred_val)
plt.show()
