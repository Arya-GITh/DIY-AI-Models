
"""
Using LDA as a  simple classifier
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np


def lin_dis_a(X_train,y_train,X_test):
    n_feature = X_train.shape[1]
    classes = np.unique(y_train)
    n_class = 3
    n_comp = n_class - 1
    total_mean = np.mean(X_train,axis=0)
    Sw = np.zeros((n_feature,n_feature))
    Sb = np.zeros((n_feature,n_feature))

    for cls in classes:
        X_train_class = X_train[y_train == cls]
        cls_mean = np.mean(X_train_class,axis=0)
        cls_size = X_train_class.shape[0]

        Sw += np.matmul((X_train_class - cls_mean).T, (X_train_class - cls_mean))
        mean_d = (cls_mean - total_mean).reshape(n_feature,1)
        Sb += cls_size * np.dot(mean_d,mean_d.T)

    Sw_inv = np.linalg.inv(Sw)
    s = np.dot(Sw_inv,Sb)
    evals, evecs = np.linalg.eig(s)
    top_n_i = np.argsort(evals)[::-1][:n_comp]
    n_evecs = evecs[:,top_n_i]
    X_train_proj = np.dot(X_train, n_evecs)
    X_test_proj = np.dot(X_test, n_evecs)
    
    centroids = {}
    for cls in classes:
        centroids[cls] = np.mean(X_train_proj[y_train == cls], axis=0)

    predictions = []
    for x in X_test_proj:
        distances = [np.linalg.norm(x - centroids[cls]) for cls in classes]
        predicted_class = classes[np.argmin(distances)]
        predictions.append(predicted_class)
    
    return predictions




if __name__ == "__main__":
    dset = load_iris()
    X = dset.data
    y = dset.target
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)

    predictions = lin_dis_a(X_train,y_train,X_test)
    print(f"own accuracy: {accuracy_score(y_test,predictions)}")

    ldr = LinearDiscriminantAnalysis(n_components=2)
    ldr.fit(X_train,y_train)
    predictions = ldr.predict(X_test)
    print(f"sklearn accuracy: {accuracy_score(y_test,predictions)}")
