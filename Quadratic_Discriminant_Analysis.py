
"""
Using QDA as a  simple classifier
"""
import math
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
import numpy as np


def quad_dis_a(X_train,y_train,X_test,ax):
    n_feature = X_train.shape[1]
    classes = np.unique(y_train)
    n_class = 3
    n_comp = n_class - 1

    
    cls_means = {}
    cls_covs = {}
    cls_priors = {}
    cls_inv_covs = {}
    cls_det_covs = {}

    for cls in classes:
        X_cls = X_train[y_train==cls]
        cls_means[cls] = np.mean(X_cls,axis=0)
        cls_covs[cls] = np.cov(X_cls, rowvar=False)
        cls_priors[cls] = len(X_cls)/len(X_train)
        cls_inv_covs[cls] = np.linalg.inv(cls_covs[cls])
        cls_det_covs[cls] = np.linalg.det(cls_covs[cls])

    predictions = []

    for x in X_test:
        max_d = -(math.inf)
        max_c = None
        for cls in classes:
            term1 = -0.5 * np.log(cls_det_covs[cls])
            diff = x - cls_means[cls]
            term2 = -0.5 * np.dot(np.dot(diff.T,  cls_inv_covs[cls]), diff)
            term3 = np.log(cls_priors[cls])
            d = term1 + term2 + term3
            if d > max_d:
                max_d = d
                max_c = cls
        predictions.append(max_c)
    #plot
    for cls in classes:
        ax.scatter(X_train[y_train == cls, 0], 
                   X_train[y_train == cls, 1], 
                   label=f"Train Class {cls}", alpha = 0.6)
        ax.scatter(X_test[np.array(predictions) == cls, 0], X_test[np.array(predictions) == cls, 1], 
        label=f"Predicted Test Class {cls}", marker='D', edgecolor='k')
        ax.scatter(cls_means[cls][0], cls_means[cls][1], marker='X', s=200, label=f"Class Measn {cls}", edgecolor='k')
    ax.set_title('QDA Outputs')
    ax.set_xlabel('dim1')
    ax.set_ylabel('dim2')
    ax.legend()
  
    
    return predictions




if __name__ == "__main__":
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    dset = load_iris()
    X = dset.data
    y = dset.target
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)

    predictions = quad_dis_a(X_train,y_train,X_test,axes)
    print(f"own accuracy: {accuracy_score(y_test,predictions)}")

    qdr = QuadraticDiscriminantAnalysis()
    qdr.fit(X_train,y_train)
    predictions = qdr.predict(X_test)
    print(f"sklearn accuracy: {accuracy_score(y_test,predictions)}")
    plt.show()