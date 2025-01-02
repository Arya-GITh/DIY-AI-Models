import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1/(1+np.exp(-x))

def log_regress(X_train,y_train,X_test,lr_rate = 0.1, num_iter = 100,threshold = 0.5):
    m,n_feature = X_train.shape
    weights = np.zeros(n_feature)
    bias  = 0   
    
    for _ in range(num_iter):
        lin_model = np.matmul(X_train,weights) + bias
        y_pred_prob = sigmoid(lin_model)
        error = y_pred_prob - y_train
        dW = (1/m)*np.matmul(X_train.T,error)
        db = (1/m)*np.sum(error)
        weights -= lr_rate * dW
        bias  -= lr_rate * db

    lin_model_pred = np.dot(X_test, weights) + bias
    y_pred = sigmoid(lin_model_pred)
    return (y_pred>=threshold).astype(int)

def score(Y_predict,Y_test):
    score = np.mean(Y_predict == Y_test)
    return score

if __name__ == "__main__":
    dset = load_breast_cancer()
    X = dset.data
    y = dset.target
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=3)
    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)


    def sklearn_model(X_train,X_test,y_train):
        lg = LogisticRegression(penalty='l2')
        lg.fit(X_train,y_train)
        y_pred_prob_sci = lg.predict_proba(X_test)
        return y_pred_prob_sci
    
    y_pred_prob_sci = sklearn_model(X_train,X_test,y_train)

    threshold = np.linspace(0.45,0.6,10)
    y_pred_own = []
    y_pred_sci = []
    for t in threshold:
        y_pred = log_regress(X_train,y_train,X_test,threshold=t)
        y_pred_own.append(score(y_pred,y_test))

        y_pred_s = (y_pred_prob_sci[:,1] >= t).astype(int)
        y_pred_sci.append(score(y_pred_s,y_test))



    plt.xlabel("Threshold Values")
    plt.ylabel("Accuracy Scores")
    plt.plot(threshold,y_pred_own, label = "own")
    plt.plot(threshold,y_pred_sci, label = "sklearn")
    plt.legend()
    plt.show()

    