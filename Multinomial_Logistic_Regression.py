import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)  
    return exp_x / sum_exp_x

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

def accuracy(y_pred,y_test):
    return np.mean(y_pred == y_test)
def multlog_regress(X_train,X_test,y_train, lr_rate = 0.01, num_iter = 100):
    m,num_features = X_train.shape
    num_classes  = 3
    W = np.zeros((num_features,num_classes))
    b = np.zeros((1,num_classes))
    for _ in range(num_iter):
        y_prob = softmax(np.dot(X_train,W) + b )
        error = y_prob - y_train
        db = (1/m)* np.sum(error,axis=0)
        dW = (1/m) * np.dot(X_train.T,error)
        W -= lr_rate * dW
        b -= lr_rate * db

    y_hat_prob = softmax(np.dot(X_test,W) + b)
    y_hat = np.argmax(y_hat_prob, axis=1)
    return y_hat




if __name__ == "__main__":
    dset = load_wine()
    X = dset.data
    y = dset.target

    y_onehot = one_hot_encode(y,3)
    X_train,X_test,y_train,y_test = train_test_split(X,y_onehot, test_size=0.2,stratify=y,random_state=42)
    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)
    y_pred = multlog_regress(X_train,X_test,y_train)
    y_test = np.argmax(y_test,axis =1)
    print( "own accuracy :" + str(accuracy(y_pred,y_test)))

    lg = LogisticRegression(max_iter = 100)
    y_train = np.argmax(y_train,axis=1)
    lg.fit(X_train,y_train)
    y_hat_sklearn = lg.predict(X_test)
    print("sklearn accuracy:" + str(accuracy(y_hat_sklearn,y_test)))
