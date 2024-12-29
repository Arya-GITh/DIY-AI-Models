from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from Simple_Linear_Regression import r2_score_own
import numpy as np


def mult_lin_reg(X_train,X_test,y_train):
    ones_train = np.ones((X_train.shape[0],1))
    X_train = np.hstack([ones_train,X_train])
    beta =  np.linalg.pinv(X_train) @ y_train
    ones_test = np.ones((X_test.shape[0],1))
    X_test = np.hstack([ones_test,X_test])
    y_hat =  X_test @ beta
    return y_hat

def adjusted_r2(y_pred,y_test,n_features):
    r2 = r2_score_own(y_pred,y_test)
    n = len(y_pred)
    return 1 - ((1-r2)*(n-1)/(n-n_features-1))


if __name__ == "__main__":
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    std_scl = StandardScaler()
    X_train = std_scl.fit_transform(X_train)
    X_test = std_scl.transform(X_test)


    y_pred_own = mult_lin_reg(X_train,X_test,y_train)

    lr = LinearRegression()
    lr.fit(X_train,y_train)
    y_pred_sci = lr.predict(X_test)

    print("sklearn score:" + str(adjusted_r2(y_pred_sci,y_test,X_train.shape[1])))
    print("own score:" + str(adjusted_r2(y_pred_own,y_test,X_train.shape[1])))