from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from itertools import combinations_with_replacement
from Multiple_Linear_Regression import mult_lin_reg,adjusted_r2
import numpy as np
from math import comb

def pol_features(X,degree):
    n_samples, n_features = X.shape
    p_array = [X]
    for deg in range(2,degree+1):
        for combs in combinations_with_replacement(range(n_features),deg):
            p_array.append((np.prod(X[:,combs],axis=1)).reshape(n_samples,1))
    
    return np.hstack(p_array)

def mult_pol_regress(X_train,X_test,y_train,degree):
    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)  
    X_train = pol_features(X_train,degree)
    X_test = pol_features(X_test,degree)
    y_pred = mult_lin_reg(X_train,X_test,y_train)
    return y_pred

if __name__ == "__main__":

    #creating dummy data
    n_features = 2
    n_samples = 100
    n_model_degree = 2
    np.random.seed(45)
    x1 = np.random.rand(n_samples)
    x2 = np.random.rand(n_samples)
    noise = np.random.randn(n_samples) * 0.1
    y = 5 + 4 * x1 - 1.6 * x2 + 0.95 * (x1 ** 2) + 0.67 * (x2 ** 2) + 1.02 * (x1 * x2) + noise
    X = np.column_stack((x1, x2))

    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)
    y_pred = mult_pol_regress(X_train,X_test,y_train,2)
    
    deg = range(1,5)
    y_hat_list = []

    for d in deg:
        val = adjusted_r2(mult_pol_regress(X_train,X_test,y_train,d),y_test,comb(n_features+d,d))
        y_hat_list.append(val)
    
    plt.plot(deg,y_hat_list)
    plt.title("Degree of fit vs Adjusted R2")
    plt.ylabel("Adjusted R2")
    plt.xlabel("Degree")
    plt.show()