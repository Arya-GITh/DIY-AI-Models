import numpy as np
from sklearn.datasets import make_regression
X,y = make_regression(n_samples = 200, n_features=1, noise=50)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2)
X_train_ravel = X_train.ravel()
X_test_ravel = X_test.ravel()

def linear_regressor(X_train,y_train,X_test):
    X_mean = np.mean(X_train)
    y_mean = np.mean(y_train)
    sum_cross_dev = np.sum(np.multiply((X_train-X_mean),(y_train-y_mean)))
    SSx = np.sum(np.square(X_train-X_mean))
    b1 = sum_cross_dev/SSx
    b0 = y_mean - b1*X_mean
    Y_hat = []
    for x in X_test:
        Y_hat.append(b1*x+b0)
    
    return np.array(Y_hat)

Y_pred_own = linear_regressor(X_train_ravel,Y_train,X_test_ravel)


lin_reg = LinearRegression()
lin_reg.fit(X_train,Y_train)
Y_pred_sci = lin_reg.predict(X_test)

fig,ax = plt.subplots(1,2)

ax[0].scatter(X_test,Y_test)
ax[1].scatter(X_test,Y_test)
ax[0].plot(X_test,Y_pred_own, c="blue", label = "Own")
ax[0].legend()
ax[1].plot(X_test,Y_pred_sci,label = "ScikitLearn", c="red")
ax[1].legend()
fig.suptitle("Simple Linear Regresison")


def r2_score_own(Y_pred, Y_test):
    ssr = np.sum(np.square(Y_pred-Y_test))
    mean = np.mean(Y_test)
    sst = np.sum(np.square(Y_test-mean))
    return 1 - (ssr/sst)

sci_r2 = r2_score_own(Y_pred_sci,Y_test)
own_r2 = r2_score_own(Y_pred_own,Y_test)

print(f" own r2: {own_r2} and sklearn r2 {sci_r2}")
plt.show()