import numpy as np
from sklearn.datasets import load_diabetes 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import random

class PLS_own():
     # find the first n-principal components (if n == None, find all components)
    
    def __init__(self,n=None):
            self.n = n

    def fit_transform(self,X_train,y_train):
        if self.n == None:
            n = X_train.shape[1]
        

        self.x_mean = np.mean(X_train,axis=0)
        self.y_mean = np.mean(y_train)
        X_c = X_train - self.x_mean
        Y_c = y_train - self.y_mean

        m,n_feature = X_train.shape
        q = y_train.shape[0]

        #weights distinct from loadings in pls
        self.x_weights = np.zeros((n_feature,self.n))
        self.y_weights = np.zeros((q,self.n))
        self.x_score = np.zeros((m,self.n))
        self.y_score = np.zeros((m,self.n))

        for i in range(self.n):
            
            u0 = Y_c.copy()

            while True:
                 
                w = np.matmul(u0,X_c)
                w /= np.linalg.norm(w)
                t = np.matmul(X_c,w)
                c = np.matmul(t,Y_c)
                c /= np.linalg.norm(c)
                u1 = Y_c * c

                if np.linalg.norm(u0-u1) < 1e-12:
                    break
                u0 = u1
            
            p = np.dot(t,X_c)/np.dot(t,t)
            q = np.dot(t,Y_c)/np.dot(t,t)

            self.x_weights[:,i] = w
            self.x_score[:,i] = t 
            X_c -= np.outer(t,p)
            Y_c -= t*q

        return self.x_score
        

    def transform(self,X_test):
        
        X_test_c = X_test - self.x_mean
        X_test_transform = X_test_c @ self.x_weights
        return X_test_transform

if __name__ == "__main__":
    
    fig,ax = plt.subplots(1,2, figsize=(10,7))
    dset = load_diabetes()
    X,y = (dset.data,dset.target)
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=4)
    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)

    #accuracy score without PLS
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    acc_score_raw = mean_squared_error(y_test,lr.predict(X_test))
    print("LR mse without PLS:" + str(acc_score_raw))


    pls_own = PLS_own(n=10)
    X_train_own = pls_own.fit_transform(X_train,y_train)
    X_test_own = pls_own.transform(X_test)

    mse_sci = []
    mse_own = []

    lr_own = LinearRegression()
    n_comp = 10
    #mse score using n-components
    for i in range(1,n_comp+1):

        pls = PLSRegression(n_components=i)
        pls.fit(X_train,y_train) 
        pred_sci = pls.predict(X_test)#[:,:i]
        mse_sci.append(mean_squared_error(y_test,pred_sci)) 

        lr_own.fit(X_train_own[:,:i],y_train) 
        pred_own = lr_own.predict(X_test_own[:,:i])
        mse_own.append(mean_squared_error(y_test,pred_own))
    
    
    ax[0].plot(range(1,n_comp+1),mse_sci, label = "MSE_Sklearn")
    ax[0].axhline(y=acc_score_raw, label = "MSE_noPLS", c = "red")
    ax[1].plot(range(1,n_comp+1),mse_own, label = "MSE_Own")
    ax[1].axhline(y=acc_score_raw, label = "MSE_noPLS", c="red")
    ax[0].set_xlabel("Number of PLS Components")
    ax[0].set_ylabel("Mean Squared Error")
    ax[1].set_xlabel("Number of PLS Components")
    ax[1].set_ylabel("Mean Squared Error")
    fig.suptitle("PLS")
    ax[0].legend()
    ax[1].legend()
    plt.show()