import numpy as np
from sklearn.datasets import load_diabetes 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA   
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class PCA_own():
     # find the first n-principal components (if n == None, find all components)
    
    def __init__(self,n=None):
            self.n = n

    def fit_transform(self,X_train):
        
        if self.n == None:
             self.n = X_train.shape[1]

        self.mean = np.mean(X_train,axis=0)
        X_train_m = X_train - self.mean
        cov = np.cov(X_train_m,rowvar=0)
        eigvals,eigvecs = np.linalg.eig(cov)
        sorted_indices = np.argsort(eigvals)[::-1][:self.n]
        sorted_eigvecs = eigvecs[sorted_indices]
        self.eigvecs = sorted_eigvecs
        X_train_transform = np.dot(X_train_m,sorted_eigvecs)

        return X_train_transform

    def transform(self,X_test):
        
        X_test_m = X_test - self.mean
        X_test_transform = np.dot(X_test_m,self.eigvecs)
        return X_test_transform

if __name__ == "__main__":
    
    fig,ax = plt.subplots(1,2, figsize=(10,7))
    dset = load_diabetes()
    X,y = (dset.data,dset.target)
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=4)
    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)
    #accuracy score without PCA
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    acc_score_raw = mean_squared_error(y_test,lr.predict(X_test))
    print("LR mse without PCA:" + str(acc_score_raw))

    pca_sci = PCA(n_components=10)
    X_train_sci = pca_sci.fit_transform(X_train)
    X_test_sci  = pca_sci.transform(X_test)
    pca_own = PCA_own(n=10)
    X_train_own = pca_own.fit_transform(X_train)
    X_test_own = pca_own.transform(X_test)

    mse_sci = []
    mse_own = []

    lr_sci = LinearRegression()
    lr_own = LinearRegression()
    n_comp = 10
    #mse score using n-components
    for i in range(1,n_comp+1):

        lr_sci.fit(X_train_sci[:,:i],y_train) 
        pred_sci = lr_sci.predict(X_test_sci[:,:i])
        mse_sci.append(mean_squared_error(y_test,pred_sci)) 

        lr_own.fit(X_train_own[:,:i],y_train) 
        pred_own = lr_own.predict(X_test_own[:,:i])
        mse_own.append(mean_squared_error(y_test,pred_own))
    
    
    ax[0].plot(range(1,n_comp+1),mse_sci, label = "MSE_Sklearn")
    ax[0].axhline(y=acc_score_raw, label = "MSE_noPCA", c = "red")
    ax[1].plot(range(1,n_comp+1),mse_own, label = "MSE_Own")
    ax[1].axhline(y=acc_score_raw, label = "MSE_noPCA", c="red")
    ax[0].set_xlabel("Number of Principal Components")
    ax[0].set_ylabel("Mean Squared Error")
    ax[1].set_xlabel("Number of Principal Components")
    ax[1].set_ylabel("Mean Squared Error")
    fig.suptitle("PCA")
    ax[0].legend()
    ax[1].legend()
    plt.show()