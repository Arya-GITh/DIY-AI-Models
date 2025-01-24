import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from scipy.interpolate import CubicSpline

def penalty():
    #should define a penalty of a second degree since its a cubic spline

def spline( X_train,y_train,X_test):
   
    spline = CubicSpline(X_train,y_train)

    y_pred = spline(X_test)

    return y_pred

def smoothing_spline(X_train,y_train):

    
if __name__ == "__main__":

    #creating random data
    n_samples = 150
    np.random.seed(45)
    X = np.random.uniform(-5,5,n_samples)
    coeff = np.random.normal(0,10,6)
    noise = np.random.normal(0,1,n_samples) * 0.01
    y = np.poly1d(coeff) + noise
    X_train,X_test,y_train,y_test = train_test_split(X,y)

    #sub-plot 1 of X versus y_pred

   