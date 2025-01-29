import numpy as np
from sklearn.datasets import fetch_covtype,fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor

#define a node in a tree and subsidiary functions

class Node:
    def __init__(self,threshold=None,left=None,right=None,value=None):
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def mse_own(y):
    return np.mean(np.square(y-np.mean(y)))

def gini(y):
    cls, count = np.unique(y,return_counts=True)
    probs = count/np.sum(count)
    return (1 - np.sum(probs**2))



#type can be either "reg" or "clasf"
def decision_tree (X_train,y_train,X_test,type):
    assert (type == "clasf" or type == "reg")

    if type == "reg":
        #use squared error 
    else:
        #use gini impurity for clasf
    

    #returns y pred
    pass

def main():
    reg_data = fetch_california_housing()
    clasf_data = fetch_covtype()
    X_reg = reg_data.data
    y_reg = reg_data.target
    X_c = clasf_data.data
    y_c = clasf_data.target
    X_train_r,X_test_r,y_train_r,y_test_r = train_test_split(X_reg,y_reg)
    X_train_c,X_test_c,y_train_c,y_test_c = train_test_split(X_c,y_c)

    y_hat_clasf_own = decision_tree(X_train_c,y_train_c,X_test_c,type = "clasf")
    y_hat_reg_own = decision_tree(X_train_r,y_train_r,X_test_r,type="reg")
    
    score_own_clasf = accuracy_score(y_test_c,y_hat_clasf_own)
    score_own_reg = mean_squared_error(y_test_r,y_hat_reg_own)

    clasf_sci = DecisionTreeClassifier()
    reg_sci = DecisionTreeRegressor()

    clasf_sci.fit(X_train_c,y_train_c)
    reg_sci.fit(X_train_r,y_train_r)

    score_sci_clasf = clasf_sci.score(X_test_c,y_test_c)
    score_sci_reg = reg_sci.score(X_test_r,y_test_c)

    print("Reg own score:" + score_own_reg + "Sklearn score:" + score_sci_reg)
    print("Clasf own score:" + score_own_clasf + "Sklearn score:" + score_sci_clasf)

if __name__ == "__main__":
    main()

