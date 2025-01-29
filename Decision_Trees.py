import numpy as np
from sklearn.datasets import fetch_covtype,fetch_california_housing,load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from scipy.stats import mode

#define a node in a tree and subsidiary functions

class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,value=None):
        self.feature = feature
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

def split(X_train,y_train,criteria):
    m,n_feat = X_train.shape
    best_feature,best_threshold,best_score,best_groups = None,None,float("inf"), None
    for feature in range(n_feat):
        thresholds = np.unique(X_train[:,feature]) #consider using np.percentile
        for threshold in thresholds:
            left_bool = X_train[:,feature] <= threshold
            right_bool = np.logical_not(left_bool)
            if np.sum(left_bool) == 0 or np.sum(right_bool) == 0:
                continue

            left_score = criteria(y_train[left_bool])
            right_score = criteria(y_train[right_bool])
            agg_score = (np.sum(left_bool)*left_score + np.sum(right_bool)*right_score)/m
            if agg_score < best_score:
                best_feature,best_threshold,best_score,best_groups = feature,threshold,agg_score,(left_bool,right_bool)
    return best_feature,best_threshold,best_groups



def build(X_train,y_train,criteria, min_samp_split = 2, max_depth = 10, depth=0):
    #build tree
    if len(np.unique(y_train)) <= 1 or len(y_train) < min_samp_split or depth == max_depth:
        return Node(value = np.mean(y_train) if criteria == mse_own else np.bincount(y_train).argmax())
    feature,threshold,(left_bool,right_bool) = split(X_train,y_train,criteria)

    if feature == None:
        return Node(value = np.mean(y_train) if criteria == mse_own else np.bincount(y_train).argmax())

    left = build(X_train[left_bool],y_train[left_bool],criteria, depth = depth+1)
    right = build(X_train[right_bool],y_train[right_bool],criteria, depth = depth+1)

    return Node(feature=feature,threshold=threshold,left=left,right=right)

def predict(X_test,tree):
    predictions = []
    for x in X_test:
        node = tree
        while node.value == None:
            if x[node.feature] <= node.threshold:
                 node = node.left
            else:
                node = node.right
        predictions.append(node.value)
    return predictions

#function to call: type can be either "reg" or "clasf" 
def decision_tree (X_train,y_train,X_test,type):
    assert (type == "clasf" or type == "reg")

    criteria = mse_own if type == "reg" else gini 

    #returns y pred
    tree = build(X_train,y_train,criteria)
    return predict(X_test,tree)

def main():
    reg_data = fetch_california_housing()
    clasf_data = load_iris()
    X_reg = reg_data.data
    y_reg = reg_data.target
    X_c = clasf_data.data
    y_c = clasf_data.target
    X_train_r,X_test_r,y_train_r,y_test_r = train_test_split(X_reg,y_reg)
    X_train_c,X_test_c,y_train_c,y_test_c = train_test_split(X_c,y_c)

    #run classification
    y_hat_clasf_own = decision_tree(X_train_c,y_train_c,X_test_c,type = "clasf")
    score_own_clasf = accuracy_score(y_test_c,y_hat_clasf_own)
    clasf_sci = DecisionTreeClassifier()
    clasf_sci.fit(X_train_c,y_train_c)
    score_sci_clasf = clasf_sci.score(X_test_c,y_test_c)
    print("Clasf own score:" + str(score_own_clasf) + ". Sklearn score:" + str(score_sci_clasf))

    #run regression
    #y_hat_reg_own = decision_tree(X_train_r,y_train_r,X_test_r,type="reg")
    #score_own_reg = mean_squared_error(y_test_r,y_hat_reg_own)
    #reg_sci = DecisionTreeRegressor(max_depth=None)
    #reg_sci.fit(X_train_r,y_train_r)
    #score_sci_reg = reg_sci.score(X_test_r,y_test_r)
    #print("Reg own score:" + str(score_own_reg) + ". Sklearn score:" + str(score_sci_reg))


if __name__ == "__main__":
    main()

