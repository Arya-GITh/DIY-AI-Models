import numpy as np
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from Decision_Trees import decision_tree
from scipy.stats import mode

def boosting_own(X_train, y_train, X_test, n_trees=10, learning_rate=0.1, regress=True):
    if regress:
        predictions = np.full(X_test.shape[0], np.mean(y_train))
        residuals = y_train - np.mean(y_train)
        
        for _ in range(n_trees):
            model = decision_tree(X_train, residuals, X_train, type="reg")
            residuals -= learning_rate * np.array(model)  # Update residuals
            update = decision_tree(X_train, residuals, X_test, type="reg")
            predictions += learning_rate * np.array(update)  # Accumulate predictions
        
        return predictions
    else:
        predictions = np.zeros((X_test.shape[0], np.max(y_train) + 1))  # Store class probabilities
        
        for _ in range(n_trees):
            model = decision_tree(X_train, y_train, X_train, type="clasf")
            update = decision_tree(X_train, y_train, X_test, type="clasf")
            for i, pred in enumerate(update):
                predictions[i, pred] += 1  # Count votes
        
        return np.argmax(predictions, axis=1)  # Return class with the most votes

def main():
    reg_data = fetch_california_housing()
    clasf_data = load_iris()
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(reg_data.data, reg_data.target)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(clasf_data.data, clasf_data.target)
    
    # Boosting for regression
    y_pred_r_boost = boosting_own(X_train_r, y_train_r, X_test_r, regress=True)
    score_r_boost = mean_squared_error(y_test_r, y_pred_r_boost)
    print("Boosting Regression MSE:", score_r_boost)
    
    # Compare with Sklearn Gradient Boosting Regressor
    reg_sklearn = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1)
    reg_sklearn.fit(X_train_r, y_train_r)
    score_r_sklearn = mean_squared_error(y_test_r, reg_sklearn.predict(X_test_r))
    print("Sklearn Boosting Regression MSE:", score_r_sklearn)
    
    # Boosting for classification
    y_pred_c_boost = boosting_own(X_train_c, y_train_c, X_test_c, regress=False)
    score_c_boost = accuracy_score(y_test_c, y_pred_c_boost)
    print("Boosting Classification Accuracy:", score_c_boost)
    
    # Compare with Sklearn Gradient Boosting Classifier
    clasf_sklearn = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1)
    clasf_sklearn.fit(X_train_c, y_train_c)
    score_c_sklearn = accuracy_score(y_test_c, clasf_sklearn.predict(X_test_c))
    print("Sklearn Boosting Classification Accuracy:", score_c_sklearn)

if __name__ == "__main__":
    main()
