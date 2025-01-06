from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

#own multinomial nb classifier
def multinomial_nb(X_train,X_test,y_train, alpha = 1):
    
    cls = np.unique(y_train)
    n_classes = len(cls)
    m,n_features = X_train.shape

    priors = np.array([np.sum([y_train==c]) for c in cls])/m

    log_priors = np.log(priors)

    feature_count = np.zeros((n_classes,n_features))

    for i in range(n_classes):
        Xc = X_train[y_train == i]
        feature_count[i] = Xc.sum(axis=0)
    
    total_counts = feature_count.sum(axis=1)

    log_feature_prob = np.log((feature_count+ alpha)/ (total_counts[:,np.newaxis] + alpha * n_features) )

    predictions = []

    for x in X_test:
        log_prob = log_priors + np.dot(log_feature_prob,x)
        id_max = np.argmax(log_prob)
        predictions.append(id_max)
    
    return predictions
if __name__ == "__main__":

    ###Chat-GPT generated synthetic data
    np.random.set_state = 1
    X = []
    y = []
    n_features = 10
    n_classes = 3
    samples_per_class  = 20
    # Define distinct lambda (rate) parameters for each class to create separable distributions
    # Higher lambda means higher expected counts for that feature
    lambda_params = {
    0: np.random.uniform(1, 5, n_features),  # Class 0
    1: np.random.uniform(3, 7, n_features),  # Class 1
    2: np.random.uniform(5, 10, n_features)  # Class 2
    }

    # Generate data for each class
    for cls in range(n_classes):
    # Generate Poisson-distributed features for the class
        class_features = np.random.poisson(lam=lambda_params[cls], size=(samples_per_class, n_features))
        X.append(class_features)
        y.extend([cls] * samples_per_class)

    # Combine data from all classes
    X = np.vstack(X)
    y = np.array(y)

    X_train,X_test,y_train,y_test = train_test_split(X,y, stratify=y, test_size=0.3, random_state=30)

    predict_own = multinomial_nb(X_train,X_test,y_train)
    own_score = accuracy_score(y_test,predict_own)

    print(f"own accuracy: {own_score}")

    #sklearn multinomial nb classifier
    mnb_sci = MultinomialNB()
    mnb_sci.fit(X_train,y_train)
    predict_sci = mnb_sci.predict(X_test)
    sci_score = accuracy_score(y_test,predict_sci)

    print(f"sklearn accuracy: {sci_score}")

