from sklearn.naive_bayes import BernoulliNB
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

#own bernoulli nb classifier
def bernoulli_nb(X_train,X_test,y_train, alpha = 1):
    #code

    classes = [0,1]
    n_class = 2
    m,n_features = X_train.shape

    log_priors = np.log(np.array([np.sum(y_train==c) for c in classes])/ m)

    probs = []
    for i in range(n_class):
        X_cls = X_train[y_train==i]
        count = len(X_cls)
        prob = (X_cls.sum(axis=0) + alpha) / (count + 2 * alpha)

        probs.append(prob)
    
    predictions = []
    for x in X_test:
        cls_probs = np.zeros((2,))
        for c in classes:
            score = log_priors[c] + np.sum(x * np.log(probs[c]) + (1-x)* np.log(1 - (probs[c])))
            cls_probs[c] = score 
        predictions.append(np.argmax(cls_probs))

    return predictions

if __name__ == "__main__":
    np.random.seed(1)
   #generate random synthetic data binary data ~ Chat GPT
    n_samples = 1000
    n_features = 20
    y = np.random.choice([0, 1], size=n_samples)

    # Initialize feature matrix
    X = np.zeros((n_samples, n_features))

    # Define feature probabilities for each class
    # Class 0: P(x_i=1) = 0.3
    # Class 1: P(x_i=1) = 0.7
    p_class = {0: 0.3, 1: 0.7}
    
    for c in [0, 1]:
        X[y == c] = np.random.binomial(1, p_class[c], size=(np.sum(y == c), n_features))

    X_train,X_test,y_train,y_test = train_test_split(X,y, stratify=y, test_size=0.3, random_state=30)

    predict_own = bernoulli_nb(X_train,X_test,y_train)
    own_score = accuracy_score(y_test,predict_own)

    print(f"own accuracy: {own_score}")

    #sklearn bernoulli nb classifier
    bnb_sci = BernoulliNB()
    bnb_sci.fit(X_train,y_train)
    predict_sci = bnb_sci.predict(X_test)
    sci_score = accuracy_score(y_test,predict_sci)

    print(f"sklearn accuracy: {sci_score}")

