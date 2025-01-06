from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score

#own gaussian nb classifier
def gaussian_nb(X_train,X_test,y_train):
    
    cls = np.unique(y_train)
    n_classes = 3
    m,n_features = X_train.shape

    mean = np.zeros((m,n_features))
    var = np.zeros((m,n_features))
    prior = np.zeros((n_classes))

    for i,cls in enumerate(cls):
        X_cls = X_train[y_train==cls]
        mean[i] = np.mean(X_cls,axis=0)
        var[i] = np.var(X_cls,axis=0)
        prior[i] = len(X_cls)/m

    predictions = []
    for x in X_test:
        probs = []
        for i in range(n_classes):
            log_prior  = np.log(prior[i])
            log_likelihood = np.sum(-0.5*np.log(2*np.pi*var[i])-0.5*np.square(x-mean[i])/var[i])
            probs.append(log_prior + log_likelihood) 
        
        predictions.append(np.argmax(probs))

    return predictions
if __name__ == "__main__":

    dset = load_wine()
    X = dset.data
    y = dset.target
    X_train,X_test,y_train,y_test = train_test_split(X,y)

    predict_own = gaussian_nb(X_train,X_test,y_train)
    own_score = accuracy_score(y_test,predict_own)

    print(f"own accuracy: {own_score}")

    #sklearn gaussian nb classifier
    gnb_sci = GaussianNB()
    gnb_sci.fit(X_train,y_train)
    predict_sci = gnb_sci.predict(X_test)
    sci_score = accuracy_score(y_test,predict_sci)

    print(f"sklearn accuracy: {sci_score}")

