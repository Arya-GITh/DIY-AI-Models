import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

data = load_diabetes()
X = data.data
Y = data.target
std_scaler = StandardScaler()
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)

#helpers
def euclidean(X_train,X_test):
    
    X_train_2 = np.sum(np.square(X_train),axis=1)
    X_test_2 = np.sum(np.square(X_test),axis=1)
    cross = np.dot(X_test,X_train.T)
    dist_2 = X_test_2[:,np.newaxis] + X_train_2 - 2*cross
    dist =  np.sqrt(np.maximum(dist_2,0))
    return dist

def manhattan(X_train,X_test):
    return np.sum(np.abs(X_test[:,np.newaxis,:]-X_train[np.newaxis,:,:]), axis = 2)

#KNN_regression
def KNN_regress(X_train,X_test,Y_train,k,distance = "euclidean"):
    if distance == "euclidean":
        dist = euclidean(X_train,X_test)
    else:
        dist = manhattan(X_train,X_test)
    indices = np.argsort(dist, axis=1)[:,:k]
    return Y_train[indices].mean(axis=1)

def r2_score_own(Y_pred, Y_test):
    ssr = np.sum(np.square(Y_pred-Y_test))
    mean = np.mean(Y_test)
    sst = np.sum(np.square(Y_test-mean))
    return 1 - (ssr/sst)

k_val = range(1,50)
pred_val = []
sk_val = []

for k in k_val:
    Y_pred = KNN_regress(X_train,X_test,Y_train,k)
    pred_val.append(r2_score_own(Y_pred,Y_test))
    sk_model = KNeighborsRegressor(k)
    sk_model.fit(X_train,Y_train)
    sk_pred = sk_model.predict(X_test)
    sk_val.append(r2_score(Y_test,sk_pred))

fig,ax = plt.subplots(1,2)
ax[0].plot(k_val, sk_val, label =  "scikit")
ax[0].plot(k_val, pred_val, label = "own", linestyle = "dotted")
ax[0].set_ylabel("r2_score")
ax[0].set_xlabel("k-value")
ax[0].set_title("Euclidean")
ax[0].legend()


pred_val = []
sk_val = []

for k in k_val:
    Y_pred = KNN_regress(X_train,X_test,Y_train,k,"manhattan")
    pred_val.append(r2_score_own(Y_pred,Y_test))
    sk_model = KNeighborsRegressor(k, metric='cityblock')
    sk_model.fit(X_train,Y_train)
    sk_pred = sk_model.predict(X_test)
    sk_val.append(r2_score(Y_test,sk_pred))

ax[1].plot(k_val, sk_val, label =  "scikit")
ax[1].plot(k_val, pred_val, label = "own", linestyle = "dotted")
ax[1].set_ylabel("r2_score")
ax[1].set_xlabel("k-value")
ax[1].set_title("Manhattan")
ax[1].legend()

fig.suptitle("Finding the optimal K")
plt.show()