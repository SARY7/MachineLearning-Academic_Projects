
#%% I M P O R T I N G----L I B R A R I E S-------------------------------------
'Load Sequence:1'

from keras.datasets import mnist
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics


#%% I M P O R T I N G----D A T A & P R O C E S S I N G-------------------------
'Load Sequence:2'

#------------------------------------------------------------------------------
(X,Y),(_,_) = mnist.load_data()
X = X.reshape(X.shape[0], 784)
X = np.asarray( X, 'float32')
# data normalization
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  
# Converting to images
X = X > 0.5

#%% T R A I N----THE----M O D E L----------------------------------------------
'Load Sequence:3'

#------------------------------------------------------------------------------
# Implementing RBM
#------------------------------------------------------------------------------
rbm = BernoulliRBM(n_components=200, learning_rate=0.01, batch_size=10, n_iter=40, verbose=True, random_state=None)
logistic = LogisticRegression(C=10)
clf = Pipeline(steps=[('rbm', rbm), ('clf', logistic)])
Xtrain, Xtest, Ytrain, Ytest = train_test_split( X, Y, test_size=0.2, random_state=0)
clf.fit(Xtrain, Ytrain)
Ypred = clf.predict(Xtest)
print ('Validation score:  ',(metrics.classification_report(Ytest, Ypred)))
