
#%% I M P O R T I N G----L I B R A R I E S-------------------------------------
'Load Sequence:1'

#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plot
from keras import metrics


#%% I M P O R T I N G----D A T A & P R O C E S S I N G-------------------------
'Load Sequence:2'

#------------------------------------------------------------------------------
df = pd.read_csv('house_data.csv')
dataset = df.values
dataset.shape
X = dataset[: , 0:13]
Y = dataset[: , 13]
TransformY = preprocessing.MinMaxScaler()
Transform = preprocessing.MinMaxScaler()
Xscale = Transform.fit_transform(X)
Yscale = TransformY.fit_transform(Y.reshape(Y.shape[0],1))
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xscale, Yscale, test_size=0.2)


#%% P C A----------------------------------------------------------------------
'Load Sequence:3'

#------------------------------------------------------------------------------
EncodingDimension = 10

from sklearn.decomposition import PCA
PCA = PCA(n_components = EncodingDimension)
XtrainPCA = PCA.fit_transform(Xtrain)
XtestPCA = PCA.transform(Xtest)
explained_var = PCA.explained_variance_ratio_
np.sum(explained_var[0:256])


#%% D E S I N I N G----THE----M O D E L----------------------------------------
'Load Sequence:4'

#------------------------------------------------------------------------------
MLP = Sequential()
MLP.add(Dense(13,activation = 'relu',kernel_initializer='normal',input_shape = (10,)))
MLP.add(Dense(7,activation = 'relu',kernel_initializer='normal'))
MLP.add(Dense(1,kernel_initializer='normal'))
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
MLP.compile(loss='mean_squared_error', optimizer='adam',  metrics=[metrics.mae])
MLP.summary()


#%% T R A I N----THE----M O D E L----------------------------------------------
'Load Sequence:5'

#------------------------------------------------------------------------------
TrainMLP = MLP.fit(XtrainPCA,Ytrain , batch_size=4 , epochs=40)
Ypredict = MLP.predict(XtestPCA)
Test = np.hstack((Ypredict,Ytest))
history = TrainMLP.history
MeanSquareError=history['loss']
#------------------------------------------------------------------------------
#plotting
plot.xlabel('Epochs')
plot.ylabel('Mean Square Error')
plot.plot(MeanSquareError)
plot.legend(['Mean Square Error'])
plot.title('Mean Square Error for each epoch for a PCA network')
plot.savefig('Mean Square Error for each epoch for a PCA network')

