import numpy as np
import pandas as pd 
import datetime 
import pandas_datareader.data as web 
from pandas import Series, DataFrame 
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU,LSTM,SimpleRNN

from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import time
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import csv
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from pandas import concat

#%%
 
start = datetime.datetime(2010, 1, 1) 
end = datetime.datetime(2017, 1, 11) 
 
df = web.DataReader("AAPL", 'yahoo', start, end) 
df.tail() 
start = datetime.datetime(2010, 1, 1) 
end = datetime.datetime(2017, 1, 11) 

df1 = web.DataReader("GOOG", 'yahoo', start, end) 
df1.tail()

df2=np.concatenate((df,df1),axis=0)

data=np.zeros((3538,5))
apple=np.zeros((1769,5))
google=np.zeros((1769,5))
close=np.zeros((3538,1))

j=0
for i in range(1769):
    data[j,0:3]=df2[i,0:3]
    data[j+1,0:3]=df2[i+1769,0:3]
    data[j,3:5]=df2[i,4:6]
    data[j+1,3:5]=df2[i+1769,4:6]
    close[j,0]=df2[i,3]
    close[j+1,0]=df2[i+1769,3]
    apple[i,0:3]=df2[i,0:3]
    apple[i,3:5]=df2[i,4:6]
    google[i,0:3]=df2[i+1769,0:3]
    google[i,3:5]=df2[i+1769,4:6]
    j=j+2

def series_to_supervised(data, n_in, n_out, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        
    agg = concat(cols, axis=1)
    
    if dropnan:
        agg.dropna(inplace=True)
    
    return agg
apple = series_to_supervised(apple, 30, 1)
apple = apple.values
google = series_to_supervised(google, 30, 1)
google = google.values

y_dataset = close[60:,0]
x_dataset=np.zeros((3478,155))

j=0
for i in range(1739):
    x_dataset[j,:]=apple[i,:]
    x_dataset[j+1,:]=google[i,:]
    j=j+2
x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.20, random_state=42)
y_train=y_train.reshape(2782,1)
y_test=y_test.reshape(696,1)
scaler1 = MinMaxScaler(feature_range=(0, 1))
x_train = scaler1.fit_transform(x_train)
scaler2 = MinMaxScaler(feature_range=(0, 1))
x_test = scaler2.fit_transform(x_test)
scaler3 = MinMaxScaler(feature_range=(0, 1))
y_train = scaler3.fit_transform(y_train)
scaler4 = MinMaxScaler(feature_range=(0, 1))
y_test = scaler4.fit_transform(y_test)

x_train = x_train.reshape((1391, 2, 155))
x_test = x_test.reshape((348, 2, 155))
y_train = y_train.reshape((1391, 2))
y_test = y_test.reshape((348, 2))

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape, "y_test shape:", y_test.shape)

start=datetime.datetime.now()




######################################################LSTM
#model = keras.models.Sequential()
#model.add(keras.layers.LSTM(10, input_shape=(2,155)))
#model.add(keras.layers.Dense(16, activation='tanh'))
#model.add(keras.layers.Dense(2, activation='tanh'))
#####################################################

###############################simple RNN & GRU
model = Sequential()
model.add(SimpleRNN(160, input_shape=(2, x_train.shape[2]), 
                    dropout = 0, recurrent_dropout=10))
model.add(Dense(8))
model.add(Dense(2))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=20, batch_size=16, 
                    validation_data=(x_test, y_test), verbose=2, shuffle=False)

#stopping the timer:
end=datetime.datetime.now()
Total_time_training=end-start

########################
print ('\nTotal training time of RNN with dropout:',Total_time_training )
#######################

fig, axs = plt.subplots(1,2,figsize=(20,8))
axs[0].plot(history.history['loss'], label='train')
axs[0].plot(history.history['val_loss'], label='test')

#########################################
axs[0].set_title('Model Loss for RNN with dropout')
########################################

axs[0].set_ylabel('Loss')
axs[0].set_xlabel('Epoch')
axs[0].legend()

axs[1].plot(history.history['accuracy'], label='train')
axs[1].plot(history.history['val_accuracy'], label='test')

#########################################
axs[1].set_title('Model Accuracy for RNN with dropout')
#########################################

axs[1].set_ylabel('Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].legend()

#plt.show()
########################################
plt.savefig('Model Loss for RNN with dropout')
########################################

yhat = model.predict(x_test) 
inv_yhat = scaler4.inverse_transform(yhat)

#print("y_hat shape:", inv_yhat.shape)

inv_y = scaler4.inverse_transform(y_test)

#print("y shape:", inv_y.shape)

rmse0 = sqrt(mean_squared_error(inv_y[:,0], inv_yhat[:,0]))
rmse1 = sqrt(mean_squared_error(inv_y[:,1], inv_yhat[:,1]))
rmse = (rmse0 + rmse1)/2

##############################
print('\nTest MSE for RNN with dropout: %.3f' % rmse)
################################

fig, axs = plt.subplots(1,2, figsize=(20,8))

#############################################
axs[0].set_title(' APPLE prediction with RNN with dropout')
#############################################

axs[0].plot(inv_y[:,0], label='True Data', color='yellow', linewidth='3')
axs[0].plot(inv_yhat[:,0], label='Prediction', color='green', linewidth='2')
axs[0].set_xlabel('test hour', color='red')
axs[0].legend()

####################################
plt.savefig('predictions of RNN with dropout')
####################################
#plt.show()

###################################
axs[1].set_title(' GOOGLE prediction with RNN with dropout')
#####################################

axs[1].plot(inv_y[:,1], label='True Data', color='yellow', linewidth='3')
axs[1].plot(inv_yhat[:,1], label='Prediction', color='green', linewidth='2')
axs[1].set_xlabel('test hour', color='red')
axs[1].legend()
#plt.show()

#####################
plt.savefig('predictions of RNN with dropout')
########################