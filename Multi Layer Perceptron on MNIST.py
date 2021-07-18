
#%% I M P O R T I N G----L I B R A R I E S-------------------------------------
'Load Sequence:1'

from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense
from keras import optimizers
import matplotlib.pyplot as plot


#%% I M P O R T I N G----D A T A & P R O C E S S I N G-------------------------
'Load Sequence:2'

#------------------------------------------------------------------------------
(Xtrain,Ytrain),(Xtest,Ytest)=mnist.load_data()
#plt.imshow(Xtrain[1])
# We have 60000 data for train with the dimension of 28*28
Xtr = Xtrain.reshape(Xtrain.shape[0], 784)
Xte = Xtest.reshape(Xtest.shape[0], 784)
Xtr = Xtr.astype('float32')
Xte = Xte.astype('float32')
#normalizong data
Xtr /= 255
Xte /= 255
# use utilities to transform data
# Convert 1-dimensional class arrays to 10-dimensional class matrices
Ytr = np_utils.to_categorical(Ytrain)
Yte = np_utils.to_categorical(Ytest)


#%% D E S I N I N G----THE----M O D E L----------------------------------------
'Load Sequence:3'

#------------------------------------------------------------------------------
MLP = Sequential()
# first hidden layer 
MLP.add(Dense(512,activation = 'relu',input_shape = (784,)))
# second hidden layer
MLP.add(Dense(512,activation = 'relu'))
# output layer have 10 nodes for 10 labels
MLP.add(Dense(10,activation = 'softmax'))
#declare the loss function and the optimizer 
sgd = optimizers.SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True )
MLP.compile(optimizer=sgd, loss='categorical_crossentropy', 
            metrics=['accuracy'])


#%% T R A I N----THE----M O D E L----------------------------------------------
'Load Sequence:3'

#------------------------------------------------------------------------------
TrainMLP = MLP.fit(Xtr,Ytr , batch_size=32 , epochs=30 , validation_split=0.2)
history = TrainMLP.history
# attributes for demonstrating results
#['accuracy', 'loss', 'val_accuracy', 'val_loss']
TrainLoss = history['loss']
ValidationLoss = history['val_loss']
TrainAccuracy = history['accuracy']
ValidationAccuracy = history['val_accuracy']
#------------------------------------------------------------------------------
#figure visualizations
plot.xlabel('Epochs')
plot.ylabel('Loss')
plot.plot(ValidationLoss)
plot.plot(TrainLoss)
plot.legend(['Train','Validation'])
plot.savefig('Loss vs Epochs for MNIST MLP')
plot.show()
#------------------------------------------------------------------------------
plot.plot(TrainAccuracy)
plot.plot(ValidationAccuracy)
plot.xlabel('Epochs')
plot.ylabel('Accuracy')
plot.legend(['Train','Validation'])
plot.savefig('Accuracy vs Epoch for MNIST MLP')
plot.show()
#print ('The loss on validation data is  : ',ValidationLoss[29])
#print ('The accuracy on validation data is  : ',ValidationAccuracy[29])


#%% T E S T----THE----M O D E L------------------------------------------------
'Load Sequence:3'

#------------------------------------------------------------------------------
Predict = MLP.predict(Xte)
TestLoss,TestAccuracy = MLP.evaluate(Xte,Yte)
print ('The loss on test data is  : ',TestLoss )
print ('The accuracy on test data is : ',TestAccuracy )
plot.imshow (Xtest[20],cmap='binary')
#
