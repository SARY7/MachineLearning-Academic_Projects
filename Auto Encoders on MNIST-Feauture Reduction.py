
#%% I M P O R T I N G----L I B R A R I E S-------------------------------------
'Load Sequence:1'

from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense,Input
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


#%% T R A I N----THE----M O D E L----------------------------------------------
'Load Sequence:3'

#------------------------------------------------------------------------------
# Implementing autoencoder
#------------------------------------------------------------------------------
EncodingDimension = 32
InputLayer = Input(shape = (784,))
EncoderLayer = Dense(EncodingDimension, activation = 'relu')(InputLayer)
DecoderLayer = Dense(784, activation = 'sigmoid')(EncoderLayer)
AutoEncoder = Model(InputLayer,DecoderLayer)
AutoEncoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy', metrics=['accuracy'])
#------------------------------------------------------------------------------
Encoder = Model(InputLayer,EncoderLayer)
AutoEncoder.fit(Xtr, Xtr, epochs = 10, batch_size = 4 , validation_split=0.2)
#------------------------------------------------------------------------------
# Reducing Features
EncodedTrain = Encoder.predict(Xtr)
EncodedTest = Encoder.predict(Xte)
#------------------------------------------------------------------------------
# Designing MLP
MLP = Sequential()
MLP.add(Dense(512,activation = 'relu',input_shape = (EncodingDimension,)))
MLP.add(Dense(512,activation = 'relu'))
MLP.add(Dense(10,activation = 'softmax'))
sgd = optimizers.SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)
MLP.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
TrainMLP = MLP.fit(EncodedTrain ,Ytr , batch_size=32 , epochs=30, validation_split=0.2)
#------------------------------------------------------------------------------
history=TrainMLP.history
# attributes for demonstrating results
#['accuracy', 'loss', 'val_accuracy', 'val_loss']
TrainLoss = history['loss']
#ValidationLoss = history['val_loss']
TrainAccuracy = history['accuracy']
#ValidationAccuracy = history['val_accuracy']
#------------------------------------------------------------------------------
#figure visualizations
plot.xlabel('Epochs')
plot.ylabel('Loss')
#plot.plot(ValidationLoss)
plot.plot(TrainLoss)
plot.legend(['Train'])
plot.title('Accuracy vs Epoch for MNIST MLP with Encoder')
plot.savefig('Loss vs Epochs for MNIST MLP with Encoder')
plot.show()
#------------------------------------------------------------------------------
plot.plot(TrainAccuracy)
#plot.plot(ValidationAccuracy)
plot.xlabel('Epochs')
plot.ylabel('Accuracy')
plot.legend(['Train'])
plot.title('Accuracy vs Epoch for MNIST MLP with Encoder')
plot.savefig('Accuracy vs Epoch for MNIST MLP with Encoder')
plot.show()
#print ('The loss on validation data is  : ',ValidationLoss[29])
#print ('The accuracy on validation data is  : ',ValidationAccuracy[29])

#%% T E S T----THE----M O D E L------------------------------------------------
'Load Sequence:3'

#------------------------------------------------------------------------------
Predict = MLP.predict(EncodedTest)
TestLoss,TestAccuracy = MLP.evaluate(EncodedTest,Yte)
print ('The loss on test data is  : ',TestLoss )
print ('The accuracy on test data is : ',TestAccuracy )
#plot.imshow(Xte[20],cmap='binary')
