

"""
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation,Reshape
from keras.layers import LSTM,GRU,SimpleRNN
from keras.optimizers import RMSprop,Adam
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
path = get_file('shakespeare.txt', origin='https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path).read().lower()
print('text length:', len(text))
chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
def shifting(tool_koleh_data,tool_bordar_amoozesh,shifting,step):
 global maxlen
 maxlen = tool_bordar_amoozesh
 shift=tool_koleh_data-shifting
 step = step
 sentences = []
 next_chars = []
 for i in range(0, len(text) - maxlen, step):
     sentences.append(text[i: i + maxlen])
     next_chars.append(text[i+shifting:i + maxlen+shift])

 X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
 y = np.zeros((len(sentences),5, len(chars)), dtype=np.bool)
 for i, sentence in enumerate(sentences):
     for t, char in enumerate(sentence):
         X[i, t, char_indices[char]] = 1
 for i, next_char in enumerate(next_chars):
     for t, char in enumerate(next_char):
         y[i ,t ,char_indices[char]] = 1
 return X,y
def make(preds):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / 0.15
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
x_train,y_train=shifting(45,40,40,10)
model = Sequential()
model.add(LSTM(200, input_shape=(maxlen, len(chars))))
#model.add(GRU(200, input_shape=(maxlen, len(chars))))
#model.add(SimpleRNN(200, input_shape=(maxlen, len(chars))))
model.add(Dense(195))
model.add(Reshape((5,39)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_hinge', optimizer=optimizer)
import datetime
start=datetime.datetime.now()
trained_model=model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2)
end=datetime.datetime.now()
Total_time_training=end-start
print ('Total_time_training:',Total_time_training )

history=trained_model.history

losses=history['loss']
val_losses=history['val_loss']

import matplotlib.pyplot as plt
plt.title("loss of LSTM _ RMSprop _ categorical_hinge")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(losses)
plt.plot(val_losses)
plt.legend(['loss','val_loss'])
plt.savefig("loss of LSTM _ RMSprop _ categorical_hinge.png")
plt.figure()
start_index=13
#start_index = random.randint(0, len(text) - maxlen - 1)


generated = ''
sentence = text[start_index: start_index + maxlen]
generated += sentence
print('" the beginning sentence' + sentence + '"')
sys.stdout.write(generated)
for i in range(40):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = make(preds[0,:])
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
print()
