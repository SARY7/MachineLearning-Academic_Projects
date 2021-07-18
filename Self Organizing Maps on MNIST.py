
#%% I M P O R T I N G----D A T A----&----L I B R A R I E S---------------------
 
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
import keras
# loading data
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train = train_images
labels = train_labels
# normalizing data
train = train / 255.0


#%% D E S I G N I N G----S O M----N E T W O R K--------------------------------

# Preparing data
Train = np.copy((train[0:1000, ]).astype("float"))
Labels = np.copy(labels[0:1000, ])
Train = np.copy(Train.reshape(Train.shape[0], Train.shape[1]*Train.shape[2]))
np.random.shuffle(Train)
# initiating weights randomly
Weights = 0.5*np.random.rand(Train.shape[1], 625).astype("float")
iteration = 1
LearningRate = 1
decay = 0.8
while True:
    print("Epoch %d" %iteration)
    OldWeights = np.copy(Weights)
    for i in range (Train.shape[0]):
        # Calculating Euclidian Distance
        EuclidianDist = np.sum((Weights-Train[i, :].reshape(784, 1))**2, axis = 0)
        argmin = np.argmin(EuclidianDist)
        # Zero-Radius neurons
        Weights[:,argmin]+=LearningRate*(Train[i,:].reshape(Train[i,:].size, )-Weights[:, argmin].reshape(Weights.shape[0]))
    iteration = iteration + 1
    LearningRate = LearningRate*decay
    if (np.linalg.norm(OldWeights-Weights) < 0.1):
        break
    if (iteration == 3):
        break


#%% P L O T T I N G----R E S U L T S-------------------------------------------

num = np.zeros((625, 1))     
for i in range (Train.shape[0]):
    EuclidianDist = np.sum((Weights-Train[i, :].reshape(Train[i, :].size,1))**2,
                    axis = 0)
    argmin = np.argmin(EuclidianDist)
    num[argmin] = num[argmin] + 1
args = np.argsort(num,axis = 0)
Image = []
for i in range (625):
     Image.append(Weights[:,args[-1-i]].reshape(28,28))
ImageArray = np.asarray(Image).reshape(700,700)
plot.figure()
plot.imshow(ImageArray)
plot.show()
winings = args[::-1]
Clusters = []
for i in range (20):
    Clusters.append("Cluster"+str((i+1)))
table = pd.DataFrame(num[winings[:20]].reshape(1, 20), index = ["Number of Patterns"],columns = Clusters)
print(table)



