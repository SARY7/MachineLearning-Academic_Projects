
#%% I M P O R T I N G----D A T A----&----L I B R A R I E S---------------------
'Load Sequence:1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from matplotlib.colors import ListedColormap

#%% D E S I N I N G----P E R C E P T R O N----C L A S S------------------------
'Load Sequence:2'

#------------------------------------------------------------------------------
# Design a Perceptron Class based on Object Oriented Programming Method
class Perceptron(object):
    def __init__(self, alpha=0.01, epochs=10, threshold=1):  
        #Defining attributes of the class
        self.alpha = alpha
        self.epochs = epochs
        self.errors = []
        self.threshold = threshold
        self.loss = []
        #makes a weight vector based on the shape of input vector X.
        self.weights = np.zeros(1+X.shape[1])

#------------------------------------------------------------------------------
    #define a linear model to fit to the data
    def LinearModel(self,X,Y):
         self.weights = np.zeros(1+X.shape[1])
         for i in range(self.epochs):
            error = 0
            for xi,target in zip(X,Y):
                update = self.alpha * (target - self.Predict(xi))
                self.weights[1:] += update*xi #updating weights
                self.weights[0] += update #updating bias term
                error += (update > 0.001) #calculating error
            self.errors.append(error)
         return self
     
#------------------------------------------------------------------------------
    #define the formulation for calculating net value 
    def NetInput(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]
    
#------------------------------------------------------------------------------ 
    #define the formulation for prediction function 
    def Predict(self, X):
        return np.where(self.NetInput(X) >= 0, 1, -1)
        

#%% P R O B L E M----1:section1----D A T A------------------------------------- 
'Load Sequence:3'

#------------------------------------------------------------------------------
IrisData = pd.read_csv('HW1_Q1_data.csv')
X = IrisData.iloc[0:100 , 0:2].values  
Y = IrisData.iloc[0:100 , 2].values
SamplesNum = len(X)
Y = np.where(Y == 'Iris-setosa', -1, 1)
for j in range(SamplesNum):
    if Y[j] == -1:
        setosa=plot.scatter(X[j,0], X[j,1], color='red', label='class1:Setosa')
    else:
        versicolor=plot.scatter(X[j, 0], X[j,1], color='blue',label='class2:Versicolor')
        
#-------------------------------------------------------------------------------------------       
#figure visualizations
plot.legend(handles=[setosa,versicolor],loc='best')
plot.title('Iris data based on sepal and letal length')
plot.xlabel('X1:Petal Length')
plot.ylabel('X2:Sepal Length')
plot.savefig('Iris data based on sepal and letal length.png')
plot.show()



#%% M O D E L----T R A I N I N G-----------------------------------------------
'Load Sequence:4'

#------------------------------------------------------------------------------
classifier = Perceptron(alpha=0.1, epochs=50, threshold=0)
classifier.LinearModel(X,Y)
plot.plot(range(1, len(classifier.errors) + 1), classifier.errors, marker='o')
plot.title('Number of misclassifications for alpha='+ str(classifier.alpha))
plot.xlabel('Numnber of Epochs')
plot.ylabel('Number of misclassifications')
plot.savefig('Number of misclassifications of simple perceprton')
plot.show()

#%% C L A S S I F I E N G----R E G I O N---------------------------------------
'Load Sequence:5'

#------------------------------------------------------------------------------
#Defining function that plots the decision regions.
def plot_decision_regions(X, y, classifier, resolution=0.01):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.Predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plot.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plot.xlim(xx1.min(), xx1.max())
    plot.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plot.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

#----------------------------------------------------------------------------------------------------------
# Showing the final results of the perceptron model.
plot_decision_regions(X, Y, classifier=classifier)
#plot.title('classified regions of Iris problem with linear perceptron for alpha='+ str(classifier.alpha))
plot.title('classified regions of random data problem')
plot.xlabel('X1:Petal Length')
plot.ylabel('X2:Sepal Length')
#plot.savefig('classified regions of Iris problem with linear perceptron')
plot.savefig('classified regions of random data problem')
plot.show()



#%% G E N E R A L----I N F O R M A T I O N S-----------------------------------
'Load Sequence:6'

#------------------------------------------------------------------------------
weights = classifier.weights[1:]
b = classifier.weights[0]
threshold = classifier.threshold
epochs = classifier.epochs


















