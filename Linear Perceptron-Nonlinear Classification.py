
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
                error += (update != 0) #calculating error
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




#%% P R O B L E M----1:section3:dataset2----D A T A---------------------------- 
'Load Sequence:3'

#------------------------------------------------------------------------------

data = [[0.75 ,0.75,0 ,0,-1 ],[1 ,1,0 ,0,-1 ],[1.25 ,1.5,0,0,-1 ],
        [0 ,1,0,0 ,1 ],[0.25 ,1.75,0,0 ,1 ],[1 ,2,0,0 ,1 ],
        [1.75 ,1.75,0,0 ,1 ],[2 ,1,0,0 ,1 ],[1.75 ,0,0,0 ,1 ],
        [1 ,0,0 ,0,1 ], [0.25 ,0.25,0,0 ,1 ] ]

dataframe1 = pd.DataFrame(data)
X = dataframe1.iloc[0:10 , 0:4].values
for i in range(10):
    X[i][3]= X[i][0]**2
    X[i][2]= X[i][1]**2 
Y = dataframe1.iloc[0:10 , 4].values
SamplesNum = len(X)
for j in range(SamplesNum):
    if Y[j] == -1:
        red = plot.scatter(X[j,0], X[j,1], color='red', label='class1')
    else:
        blue = plot.scatter(X[j,0], X[j,1], color='blue',  label='class2')
        
#------------------------------------------------------------------------------
#figure visualizations
plot.title('2 class of data classification')
plot.xlabel('X1: Feature1')
plot.ylabel('X2: Feature2')
plot.legend(handles=[red,blue],loc='best')
plot.savefig('2 class of data classification problem.png')
plot.show()




#%% M O D E L----T R A I N I N G-----------------------------------------------
'Load Sequence:4'

#------------------------------------------------------------------------------
classifier = Perceptron(alpha=0.1, epochs=100, threshold=0)
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
x1 = []
x2 = []
b = []
b1 = []
b2 = []
t = np.linspace(0,360,360)
x1 = (classifier.weights[3]/classifier.weights[2])*np.cos(np.radians(t))+classifier.weights[0] - 0.75 
x2 = (classifier.weights[4]/classifier.weights[1])*np.sin(np.radians(t))+classifier.weights[0] - 0.75 
plot.plot(x1,x2,color='black',)
plot.xlabel("X1: Feature1")
plot.ylabel("X2: Feature2")
plot.title("Classification of 2 Class of Data")  
for j in range(SamplesNum):
    if Y[j] == -1:
        red = plot.scatter(X[j,0], X[j,1], color='red', label='class1')
    else:
        blue = plot.scatter(X[j,0], X[j,1], color='blue',  label='class2')
plot.legend(handles=[red,blue],loc='best')
plot.savefig('classified regions of 2 class of data problem with linear perceptron')
plot.show()



#%% G E N E R A L----I N F O R M A T I O N S-----------------------------------
'Load Sequence:6'

#------------------------------------------------------------------------------
weights = classifier.weights[1:]
b = classifier.weights[0]
threshold = classifier.threshold
epochs = classifier.epochs




















