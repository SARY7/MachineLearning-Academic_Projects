
#%% I M P O R T I N G----D A T A----&----L I B R A R I E S---------------------
'Load Sequence:1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from matplotlib.colors import ListedColormap

#%% P R O B L E M----4----D A T A----------------------------------------------
'Load Sequence:2'

#------------------------------------------------------------------------------
data = []
x1 = np.random.rand(30)*5
x2 = np.random.rand(30)*5
for i in range(len(x1)):
    data.append([x1[i],x2[i]])
    
#%% D E S I N I N G----M & P N E U R O N------C L A S S------------------------
'Load Sequence:3'

#------------------------------------------------------------------------------
for x in data:
    net1 = x[1] - 2
    net2 = x[1] + 2
    net3 = x[1] - 4*x[0] -2
    net4 = x[1] + 4*x[0] -14 
    if net1<=0 and net2>=0 and net3<=0 and net4<=0 :
        netTotal = 1
        X1 = plot.scatter(x[0],x[1] ,color = 'black')
    else:
        netTotal = -1
        X2 = plot.scatter(x[0],x[1] ,color = 'blue')
plot.legend((X1,X2),('inside convex','out of convex'),loc='best')

        
#%% P L O T T I N G----R E S U L T S-------------------------------------------
'Load Sequence:3'

#------------------------------------------------------------------------------       
X1x = list(np.arange(0,3.1, 0.1))
X1y = 2*np.ones(len(X1x))
X2x = list(np.arange(-1,4.1,0.1))
X2y = -2*np.ones(len(X2x))   
X3x = list(np.arange(-1 ,0.1 , 0.1))
X3y = []
for j in X3x:
    x3 = 4 * j +2
    X3y.append(x3)
X4x = list(np.arange(3 ,4.1 , 0.1))
X4y = []
for k in X4x:
    x4 = ((-4) * k) +14
    X4y.append(x4)
plot.plot(X1x,X1y , color = 'green')
plot.plot(X2x,X2y ,color = 'green')
plot.plot(X3x,X3y ,color = 'green')
plot.plot(X4x,X4y , color = 'green')
plot.xlabel("X1: Feature1")
plot.ylabel("X2: Feature2")
plot.title("Data Classification of a Convex shape")
plot.savefig("Data Classification of a Convex shape problem4")
plot.show
