
#%% I M P O R T I N G----D A T A----&----L I B R A R I E S---------------------
'Load Sequence:1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from matplotlib.colors import ListedColormap

#%% P R O B L E M----1:section3:dataset1----D A T A----------------------------
'Load Sequence:3'

#------------------------------------------------------------------------------
data = [[-1,-1,1],[-1,1,-1],[1,-1,-1],[1,1,1]]
#set  non-zero initial value for weights
weights = [0.02,0.07,0.05,0.03,0.06,0.05 ]
x11 = [-1 , 1]
x12 = [-1 , 1]
x21 = [-1 , 1]
x22 = [1 , -1]

#%% D E S I N I N G----M A d a l i n e----C L A S S------------------------
'Load Sequence:2'

#------------------------------------------------------------------------------
#defining weigth of desired MAdaline
b10 = 0.1
b20 = 0.1
b3 = 0.1
alpha = 0.001 #learning rate
epoch = 100 #number of iterations 
b =[]
b1 =[]
#-----------------------------------------------------------------------------
for i in range(epoch):
    L=0
    for x in data:
        NetValue1 = weights[0] + weights[1]*x[0] + weights[2]*x[1]
        NetValue2 = weights[3] + weights[4]*x[0] + weights[5]*x[1]
#------------------------------------------------------------------------------
        if NetValue1 >=0:
            Predict1 = 1
        else:
            Predict1 = -1
#------------------------------------------------------------------------------       
        if NetValue2 >=0:
            Predict2 = 1
        else:
            Predict2 = -1
#------------------------------------------------------------------------------
        y = b3 + b10*Predict1 + b20*Predict2
#------------------------------------------------------------------------------       
        if y >=0:
            Y = 1
        else:
            Y = -1
#------------------------------------------------------------------------------
        if abs(NetValue1)>abs(NetValue2):
            xnor = -1
        else:
            xnor = 1
#------------------------------------------------------------------------------
#updating weights based on adaline method and mri algorithm
        if x[2]== Y:
            weights[0]==weights[0]
            weights[1]==weights[1]
            weights[2]==weights[2]
            weights[3]==weights[3]
            weights[4]==weights[4]
            weights[5]==weights[5]
            L+=1
#------------------------------------------------------------------------------ 
        elif x[2] == 1 and xnor == 1:
            weights[0] += alpha*(1-NetValue1)
            weights[1] += alpha*(1-NetValue1)*x[0]
            weights[2] += alpha*(1-NetValue1)*x[1]
#------------------------------------------------------------------------------            
        elif x[2] == 1 and xnor == -1:
            weights[3] += alpha*(1-NetValue2)
            weights[4] += alpha*(1-NetValue2)*x[0]
            weights[5] += alpha*(1-NetValue2)*x[1]
#------------------------------------------------------------------------------            
        elif x[2] == -1 and NetValue1>=0 :
            weights[0] += alpha*(-1-NetValue1)
            
            weights[1] += alpha*(-1-NetValue1)*x[0]
            
            weights[2] += alpha*(-1-NetValue1)*x[1]
#------------------------------------------------------------------------------             
        elif x[2] == -1 and NetValue2>=0:
            weights[3] += alpha*(-1-NetValue1)
            weights[4] += alpha*(-1-NetValue1)*x[0]
            weights[5] += alpha*(-1-NetValue1)*x[1]
#------------------------------------------------------------------------------     
        error = x[2] - Y
#        if error!=0:
#            L+=1
#------------------------------------------------------------------------------
    if L==4:
        print("epoch : ",i)
        print("b1 : ",weights[0])
        print("weight11 : ",weights[1])
        print("weight21 : ",weights[2])
        print("b2 : ",weights[3])
        print("weight12 : ",weights[4])
        print("weight22 : ",weights[5])
        break

#%% C L A S S I F I E N G----R E G I O N---------------------------------------
'Load Sequence:5'

#------------------------------------------------------------------------------
a =list( np.arange(-2.0,2.0 , 0.1))
for k in a: 
    y = ((-weights[0]-weights[1]*k)/weights[2])
    b.append(y)

for k in a: 
    y1 = ((-weights[3]-weights[4]*k)/weights[5])
    b1.append(y1)
plot.title("XNOR for learning rate= "+str(alpha))
plot.xlim(-2.0,2.0)
plot.ylim(-2.0,2.0)
plot.plot(a,b, color ='r')
plot.plot(a,b1,color ='b')
plot.xlabel("x1")
plot.ylabel("x2")
plot.scatter(x11,x12,s=50.0,c='r',label='Class -1')
plot.scatter(x21,x22,s=50.0,c='b',label='Class 1')
plot.legend(loc='best')
plot.savefig("XNOR")
plot.show()