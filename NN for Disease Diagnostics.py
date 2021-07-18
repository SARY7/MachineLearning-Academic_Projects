
#%% I M P O R T I N G----D A T A----&----L I B R A R I E S---------------------
import numpy as np
 

#%% D E S I N I N G----C O P E T E T I V E----N E T W O R K--------------------

# Designing a class to predict the test of illness
def Test(person):
 positive =np.array([[-1,-1,-1,-1,-1,1,1,1,1,1,1]])
 negative =np.array([[-1,-1,-1,-1,-1,-1,1,1,1,1,1]])
 positive = positive/2
 negative = negative/2
 Weights = np.zeros((11,2))
 Weights[:,0] = positive.T[:,0]
 Weights[:,1] = negative.T[:,0]
 b = np.size(positive)/2
 h = b + np.dot(person,Weights)
 p,d=np.where(h==np.max(h))
 if d==0:
     print(":(( The person is ill")
 else:
     print(":)) The person is healthy")

#%% T E S T---of---the N E T W O R K-------------------------------------------
# in this part, we will test the network with 4 example
p1 = np.array([[-1,-1,-1,-1,-1,-1,1,1,1,1,1]])
print("Result of the test for person1: ")
Test(p1)




