
#%% I M P O R T I N G----D A T A----&----L I B R A R I E S---------------------
import numpy as np

#%% I N I T I A T I N G----W E I G H T S---------------------------------------

v1 = np.ones((4,1))
v1[3] = -1
weight = np.zeros((4,4))
weightp = np.zeros((4,4))
weight = weight + np.matmul(v1,np.transpose(v1))

#%% C A L C U L A T I N G----W E I G H T S-------------------------------------

for i in range (weight.shape[0]):
    for j in range (weight.shape[1]):
        if (i != j):
            weightp[i, j] = weight[i, j]
print("Weigth matrix with non-zero diagonal trems is: ")
print(weight)
print("Weigth matrix with zero diagonal trems is: ")
print(weightp)



#%%
h1 = np.matmul(weight, v1)
h2 = np.matmul(weightp, v1)
y1 = np.sign(h1)
y2 = np.sign(h2)
print("Weigth matrix with non-zero diagonal trems transformed to first vector is: ")
print(y1)
print("Weigth matrix with zero diagonal trems transformed to first vector is: ")
print(y2)

if (y1 == v1).all():
    print('input vector is associative')
else:
    print('input vector is not associative')
    
    
#%% R O B U S T N E S S----T E S T---------------------------------------------
    
v2 = np.ones((4,1))
v2[3] = -1
v2[0] = -1

h1 = np.matmul(weight, v2)
h2 = np.matmul(weightp, v2)
y1 = np.sign(h1)
y2 = np.sign(h2)
print("Weigth matrix with non-zero diagonal trems transformed to first vector is: ")
print(y1)
print("Weigth matrix with zero diagonal trems transformed to first vector is: ")
print(y2)

if (y1 == v1).all():
    print('input vector is associative')
else:
    print('input vector is not associative')
    
    
#%% R O B U S T N E S S----T E S T---------------------------------------------  

v2 = np.ones((4,1))
v2[3] = -1
v2[2] = -1
v2[0] = -1

h1 = np.matmul(weight, v2)
h2 = np.matmul(weightp, v2)
y1 = np.sign(h1)
y2 = np.sign(h2)
print("Weigth matrix with non-zero diagonal trems transformed to first vector is: ")
print(y1)
print("Weigth matrix with zero diagonal trems transformed to first vector is: ")
print(y2)

if (y1 == v1).all():
    print('orginal vector is associate')
else:
    print('input vector is not associative')
