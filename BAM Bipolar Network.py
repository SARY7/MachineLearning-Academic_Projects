

#%% I M P O R T I N G----D A T A----&----L I B R A R I E S---------------------

import numpy as np

A = np.ones((5,3))
A[0, 0] = -1 
A[0, 2] = -1
A[1, 1] = -1
A[3, 1] = -1
A[4, 1] = -1
Atarget = np.array([[-1],[-1],[-1]])
B = np.ones((5,3))
B[0, 2] = -1
B[1, 1] = -1
B[2, 2] = -1
B[3, 1] = -1
B[4, 2] = -1
Btarget = np.array([[-1],[-1],[1]])
C = np.ones((5,3))
C[0, 0] = -1
C[1, 1] = -1
C[1, 2] = -1
C[2, 1] = -1
C[2, 2] = -1
C[3, 1] = -1
C[3, 2] = -1
C[4, 0] = -1
Ctarget = np.array([[-1],[1],[-1]])
Patterns = np.zeros((15, 3))
Targets = np.zeros((3, 3))
Patterns[:, 0] = A.reshape(A.size, )
Patterns[:, 1] = B.reshape(B.size, )
Patterns[:, 2] = C.reshape(C.size, )
Targets[:, 0] = Atarget.reshape(Atarget.size)
Targets[:, 1] = Btarget.reshape(Btarget.size)
Targets[:, 2] = Ctarget.reshape(Ctarget.size)


#%% B A M----N E T W O R K-----------------------------------------------------

weight = np.zeros((15, 3))
for i in range (Patterns.shape[1]):
    weight = weight + np.matmul(Patterns[:, i].reshape(Patterns[:, i].size, 1),
    np.transpose(Targets[:, i].reshape(Targets[:, i].size, 1)))
    
    

#%% D I S T O R T E D----D A T A-----------------------------------------------
    
DistData = np.copy(Patterns)

inA = np.random.choice(np.arange(15),size = 2,replace = False)
inB = np.random.choice(np.arange(15),size = 3,replace = False)
inC = np.random.choice(np.arange(15),size = 4,replace = False)

# A pattern
for i in range (inA.size):
    DistData[inA[i], 0] = np.random.choice([-DistData[inA[i], 0], 0])

# B pattern   
for i in range (inB.size):
    DistData[inB[i], 1] = np.random.choice([-DistData[inB[i], 1], 0])

# C pattern  
for i in range (inC.size):
    DistData[inC[i], 2] = np.random.choice([-DistData[inC[i], 2], 0])
    


#%% T E S T I N G----N E T W O R K----WITH----D I S T O R T E D----D A T A-----

def Output(x,y):
    h = x.reshape(x.size, 1)
    net = y.reshape(y.size, 1)
    for i in range (x.size):
        if (net[i] > 0):
            h[i] = 1
        if (net[i] < 0):
            h[i] = -1
    return h

print("Test of Distorted Patterns: Predicted Output: ")
for i in range(DistData.shape[1]):
    print("for pattern ", chr(ord('A')+i) , "is")
    y2 = np.copy(DistData[:, i])
    y1 = np.zeros((3, 1))
    y2p = np.zeros((DistData.shape[0], 1)) 
    y1p = np.zeros((3, 1))
    while True:
        t1 = np.matmul(np.transpose(weight), y2)
        y1 = Output(y1,t1)
        t2 = np.matmul(weight, y1)
        y2 = Output(y2,t2)
        if (sum(y1 == y1p) == y1.size):
            if (sum(y2 == y2p) == y2.size):
                print("Targeted output of the layer x is : ")
                print(y2.reshape(5, 3))
                print("Targeted Output of the layer y is : ")
                print(np.transpose(y1))
                if (sum(y1.reshape(3, ) == Targets[:, i]) == 3):
                    if (sum(y2.reshape(15, ) == Patterns[:, i]) == 15):
                        print("BAM successfuly recognized the original & the disturbed pattern")
                    else:
                        print("BAM faild to recognized the disturbed pattern but recognized the original pattern")
                else:
                    print("BAM Completely faild!")
                break
        y1p = np.copy(y1)
        y2p = np.copy(y2)
        
#%% C A L C U L A T I N G----H E M M I N G----D I S T A N C E------------------

for i in range (Targets.shape[1]):
    for j in range (Targets.shape[1]):
        if (i != j):
            print("Hamming Distance Between letter" , chr(ord('A') + i), "and", chr(ord('A') + j), "is")
            dist = sum(Targets[:, i] != Targets[:, j])
            print(dist)
            
            
            