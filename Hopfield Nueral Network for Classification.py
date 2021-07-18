
#%% I M P O R T I N G----D A T A----&----L I B R A R I E S---------------------

import numpy as np

ZeroPatt = np.ones((8,8))
ZeroPatt[2:6,2:6] = -1
ZeroPattVector = ZeroPatt.reshape(ZeroPatt.size,1)
OnePatt = -np.ones((8, 8))
OnePatt[:,3:5] = 1
OnePatt[0:2,2] = 1
OnePattVector = OnePatt.reshape(OnePatt.size, 1)


#%% H O P F I E L D----N E T W O R K-------------------------------------------

weights = np.zeros((ZeroPatt.size,ZeroPatt.size))
weights = weights + np.matmul(ZeroPattVector, np.transpose(ZeroPattVector))
weights = weights + np.matmul(OnePattVector, np.transpose(OnePattVector))
for i in range(weights.shape[0]):
    weights[i, i] = 0
    
    
#%% D I S T O R T E D----D A T A-----------------------------------------------

DistortedZero = np.copy(ZeroPattVector)
DistrotedOne = np.copy(OnePattVector)
np.random.seed(0)
in1 = np.random.choice(np.arange(64),
                            size = 15,
                            replace = False)
in2 = np.random.choice(np.arange(64),
                            size = 15,
                            replace = False)

for i in range (in1.size):
    DistortedZero[in1[i], 0] = np.random.choice([-DistortedZero[in1[i], 0], 0])
    DistrotedOne[in2[i], 0] = np.random.choice([-DistrotedOne[in2[i], 0], 0])
    
    
#%% T E S T I N G----N E T W O R K----WITH----D I S T O R T E D----D A T A-----

Arrange = np.arange(64)
def Output(x,y):
    h = x.reshape(x.size, 1)
    net = y.reshape(y.size, 1)
    for i in range (x.size):
        if (net[i] > 0):
            h[i] = 1
        if (net[i] < 0):
            h[i] = -1
    return h


# test of the network fot the distorted zero pattern as input
DZ = np.copy(DistortedZero)
print("Test of Distorted Zero: Predicted Output is: ")
while True:
    np.random.shuffle(Arrange)
    for i in range (Arrange.size):
        temp = np.matmul(weights[Arrange[i], :], DZ) + DistortedZero[Arrange[i]]
        DZ[Arrange[i]] = Output(DZ[Arrange[i]], temp)
    if (sum(DZ == ZeroPattVector) == 64):
        print(DZ.reshape(8, 8))
        print("Zero")
        break
    if (sum(DZ == OnePattVector) == 64):
        print(DZ.reshape(8, 8))
        print("One")
        break

# test of the network fot the distorted zero pattern as input
DO = np.copy(DistrotedOne)
print("Test of Distorted One: Predicted Output is: ")
while True:
    np.random.shuffle(Arrange)
    for i in range (Arrange.size):
        temp = np.matmul(weights[Arrange[i], :], DO) + DistrotedOne[Arrange[i]]
        DO[Arrange[i]] = Output(DO[Arrange[i]], temp)
    if (sum(DO == ZeroPattVector) == 64):
        print(DO.reshape(8, 8))
        print("Zero")
        break
    if (sum(DO == OnePattVector) == 64):
        print(DO.reshape(8, 8))
        print("One")
        break