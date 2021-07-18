
#%% I M P O R T I N G----D A T A----&----L I B R A R I E S---------------------


import numpy as np

#7*7 Patterns
A = np.array([[-1,-1,-1,1,-1,-1,-1],
                [-1,-1,1,-1,1,-1,-1],
                [-1,-1,1,1,1,-1,-1],
                [-1,1,-1,-1,-1,1,-1],
                [-1,1,-1,-1,-1,1,-1],
                [1,-1,-1,-1,-1,-1,1],
                [1,-1,-1,-1,-1,-1,1]])
Ap = A.reshape(A.size,1)

B = np.array([[1,1,1,1,1,1,-1],
                [1,-1,-1,-1,-1,-1,1],
                [1,-1,-1,-1,-1,1,-1],
                [1,1,1,1,1,-1,-1],
                [1,-1,-1,-1,-1,1,-1],
                [1,-1,-1,-1,-1,-1,1],
                [1,1,1,1,1,1,-1]])

Bp = B.reshape(B.size,1)

C = np.array([[-1,-1,-1,1,1,1,1],
                [-1,1,1,-1,-1,-1,-1],
                [1,-1,-1,-1,-1,-1,-1],
                [1,-1,-1,-1,-1,-1,-1],
                [1,-1,-1,-1,-1,-1,-1],
                [-1,1,1,-1,-1,-1,-1],
                [-1,-1,-1,1,1,1,1]])
Cp = C.reshape(C.size,1)

# 5*3 Patterns
Av = np.array([[-1,1,-1],
                [1,-1,1],
                [1,1,1],
                [1,-1,1],
                [1,-1,1]])
Avp = Av.reshape(Av.size,1)

Bv = np.array([[1,1,-1],
                [1,-1,1],
                [1,1,-1],
                [1,-1,1],
                [1,1,-1]])
Bvp = Bv.reshape(Bv.size,1)

Cv = np.array([[-1,1,1],
                [1,-1,-1],
                [1,-1,-1],
                [1,-1,-1],
                [-1,1,1]])
Cvp = Cv.reshape(Cv.size,1)





#%%
# H E B----L E A R N I N G-----------------------------------------------------

weight = np.zeros((49,15))
weight = weight + np.matmul(Ap,np.transpose(Avp))
weight = weight + np.matmul(Bp,np.transpose(Bvp))
weight = weight + np.matmul(Cp,np.transpose(Cvp))


#%%
# T E S T----of---O R I G I N A L----------D A T A-----------------------------

h1 = np.matmul(np.transpose(Ap), weight)
h2 = np.matmul(np.transpose(Bp), weight)
h3 = np.matmul(np.transpose(Cp), weight)
y1 = np.sign(h1).T
y2 = np.sign(h2).T
y3 = np.sign(h3).T

if (y1 == Avp).all():
        print(y1.reshape(5, 3))
        print("A is associative")
        
if (y2 == Bvp).all():
        print(y2.reshape(5, 3))
        print("B is associative")
        
if (y3 == Cvp).all():
        print(y3.reshape(5, 3))
        print("C is associative")
        

#%%
# T E S T----of---D I S T O R T E D----D A T A---------------------------------
     
A1test = np.copy(Ap)
A2test = np.copy(Bp)
A3test = np.copy(Cp)

np.random.seed(0)

in1 = np.random.choice(np.arange(49),
                            size = 39,
                            replace = False)
in2 = np.random.choice(np.arange(49),
                            size = 39,
                            replace = False)
in3 = np.random.choice(np.arange(49),
                            size = 39,
                            replace = False)

for i in range (in2.size):
    A1test[in1[i], 0] = -A1test[in1[i], 0]
    A2test[in1[i], 0] = -A2test[in2[i], 0]
    A3test[in1[i], 0] = -A3test[in3[i], 0]
    
h1 = np.matmul(np.transpose(A1test), weight)
h2 = np.matmul(np.transpose(A2test), weight)
h3 = np.matmul(np.transpose(A3test), weight)
y1 = np.sign(h1).T
y2 = np.sign(h2).T
y3 = np.sign(h3).T

if (y1 == Avp).all():
        print(y1.reshape(5, 3))
        print("A is associative")
        
if (y2 == Bvp).all():
        print(y2.reshape(5, 3))
        print("B is associative")
        
if (y3 == Cvp).all():
        print(y3.reshape(5, 3))
        print("C is associative")
        

        


#%%
# T E S T----of---M I S S E D P A T T E R N----D A T A-------------------------      

A1test = np.copy(Ap)
A2test = np.copy(Bp)
A3test = np.copy(Cp)

np.random.seed(0)
in1 = np.random.choice(np.arange(49),
                            size = 47,
                            replace = False)
in2 = np.random.choice(np.arange(49),
                            size = 47,
                            replace = False)
in3 = np.random.choice(np.arange(49),
                            size = 47,
                            replace = False)

for i in range (in1.size):
    A1test[in2[i], 0] = 0
    A2test[in2[i], 0] = 0
    A3test[in2[i], 0] = 0
    
h1 = np.matmul(np.transpose(A1test), weight)
h2 = np.matmul(np.transpose(A2test), weight)
h3 = np.matmul(np.transpose(A3test), weight)
y1 = np.sign(h1).T
y2 = np.sign(h2).T
y3 = np.sign(h3).T

if (y1 == Avp).all():
        print(y1.reshape(5, 3))
        print("A is associative")
else:
        print("A is not associative")

if (y2 == Bvp).all():
        print(y2.reshape(5, 3))
        print("B is associative")
else:
        print("B is not associative")        
        
if (y3 == Cvp).all():
        print(y3.reshape(5, 3))
        print("C is associative")
else:
        print("C is not associative")        

