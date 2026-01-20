import numpy as np
x= np.array([[0,0],
             [0,1],
             [1,0],
             [1,1]]) #entradas de la compuerta or
y= np.array([0,1,1,1]) #salidas de la compuerta or
pesos1= np.random.rand(2,3)
#print(pesos1)
pesos2= np.random.rand(3,1)
#print(pesos2)

def linear(x):#Funcion de activacion
    return x #-1 a 1
def tanh(x):#Funcion de activacion
    return np.tanh(x) #0 a 1
def sigmoid(x):#Funcion de activacion
    return 1 / (1 + np.exp(-x))
def relu(x):#Funcion de activacion
    return np.maximum(0,x)
def escalon(x):
    return np.where(x >= 0, 1, 0)

for i in range(len(x)):
  c1 = tanh(np.dot(x[i],pesos1))
  c2= tanh(np.dot(c1,pesos2))
  print(c2)
print("\n")
print(pesos1)
print(pesos2)

# # p1=([[0.6846704 , 0.08354093, 0.81515629],
# #  [0.62733611 ,0.14003769 ,0.98150931]])
# # p2=([[0.94629332],
# #  [0.34388711],
# #  [0.85827111]])
# # for i in range(len(x)):
# #   c1 = tanh(np.dot(x[i],p1))
# #   c2= tanh(np.dot(c1,p2))
# #   print(c2.round())