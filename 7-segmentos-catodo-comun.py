import numpy as np
x =np.array([[0,0,0,0],
             [0,0,0,1],
             [0,0,1,0],
             [0,0,1,1],
             [0,1,0,0],
             [0,1,0,1],
             [0,1,1,0],
             [0,1,1,1],
             [1,0,0,0],
             [1,0,0,1],
             [1,0,1,0],
             [1,0,1,1],
             [1,1,0,0],
             [1,1,0,1],
             [1,1,1,0],
             [1,1,1,1]
])

y = np.array([
    [1,1,1,1,1,1,0],  
    [1,1,0,0,0,0,0],  
    [1,0,1,1,0,1,1],  
    [1,1,1,1,0,0,1],  
    [0,1,1,0,0,1,1],  
    [0,1,1,0,1,1,1], 
    [0,1,1,1,1,1,1],  
    [1,1,1,0,0,0,0],  
    [1,1,1,1,1,1,1],  
    [1,1,1,1,0,1,1],  
    [1,1,1,0,1,1,1],  
    [0,0,1,1,1,1,1],  
    [0,0,1,1,1,1,0],  
    [0,1,1,1,1,0,1],  
    [1,0,0,1,1,1,1],  
    [1,0,0,0,1,1,1]   
])



fa = 0.005 #factor de aprendizaje
def sig(x):
  return 1/(1-np.exp(-x))
def tanh(x):
  return np.tanh(x)
def escalon(x):
  return np.where(x >= 0,1,0)
def devsig(x):
  return x*(1-x)
def devtanh(x):
  return 1 - x**2



p1 = np.random.rand(4,3)# enla primer capa simepre esta la entrada
p2 = np.random.rand(3,2)
p3 = np.random.rand(2,2)
p4 = np.random.rand(2,7)# en la ultima esta la salida
epocas = 10000
for i in range(epocas):
  c1 = tanh(np.dot(x,p1))
  c2 = tanh(np.dot(c1,p2))
  c3 = tanh(np.dot(c2,p3))
  c4 = tanh(np.dot(c3,p4))#  prediccion
  #print(c4.round())
  error = y - c4
  #print(error)

  #calculo de la desviacion del error
  g1 = devtanh(c4) * error
  g2 = np.dot(g1,p4.T) * devtanh(c3)
  g3 = np.dot(g2,p3.T) * devtanh(c2)
  g4 = np.dot(g3,p2.T) * devtanh(c1)

  #nc3 = devtanh(c3) * error
  #rc3 = np.dot(nc3,p3.T) * devtanh(c2)
  #rc2 = np.dot(rc3,p2.T) * devtanh(c1)

  #actualizacion de pesos
  #p3 = p3 +(fa*(np.dot(c2.T,nc3)))
  #p2 = p2 +(fa*(np.dot(c1.T,rc3)))
  #p1 = p1 +(fa*(np.dot(x.T,rc2)))

  p4 += fa *(np.dot(c3.T,g1))
  p3 += fa *(np.dot(c2.T,g2))
  p2 += fa *(np.dot(c1.T,g3))
  p1 += fa *(np.dot(x.T,g4))

for h in range(len(y)):

  print('Prediccion: '+str(c4[h]) + '   Resultado'+ str(y[h]))

print('\n')
print(p1)
print('\n')
print(p2)
print('\n')
print(p3)
print('\n')
print(p4)


p1= np.array([[ 1.95168054, -0.91961666,  0.7934049 ],
 [ 2.36676059,  0.86170246, -0.25992743],
 [ 2.13798884,  1.63920287, -0.56761594],
 [ 0.55931397,  0.75360765, -0.53019257]])

p2 = np.array([[ 1.49072711,  2.07505955],
 [ 1.67427692, -1.26789938],
 [ 4.37954965,  0.26263378]])

p3= np.array([[ 3.655682,  -1.03011394],
 [ 1.13844826,  2.54786913]])

p4=np.array([[-0.33219637, -2.56729388,  0.19193993,  0.34185407,  0.58350357,  0.86209223,
   1.47054041],
 [ 1.17348117,  4.24189099,  1.24341503,  0.65244041,  0.4825911,   0.71231614,
   1.60970805]])


    
x1= int(input('dame un 0 o 1'))
x2= int(input('dame un 0 o 1'))
x3= int(input('dame un 0 o 1'))
x4= int(input('dame un 0 o 1'))

entrada= np.array([[x1,x2,x3,x4]])

c1 = tanh(np.dot(entrada,p1))
c2 = tanh(np.dot(c1,p2))
c3 = tanh(np.dot(c2,p3))
c4 = tanh(np.dot(c3,p4))#  prediccion
print(c4.round())