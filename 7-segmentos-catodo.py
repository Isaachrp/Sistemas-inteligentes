import numpy as np

x = np.array([
 [0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],
 [0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],
 [1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],
 [1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]
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

fa = 0.005

def sig(x):
  return 1/(1+np.exp(-x))

def devsig(x):
  return x*(1-x)

def tanh(x):
  return np.tanh(x)

def devtanh(y):
  return 1 - y**2

#pesos
p1 = np.random.uniform(-1,1,(4,3))# se ua uniform en vez de rand porque valance el aprendizaje va generando tambien numeros negativos
p2 = np.random.uniform(-1,1,(3,2))
p3 = np.random.uniform(-1,1,(2,2))
p4 = np.random.uniform(-1,1,(2,7))

#entrenamiento y ajuste de pesos
epocas = 10000

for _ in range(epocas):
  c1 = tanh(np.dot(x,p1))
  c2 = tanh(np.dot(c1,p2))
  c3 = tanh(np.dot(c2,p3))
  c4 = tanh(np.dot(c3,p4))

  error = y - c4

  g4 = error * devtanh(c4)
  g3 = np.dot(g4,p4.T) * devtanh(c3)
  g2 = np.dot(g3,p3.T) * devtanh(c2)
  g1 = np.dot(g2,p2.T) * devtanh(c1)

  p4 += fa * np.dot(c3.T,g4)
  p3 += fa * np.dot(c2.T,g3)
  p2 += fa * np.dot(c1.T,g2)
  p1 += fa * np.dot(x.T,g1)

print("p1 =", p1)
print("p2 =", p2)
print("p3 =", p3)
print("p4 =", p4)


for h in range(len(y)):
    print('Prediccion: '+str(c4[h]) + '   Resultado'+ str(y[h]))

#primeras predicciones comparadas con las salidas esperadas
# for h in range(len(y)):
#     print('Prediccion: '+str(np.where(c4[h]> 0,1,0)) + '   Resultado'+ str(y[h]))

# Prueba manual
#aqui ya no es necesario definir los pesos estaticos ya que los valores de p toman los ultimos
x1 = int(input('0 o 1: '))
x2 = int(input('0 o 1: '))
x3 = int(input('0 o 1: '))
x4 = int(input('0 o 1: '))

entrada = np.array([[x1,x2,x3,x4]])

c1 = tanh(np.dot(entrada,p1))
c2 = tanh(np.dot(c1,p2))
c3 = tanh(np.dot(c2,p3))
c4 = tanh(np.dot(c3,p4))

print("Salida 7 segmentos:")
print(np.where(c4 >= 0,1,0))# se usa where en vez de round porque la funcion tanh devuelve valores entre -1 y 0 y round da problemas al redondear negativos con los resultados 
#print(c4.round())