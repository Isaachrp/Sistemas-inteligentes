import numpy as np

# Entradas de la compuerta AND
x = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Salidas de la compuerta AND
y = np.array([0,0,0,1])

# Pesos
pesos1 = np.random.rand(2,3)
pesos2 = np.random.rand(3,1)

# Funciones de activación
def tanh(x):
    return np.tanh(x)

# Propagación hacia adelante
for i in range(len(x)):
    c1 = tanh(np.dot(x[i], pesos1))
    c2 = tanh(np.dot(c1, pesos2))
    print(c2)

print("\nPesos capa 1:")
print(pesos1)
print("Pesos capa 2:")
print(pesos2)
