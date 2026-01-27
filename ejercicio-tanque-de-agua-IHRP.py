import numpy as np

# Datos (normalizados)
x = np.array([[20], [19], [90], [100]], dtype=float)
x = x / 100

y = np.array([[1], [1], [1], [0]], dtype=float)

# Funciones de activación
def sig(x):
    return 1 / (1 + np.exp(-x))

def dsig_from_output(s):
    return s * (1 - s)

# Inicialización de pesos
def init_w(n_in, n_out):
    return np.random.randn(n_in, n_out) * np.sqrt(1 / n_in)

w1 = init_w(1, 3)
w2 = init_w(3, 3)
w3 = init_w(3, 1)

# Hiperparámetros
epocas = 100000
fa = 0.01

# Entrenamiento
for _ in range(epocas):

    # Forward
    c1 = sig(np.dot(x, w1))
    c2 = sig(np.dot(c1, w2))
    c3 = sig(np.dot(c2, w3))

    # Error
    error = y - c3

    # Backprop
    g3 = error * dsig_from_output(c3)
    g2 = np.dot(g3, w3.T) * dsig_from_output(c2)
    g1 = np.dot(g2, w2.T) * dsig_from_output(c1)

    # Actualización de pesos
    w3 += fa * np.dot(c2.T, g3)
    w2 += fa * np.dot(c1.T, g2)
    w1 += fa * np.dot(x.T, g1)

# Forward final (predicción)
c1 = sig(np.dot(x, w1))
c2 = sig(np.dot(c1, w2))
c3 = sig(np.dot(c2, w3))

# Resultados
for i in range(len(y)):
    print(f"Entrada: {x[i][0]:.2f}  Predicción: {c3[i][0]:.4f}  Resultado: {y[i][0]}")
