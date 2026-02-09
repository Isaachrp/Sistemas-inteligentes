import numpy as np
import pandas as pd

archivo = pd.read_excel('datos_entrenamiento_foco.xlsx')

x = archivo[['luminosidad', 'presencia', 'hora', 'dia']]
print(x.shape)
y = archivo[['salida']]
y = np.array(y)
print(y.shape)


# Datos normalizados
x['luminosidad'] = x['luminosidad'] / 10
x['presencia'] = x['presencia']          # se queda igual (0 o 1)
x['hora'] = x['hora'] / 24
x['dia'] = (x['dia'] - 1) / 6

x = np.array(x)

# funciones de activacion


def sig(x):
    return 1/(1+np.exp(-x))


def tanh(x):
    return np.tanh(x)


def escalon(x):
    return np.where(x >= 0, 1, 0)


def devsig(s):
    return s*(1-s)


def devtanh(s):
    return 1 - s**2

# inicializacion de pesos


def init_w(n_in, n_out):
    return np.random.randn(n_in, n_out) * np.sqrt(1 / n_in)


w1 = init_w(4, 4)
w2 = init_w(4, 3)
w3 = init_w(3, 2)
w4 = init_w(2, 1)

# Hiperparámetros
epocas = 100000
fa = 0.01

# Entrenamiento
for _ in range(epocas):

    # Forward
    c1 = sig(np.dot(x, w1))
    c2 = sig(np.dot(c1, w2))
    c3 = sig(np.dot(c2, w3))
    c4 = sig(np.dot(c3, w4))

    # Error
    error = y - c4

    # Backprop
    g4 = error * devsig(c4)
    g3 = np.dot(g4, w4.T) * devsig(c3)
    g2 = np.dot(g3, w3.T) * devsig(c2)
    g1 = np.dot(g2, w2.T) * devsig(c1)

    # Actualización de pesos
    w4 += fa * np.dot(c3.T, g4)
    w3 += fa * np.dot(c2.T, g3)
    w2 += fa * np.dot(c1.T, g2)
    w1 += fa * np.dot(x.T, g1)


# Forward final (predicción) aleatoria
# c1 = sig(np.dot(x, w1))
# c2 = sig(np.dot(c1, w2))
# c3 = sig(np.dot(c2, w3))

# Resultados
# for i in range(len(y)):
    # print(f"Entrada: {x[i][0]:.2f}  Predicción: {c3[i][0]:.4f}
    # Resultado: {y[i][0]}")


# Forward final (predicción) ingresada por teclado

x1 = int(input('dame el valor para la luminosidad (1-10)'))
x2 = int(input('dame el valor para la presencia (0/1)'))
x3 = int(input('dame el valor para la hora (0-24)'))
x4 = int(input('dame el valor para el dia (1-7)'))

entrada = np.array([[
    x1 / 10,
    x2,
    x3 / 24,
    (x4 - 1) / 6
]])

c1 = sig(np.dot(entrada, w1))
c2 = sig(np.dot(c1, w2))
c3 = sig(np.dot(c2, w3))
c4 = sig(np.dot(c3, w4))

# Resultados
pred = np.round(c4)

print(f"Salida redondeada: {int(pred[0][0])}")
