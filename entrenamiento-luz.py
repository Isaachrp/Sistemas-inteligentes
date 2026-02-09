import tensorflow as tf
import numpy as np
import pandas as pd

archivo = pd.read_excel('datos_entrenamiento_foco.xlsx')
x = archivo[['luminosidad', 'presencia', 'hora', 'dia']]
print(x.shape)
y = archivo[['salida']]
y = np.array(y)
# print(y.shape)


# Datos normalizados
x['luminosidad'] = x['luminosidad'] / 10
x['presencia'] = x['presencia']          # se queda igual (0 o 1)
x['hora'] = x['hora'] / 24
x['dia'] = (x['dia'] - 1) / 6

x = np.array(x)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(4, activation="sigmoid"),
        tf.keras.layers.Dense(5, activation="relu"),
        tf.keras.layers.Dense(5, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

model.fit(x, y, epochs=500, verbose=0)

model.save('leds.keras')

Predicciones = model.predict(x)
print(Predicciones.round())
