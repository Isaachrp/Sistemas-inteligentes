import tensorflow as tf
import numpy as np

x = np.array(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]
)

y = np.array(
    [
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
    ]
)


model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(5, activation="sigmoid"),
        tf.keras.layers.Dense(5, activation="relu"),
        tf.keras.layers.Dense(5, activation="relu"),
        tf.keras.layers.Dense(4, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

model.fit(x, y, epochs=500)

Predicciones = model.predict(x)
print(Predicciones.round())
