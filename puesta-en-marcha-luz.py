# import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('leds.keras')
x = np.array([[2, 1, 20, 1]])
prediccion = model.predict(x, verbose=0)
print(prediccion.round())
