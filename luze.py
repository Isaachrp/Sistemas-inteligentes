import numpy as np
import pandas as pd

archivo = pd.read_excel('datos_entrenamiento_foco.xlsx')

x = archivo[['luminosidad','presencia','hora','dia']]
print(x.shape)
y = archivo[['salida']]
y = np.array(y)
print(y.shape)