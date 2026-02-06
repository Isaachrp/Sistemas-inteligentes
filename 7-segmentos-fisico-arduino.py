import serial
from sklearn.neural_network import MLPClassifier
import time

# --- Configuración Serial ---
# Cambia 'COM3' por tu puerto real (en Linux algo como '/dev/ttyACM0')
#ser = serial.Serial('COM3', 9600, timeout=1)
#time.sleep(2)  # Espera que Arduino inicie

datos_prueba = [0,1,2,3,4,5,6,7,8,9,10]
for entrada in datos_prueba:
    bits = [(entrada>>3)&1, (entrada>>2)&1, (entrada>>1)&1, entrada&1]
    pred = modelo.predict([bits])
    print(f"DIP: {bits} → Predicción: {pred[0]}")


# --- Dataset para entrenar la red neuronal ---
# 4 bits de entrada (DIP switch)
X = [[(i>>3)&1, (i>>2)&1, (i>>1)&1, i&1] for i in range(16)]
# Salida: números 0-9 (si es >9, se mapea a 0-9)
y = [i if i<10 else i%10 for i in range(16)]

# --- Crear y entrenar la red neuronal ---
modelo = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', max_iter=2000)
modelo.fit(X, y)

print("Red neuronal entrenada. Esperando datos de Arduino...")

# --- Bucle principal: leer Arduino y enviar predicción ---
while True:
    dato = ser.readline().decode().strip()  # Lee línea Serial
    if dato.isdigit():
        entrada = int(dato)
        # Convertir el número a bits (DIP switch)
        bits = [(entrada>>3)&1, (entrada>>2)&1, (entrada>>1)&1, entrada&1]
        # Predicción de la red neuronal
        pred = modelo.predict([bits])
        print(f"DIP: {bits} → Predicción: {pred[0]}")
        # Enviar predicción a Arduino
        ser.write(f"{pred[0]}\n".encode())
