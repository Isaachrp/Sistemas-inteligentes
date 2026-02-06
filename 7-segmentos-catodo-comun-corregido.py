import numpy as np

# ================= DATOS =================
X = np.array([
 [0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],
 [0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],
 [1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],
 [1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]
], dtype=float)

# Salida deseada: 0-9 mapeados a [0,1]
Y = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], dtype=float).reshape(-1,1) / 15.0

# ================= FUNCIONES =================
def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1 - y**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

# ================= PESOS =================
np.random.seed(1)
p1 = np.random.uniform(-0.5, 0.5, (4, 8))
p2 = np.random.uniform(-0.5, 0.5, (8, 1))

lr = 0.01   # Reducimos un poco la tasa de aprendizaje
epochs = 10000  # Menos épocas para evitar overflow

# ================= ENTRENAMIENTO =================
for _ in range(epochs):
    h = tanh(X @ p1)
    out = sigmoid(h @ p2)
    error = Y - out
    g2 = error * dsigmoid(out)
    g1 = (g2 @ p2.T) * dtanh(h)
    p2 += lr * h.T @ g2
    p1 += lr * X.T @ g1

# ================= BUCLE DE SIMULACIÓN =================
print("=== Simulación DIP switch → Python → Arduino ===")
print("Ingresa un número entre 0 y 15 (como si viniera del DIP switch). Escribe 'salir' para terminar.")

while True:
    entrada_str = input("Valor DIP switch: ")
    if entrada_str.lower() == "salir":
        break
    
    if not entrada_str.isdigit():
        print("Ingresa un número válido 0–15")
        continue
    
    entrada = int(entrada_str)
    if entrada < 0 or entrada > 15:
        print("Número fuera de rango 0–15")
        continue
    
    bits = [(entrada >> 3) & 1, (entrada >> 2) & 1, (entrada >> 1) & 1, entrada & 1]
    
    # Predicción de la red neuronal
    salida = sigmoid(tanh(np.array([bits]) @ p1) @ p2)
    
    # Convertimos de [0,1] a 0-9
    salida_redondeada = int(round(salida[0,0] * 15))  # 0-9
    print(f"DIP {bits} -> Número a mostrar en Arduino: {salida_redondeada}\n")
