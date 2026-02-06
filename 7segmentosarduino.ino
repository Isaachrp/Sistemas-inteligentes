int dipPins[4] = {2, 3, 4, 5};
int segPins[7] = {6, 7, 8, 9, 10, 11, 12};

// Definimos números 0-9 y letras A-F
byte numeros[16][7] = {
  {1,1,1,1,1,1,0}, // 0
  {0,1,1,0,0,0,0}, // 1
  {1,1,0,1,1,0,1}, // 2
  {1,1,1,1,0,0,1}, // 3
  {0,1,1,0,0,1,1}, // 4
  {1,0,1,1,0,1,1}, // 5
  {1,0,1,1,1,1,1}, // 6
  {1,1,1,0,0,0,0}, // 7
  {1,1,1,1,1,1,1}, // 8
  {1,1,1,1,0,1,1}, // 9
  {1,1,1,0,1,1,1}, // A
  {0,0,1,1,1,1,1}, // b
  {1,0,0,1,1,1,0}, // C
  {0,1,1,1,1,0,1}, // d
  {1,0,0,1,1,1,1}, // E
  {1,0,0,0,1,1,1}  // F
};

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < 4; i++) pinMode(dipPins[i], INPUT_PULLUP); // Pull-up físico
  for (int i = 0; i < 7; i++) pinMode(segPins[i], OUTPUT);
}

int leerDIP() {
  int valor = 0;
  for (int i = 0; i < 4; i++) {
    valor |= digitalRead(dipPins[i]) << i;
  }
  return valor;
}

void mostrarNumero(int n) {
  if (n < 0 || n > 15) return; // Solo 0-15
  for (int i = 0; i < 7; i++) {
    digitalWrite(segPins[i], numeros[n][i]);
  }
}

void loop() {
  int entrada = leerDIP();
  Serial.println(entrada);  // En Monitor Serial
  delay(500);

  if (Serial.available()) {
    int prediccion = Serial.parseInt();
    mostrarNumero(prediccion);
  }
}
