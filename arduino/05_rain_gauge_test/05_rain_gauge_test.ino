#define RAIN_PIN 27

volatile int tipCount = 0;
unsigned long lastDebounce = 0;
int printCount = 0;

void IRAM_ATTR rainTip() {
  if (millis() - lastDebounce > 500) {
    tipCount++;
    lastDebounce = millis();
  }
}

void setup() {
  Serial.begin(115200);
  delay(3000);
  pinMode(RAIN_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(RAIN_PIN), rainTip, FALLING);
  Serial.println("Starting Rain Gauge Test");
}

void loop() {
  if (printCount >= 30) {
    Serial.println("30 lines reached. Stopped.");
    while (1);
  }

  Serial.print("Tips: ");
  Serial.print(tipCount);
  Serial.println();

  printCount++;
  delay(500);
}