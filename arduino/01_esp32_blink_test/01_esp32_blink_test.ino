// =========================================
// 01 - ESP32 Blink Test
// Got it off a public domain built-in arduino code
// A test to see if ESP32 is responding
// Found out my ESP32 led did not work
// Tested to see if it was responding by serial moniter output
// =========================================

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  
  Serial.begin(115200); //Check everytime
  delay(3000);
  
  Serial.println("ESP32 Blink Test");
  Serial.println("ESP32 is working");
}

void loop() {
  digitalWrite(LED_BUILTIN, HIGH);
  Serial.println("LED ON");
  delay(1000);
  
  digitalWrite(LED_BUILTIN, LOW);
  Serial.println("LED OFF");
  delay(1000);
}