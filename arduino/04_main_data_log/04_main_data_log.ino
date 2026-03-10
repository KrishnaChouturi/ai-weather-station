
  //Reads BME280 sensor every 5 minutes
  //Saves data to CSV file on SD card


#include <Wire.h>
#include <SPI.h>
#include "SD.h"
#include "FS.h"
#include <Adafruit_BME280.h>
#include <Adafruit_Sensor.h>

#define CS_PIN 5
#define LOG_INTERVAL 300000  

Adafruit_BME280 bme;
unsigned long lastLogTime = 0;
int rowCount = 0;

void setup() {
  Serial.begin(115200);
  delay(3000);

  Serial.println("Weather Station Starting");

 
  if (!bme.begin(0x76)) {
    Serial.println("BME280 failed");
    while (1);
  }
  Serial.println("BME280 successful");

  SPI.begin(18, 19, 23, 5);
  if (!SD.begin(CS_PIN)) {
    Serial.println("SD card fail");
    while (1);
  }
  Serial.println("SD card success");

  if (!SD.exists("/data.csv")) {
    File dataFile = SD.open("/data.csv", FILE_WRITE);
    if (dataFile) {
      dataFile.println("timestamp,temperature_c,humidity_pct,pressure_hpa");
      dataFile.close();
      Serial.println("header row successful");
    } else {
      Serial.println("header row failure");
    }
  } else {
    Serial.println("data.csv already exists, appending...");
  }

  Serial.println("=== Logging started! Reading every 5 minutes ===");

  logData();
}

void loop() {
  unsigned long currentTime = millis();

  if (currentTime - lastLogTime >= LOG_INTERVAL) {
    logData();
  }
}

void logData() {
  lastLogTime = millis();
  rowCount++;

  float temperature = bme.readTemperature();
  float humidity    = bme.readHumidity();
  float pressure    = bme.readPressure() / 100.0;

  unsigned long seconds = millis() / 1000;
  unsigned long minutes = seconds / 60;
  unsigned long hours   = minutes / 60;
  String timestamp = String(hours) + "h " + String(minutes % 60) + "m " + String(seconds % 60) + "s";

  String row = timestamp + "," +
               String(temperature, 2) + "," +
               String(humidity, 2) + "," +
               String(pressure, 2);

  File dataFile = SD.open("/data.csv", FILE_WRITE);
  if (dataFile) {
    dataFile.println(row);
    dataFile.close();
    Serial.println("Row " + String(rowCount) + " saved: " + row);
  } else {
    Serial.println("Failed to write row " + String(rowCount));
  }
}