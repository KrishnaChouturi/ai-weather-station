// 06_integration.ino
// Hardware:
//   BME280  - I2C, SDA=IO21, SCL=IO22, addr=0x76, VCC=3.3V
//   SD card - SPI, MISO=IO19, MOSI=IO23, SCK=IO18, CS=5, VCC=VIN (5V)
//   Rain gauge - GPIO IO27, INPUT_PULLUP, FALLING interrupt


#include <Wire.h>
#include <Adafruit_BME280.h>
#include <Adafruit_Sensor.h>
#include <SPI.h>
#include <SD.h>
#include <esp_task_wdt.h>

#define SD_CS_PIN 5
#define RAIN_PIN 27
#define MM_PER_TIP 0.2794      // each bucket tip = 0.2794mm of rain (Tested)
#define DEBOUNCE_MS 500        
#define LOG_INTERVAL_MS (5UL * 60UL * 1000UL)
#define WDT_TIMEOUT_S 30    

Adafruit_BME280 bme;
bool bmeOK = false;
bool sdOK = false;

volatile int tipCount = 0;
volatile unsigned long lastTipTime = 0;


void IRAM_ATTR rainISR() {
  unsigned long now = millis();
  if (now - lastTipTime > DEBOUNCE_MS) {
    tipCount++;
    lastTipTime = now;
  }
}


String uptime() {
  unsigned long totalSec = millis() / 1000;
  unsigned long h = totalSec / 3600;
  unsigned long m = (totalSec % 3600) / 60;
  unsigned long s = totalSec % 60;
  return String(h) + "h " + String(m) + "m " + String(s) + "s";
}


void logRow(String timestamp, String temp, String hum, String pres, String rain) {
  if (!sdOK) {
    Serial.println("Warning: SD not available, row skipped");
    return;
  }
  File f = SD.open("/data.csv", FILE_APPEND);
  if (!f) {
    Serial.println("Warning: could not open data.csv, row skipped");
    return;
  }
  f.println(timestamp + "," + temp + "," + hum + "," + pres + "," + rain);
  f.close();
}


void tryInitBME() {
  bmeOK = bme.begin(0x76);
}


void tryInitSD() {
  sdOK = SD.begin(SD_CS_PIN);
}


void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("AI Weather Station starting up...");

  esp_task_wdt_init(WDT_TIMEOUT_S, true);
  esp_task_wdt_add(NULL);
  Serial.println("Watchdog timer started (30s timeout)");

  Wire.begin(21, 22);
  tryInitBME();
  if (bmeOK) {
    Serial.println("BME280 found at 0x76");
  } else {
    Serial.println("BME280 not found - will keep retrying");
  }

  pinMode(RAIN_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(RAIN_PIN), rainISR, FALLING);
  Serial.println("Rain gauge ready on GPIO 27");

  tryInitSD();
  if (!sdOK) {
    Serial.println("SD card failed! Check: VCC on VIN? CS on GPIO 5? FAT32?");
    Serial.println("Will retry every 10 minutes - station keeps running");
  } else {
    Serial.println("SD card OK");
    if (!SD.exists("/data.csv")) {
      File f = SD.open("/data.csv", FILE_WRITE);
      if (f) {
        f.println("timestamp,temperature_c,humidity_pct,pressure_hpa,rainfall_mm");
        f.close();
        Serial.println("Created data.csv with header");
      } else {
        Serial.println("Warning: could not create data.csv");
      }
    } else {
      Serial.println("data.csv already exists, appending to it");
    }
  }

  Serial.println("Setup done. Logging every 5 minutes.\n");
}


void loop() {
  static unsigned long lastLog = 0;
  static unsigned long lastSDRetry = 0;

  esp_task_wdt_reset();

  if (!sdOK && millis() - lastSDRetry >= (10UL * 60UL * 1000UL)) {
    lastSDRetry = millis();
    tryInitSD();
    if (sdOK) {
      Serial.println("SD card reconnected at " + uptime());
      if (!SD.exists("/data.csv")) {
        File f = SD.open("/data.csv", FILE_WRITE);
        if (f) {
          f.println("timestamp,temperature_c,humidity_pct,pressure_hpa,rainfall_mm");
          f.close();
        }
      }
    }
  }

  if (millis() - lastLog >= LOG_INTERVAL_MS) {
    lastLog = millis();

    String ts = uptime();

    noInterrupts();
    int tips = tipCount;
    tipCount = 0;
    interrupts();

    float rainfallMM = tips * MM_PER_TIP;

    if (!bmeOK) {
      tryInitBME();
      if (bmeOK) {
        Serial.println("BME280 reconnected at " + ts);
      }
    }

    String temp, hum, pres;

    if (bmeOK) {
      float t = bme.readTemperature();
      float h = bme.readHumidity();
      float p = bme.readPressure() / 100.0F; 

      bool glitch = (isnan(t) || isnan(h) || isnan(p) ||
                     t < -40.0 || t > 85.0 ||
                     h < 0.0   || h > 100.0 ||
                     p < 800.0 || p > 1100.0);

      if (glitch) {
        bmeOK = false;
        temp = "ERROR";
        hum  = "ERROR";
        pres = "ERROR";
        Serial.println("Sensor glitch at " + ts + " - logged ERROR");
      } else {
        temp = String(t, 1);
        hum  = String(h, 1);
        pres = String(p, 1);
      }

    } else {
      temp = "ERROR";
      hum  = "ERROR";
      pres = "ERROR";
      Serial.println("BME280 disconnected at " + ts + " - logged ERROR");
    }

    String rain = String(rainfallMM, 4);

    logRow(ts, temp, hum, pres, rain);

    Serial.print(ts + "  |  ");
    Serial.print(temp + "C  ");
    Serial.print(hum + "%  ");
    Serial.print(pres + "hPa  ");
    Serial.print(rain + "mm");
    if (tips > 0) {
      Serial.print("  (" + String(tips) + " tip" + (tips == 1 ? "" : "s") + ")");
    }
    Serial.println();
  }

  delay(100);
}