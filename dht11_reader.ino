#include "DHT.h"

// --- Configuration ---
#define DHTPIN 2      // The Arduino pin connected to the DHT11 data pin.
#define DHTTYPE DHT11 // The type of sensor.

// --- Global Variables ---
DHT dht(DHTPIN, DHTTYPE);
bool attack_mode = false;
unsigned long startTime;

void setup() {
  // Start serial communication.
  Serial.begin(9600);
  Serial.println("Smart Factory Sensor Node: ONLINE (Reset-Capable)");

  // Initialize the sensor.
  dht.begin();
  // Record the startup time.
  startTime = millis();
}

void loop() {
  // Wait for 2 seconds between readings.
  delay(2000);

  // After 15 seconds, switch to attack mode permanently.
  if (millis() - startTime > 15000) {
    attack_mode = true;
  }

  // Read data from the sensor.
  float h = dht.readHumidity();
  float t = dht.readTemperature();

  // Check if the reading failed.
  if (isnan(h) || isnan(t)) {
    Serial.println("Failed to read from DHT sensor!");
    return;
  }

  // Calculate a simple checksum.
  int checksum = (int)t + (int)h;

  if (attack_mode) {
    // --- ATTACK MODE ---
    // Print a status message and send a FAKE temperature.
    Serial.println("--- ATTACK MODE ACTIVE ---");
    float fake_t = 25.0;
    Serial.print("T:"); Serial.print(fake_t);
    Serial.print(",H:"); Serial.print(h);
    Serial.print(",C:"); Serial.println(checksum);
  } else {
    // --- NORMAL MODE ---
    // Print a status message and send the REAL temperature.
    Serial.println("--- NORMAL MODE ---");
    Serial.print("T:"); Serial.print(t);
    Serial.print(",H:"); Serial.print(h);
    Serial.print(",C:"); Serial.println(checksum);
  }
}