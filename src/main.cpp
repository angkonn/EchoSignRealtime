#include <Arduino.h>
#include "Calib.h"
#include "predictor.h"

// Check if sentence model files exist
#if __has_include("sentence_knn_model.h")
  #define SENTENCE_MODE_AVAILABLE 1
  #include "sentence_predictor.h"
  #include "sentence_label_names.h"
#else
  #define SENTENCE_MODE_AVAILABLE 0
#endif

// -------------------- CONFIG --------------------

// 0 = DATA COLLECTION (raw log for Python tools)
// 1 = REAL-TIME PREDICTION (uses KNN model)
#define RUN_MODE 1

// Prediction modes (only used when RUN_MODE == 1)
// 0 = GESTURE MODE (instant gestures)
// 1 = SENTENCE MODE (3-second windows)
// 2 = AUTO MODE (gesture by default, sentence when button pressed)
#define PREDICTION_MODE 2

GlovePredictor predictor;

#if SENTENCE_MODE_AVAILABLE && RUN_MODE == 1
SentencePredictor sentencePredictor;
#endif

// Simple timer for collection loop
uint32_t lastPrintMs = 0;
const uint32_t COLLECT_PERIOD_MS = 50;   // ~20 Hz

// Recording state (from PC via serial command)
bool gRecordingActive = false;

// Sentence mode state
bool sentenceModeActive = false;
bool lastButtonState = HIGH;
uint32_t lastDebounceTime = 0;
const uint32_t DEBOUNCE_DELAY_MS = 50;

// --------------- LED / BUZZER HELPERS ---------------

static void beep(uint16_t onMs = 80, uint8_t times = 1, uint16_t offMs = 60) {
  for (uint8_t i = 0; i < times; ++i) {
    digitalWrite(PIN_BUZZER, HIGH);
    delay(onMs);
    digitalWrite(PIN_BUZZER, LOW);
    if (i + 1 < times) delay(offMs);
  }
}

static void signalRecordingStart() {
  digitalWrite(PIN_LED, HIGH);
  beep(80, 2, 80);  // double beep
}

static void signalRecordingStop() {
  digitalWrite(PIN_LED, LOW);
  beep(60, 1, 0);   // single short beep
}

static void signalSentenceStart() {
  digitalWrite(PIN_LED, HIGH);
  beep(100, 3, 50);  // triple beep for sentence mode
}

static void signalSentenceComplete() {
  digitalWrite(PIN_LED, LOW);
  beep(150, 1, 0);   // longer beep when complete
}

static void handleSerialCommands() {
  // We expect single-character commands from PC:
  // 'S' = start recording, 'E' = end recording
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == 'S') {
      gRecordingActive = true;
      signalRecordingStart();
    } else if (c == 'E') {
      gRecordingActive = false;
      signalRecordingStop();
    }
  }
}

// ------------------------------------------------------

void setup() {
  Serial.begin(115200);
  delay(2000); // Give serial monitor time

  // LED + buzzer pins
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  digitalWrite(PIN_LED, LOW);
  digitalWrite(PIN_BUZZER, LOW);

  // Sentence button (with internal pullup)
  pinMode(PIN_SENTENCE_BUTTON, INPUT_PULLUP);

  if (!predictor.begin()) {
    Serial.println("MPU6050 init FAILED");
  } else {
    Serial.println("EchoSignRealtime started.");
#if RUN_MODE == 0
    Serial.println("Mode: DATA COLLECTION");
#else
    Serial.println("Mode: REAL-TIME PREDICTION");
  #if PREDICTION_MODE == 0
    Serial.println("Prediction: GESTURE MODE");
  #elif PREDICTION_MODE == 1
    #if SENTENCE_MODE_AVAILABLE
      Serial.println("Prediction: SENTENCE MODE");
    #else
      Serial.println("ERROR: Sentence mode requested but model files missing!");
      Serial.println("Run: python tools/train_sentence_knn.py");
    #endif
  #elif PREDICTION_MODE == 2
    #if SENTENCE_MODE_AVAILABLE
      Serial.println("Prediction: AUTO MODE (gesture + sentence button)");
    #else
      Serial.println("Prediction: GESTURE MODE (sentence model not available)");
    #endif
  #endif
#endif
    // Power-on beep
    beep(60, 1, 0);
  }
}

void loop() {
  // Always process incoming serial commands
  handleSerialCommands();

#if RUN_MODE == 1 && SENTENCE_MODE_AVAILABLE && (PREDICTION_MODE == 1 || PREDICTION_MODE == 2)
  // Handle sentence button (with debouncing)
  int buttonReading = digitalRead(PIN_SENTENCE_BUTTON);
  
  if (buttonReading != lastButtonState) {
    lastDebounceTime = millis();
  }
  
  if ((millis() - lastDebounceTime) > DEBOUNCE_DELAY_MS) {
    // Button is stable
    if (buttonReading == LOW && lastButtonState == HIGH) {
      // Button pressed (LOW because pullup)
      if (!sentencePredictor.recording()) {
        // Start sentence recording
        sentenceModeActive = true;
        sentencePredictor.startRecording();
        signalSentenceStart();
        
        Serial.println("{\"event\":\"sentence_start\"}");
      }
    }
  }
  lastButtonState = buttonReading;
#endif

#if RUN_MODE == 0
  // --------- DATA COLLECTION MODE ---------
  uint32_t now = millis();
  if (now - lastPrintMs < COLLECT_PERIOD_MS) return;
  lastPrintMs = now;

  int flex[5];
  int16_t ax, ay, az, gx, gy, gz;
  predictor.readRawFrame(flex, ax, ay, az, gx, gy, gz);

  float fgx = (float)gx;
  float fgy = (float)gy;
  float fgz = (float)gz;
  float gdp = sqrtf(fgx * fgx + fgy * fgy + fgz * fgz);

  // EXACT format expected by Python tools:
  // FLEX: f1 f2 f3 f4 f5 | ACC: ax ay az | GYRO: gx gy gz | GDP=val
  Serial.print("FLEX: ");
  Serial.print(flex[0]); Serial.print(' ');
  Serial.print(flex[1]); Serial.print(' ');
  Serial.print(flex[2]); Serial.print(' ');
  Serial.print(flex[3]); Serial.print(' ');
  Serial.print(flex[4]);

  Serial.print(" | ACC: ");
  Serial.print(ax); Serial.print(' ');
  Serial.print(ay); Serial.print(' ');
  Serial.print(az);

  Serial.print(" | GYRO: ");
  Serial.print(gx); Serial.print(' ');
  Serial.print(gy); Serial.print(' ');
  Serial.print(gz);

  Serial.print(" | GDP=");
  Serial.println(gdp, 3);   // 3 decimal places

#else
  // --------- REAL-TIME PREDICTION MODE ---------
  
  // Read raw sensor data
  int flex[5];
  int16_t ax, ay, az, gx, gy, gz;
  predictor.readRawFrame(flex, ax, ay, az, gx, gy, gz);
  
  // Calculate GDP (gyro magnitude)
  float fgx = (float)gx;
  float fgy = (float)gy;
  float fgz = (float)gz;
  float gdp = sqrtf(fgx * fgx + fgy * fgy + fgz * fgz);
  
  // Normalize flex values (0-1 range based on calibration)
  float f1 = (float)(flex[0] - FLEX_MIN[0]) / (float)(FLEX_MAX[0] - FLEX_MIN[0]);
  float f2 = (float)(flex[1] - FLEX_MIN[1]) / (float)(FLEX_MAX[1] - FLEX_MIN[1]);
  float f3 = (float)(flex[2] - FLEX_MIN[2]) / (float)(FLEX_MAX[2] - FLEX_MIN[2]);
  float f4 = (float)(flex[3] - FLEX_MIN[3]) / (float)(FLEX_MAX[3] - FLEX_MIN[3]);
  float f5 = (float)(flex[4] - FLEX_MIN[4]) / (float)(FLEX_MAX[4] - FLEX_MIN[4]);
  
  // Clamp to 0-1
  f1 = constrain(f1, 0.0f, 1.0f);
  f2 = constrain(f2, 0.0f, 1.0f);
  f3 = constrain(f3, 0.0f, 1.0f);
  f4 = constrain(f4, 0.0f, 1.0f);
  f5 = constrain(f5, 0.0f, 1.0f);
  
  // Convert accel to g (assuming 16-bit signed, ±2g range)
  float fax = ax / 16384.0f;
  float fay = ay / 16384.0f;
  float faz = az / 16384.0f;
  
  // Convert gyro to deg/s (assuming ±250 deg/s range)
  float fgxDeg = gx / 131.0f;
  float fgyDeg = gy / 131.0f;
  float fgzDeg = gz / 131.0f;

#if SENTENCE_MODE_AVAILABLE && (PREDICTION_MODE == 1 || PREDICTION_MODE == 2)
  // Check if sentence mode is active
  if (sentenceModeActive && sentencePredictor.recording()) {
    // Add sample to sentence buffer
    bool windowComplete = sentencePredictor.addSample(
      f1, f2, f3, f4, f5, gdp, fax, fay, faz, fgxDeg, fgyDeg, fgzDeg
    );
    
    // Send progress update
    float progress = sentencePredictor.getRecordingProgress();
    Serial.print("{\"mode\":\"sentence\",\"recording\":true,\"progress\":");
    Serial.print(progress, 2);
    Serial.println("}");
    
    if (windowComplete) {
      // Window complete, predict sentence
      signalSentenceComplete();
      
      float meanDist = 0.0f;
      uint8_t labelIdx = sentencePredictor.predict(&meanDist);
      
      const char* sentenceName = "unknown";
      if (labelIdx < SENTENCE_NUM_CLASSES) {
        sentenceName = sentence_label_names[labelIdx];
      }
      
      // Calculate confidence (inverse of distance)
      float confidence = 1.0f / (1.0f + meanDist);
      
      // Output sentence prediction
      Serial.print("{\"mode\":\"sentence\",\"recording\":false,\"sentence\":\"");
      Serial.print(sentenceName);
      Serial.print("\",\"confidence\":");
      Serial.print(confidence, 3);
      Serial.print(",\"meanD\":");
      Serial.print(meanDist, 2);
      Serial.println("}");
      
      sentenceModeActive = false;
      sentencePredictor.reset();
    }
    
    delay(10);  // Small delay for sampling rate control
    return;  // Skip gesture prediction while in sentence mode
  }
#endif

#if PREDICTION_MODE == 0 || PREDICTION_MODE == 2 || !SENTENCE_MODE_AVAILABLE
  // Regular gesture prediction
  float bestDist = 0.0f;
  uint8_t labelIdx = predictor.predictGesture(&bestDist, 250, 10);
  
  const char* gestureName = "unknown";
  if (labelIdx < NUM_CLASSES) {
    gestureName = label_names[labelIdx];
  }

  // Output JSON format for web UI
  Serial.print("{");
  Serial.print("\"mode\":\"gesture\",");
  Serial.print("\"label\":\""); Serial.print(gestureName); Serial.print("\",");
  Serial.print("\"meanD\":" ); Serial.print(bestDist, 2); Serial.print(",");
  Serial.print("\"gdp\":" ); Serial.print(gdp, 1); Serial.print(",");
  Serial.print("\"f1\":" ); Serial.print(f1, 2); Serial.print(",");
  Serial.print("\"f2\":" ); Serial.print(f2, 2); Serial.print(",");
  Serial.print("\"f3\":" ); Serial.print(f3, 2); Serial.print(",");
  Serial.print("\"f4\":" ); Serial.print(f4, 2); Serial.print(",");
  Serial.print("\"f5\":" ); Serial.print(f5, 2); Serial.print(",");
  Serial.print("\"ax\":" ); Serial.print(fax, 2); Serial.print(",");
  Serial.print("\"ay\":" ); Serial.print(fay, 2); Serial.print(",");
  Serial.print("\"az\":" ); Serial.print(faz, 2); Serial.print(",");
  Serial.print("\"gx\":" ); Serial.print(fgxDeg, 1); Serial.print(",");
  Serial.print("\"gy\":" ); Serial.print(fgyDeg, 1); Serial.print(",");
  Serial.print("\"gz\":" ); Serial.print(fgzDeg, 1);
  Serial.println("}");
#endif

  delay(100);
#endif
}
