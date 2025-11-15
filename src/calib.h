#pragma once
#include <Arduino.h>

// ---------------- Pin mapping (ESP32 ADC1 only) ----------------
// Thumb  -> GPIO 36
// Index  -> GPIO 39
// Middle -> GPIO 34
// Ring   -> GPIO 35
// Pinky  -> GPIO 32
static const int PIN_FLEX[5] = {
    36, // Thumb
    39, // Index
    34, // Middle
    35, // Ring
    32  // Pinky
};

// I2C pins for MPU-6050 (GY-521)
static const int PIN_I2C_SDA = 21;
static const int PIN_I2C_SCL = 22;

// LED + buzzer pins
// LED: use built-in GPIO 2 (you can change if needed)
// BUZZER: connect a small piezo or active buzzer here via resistor
static const int PIN_LED    = 2;
static const int PIN_BUZZER = 4;

// Button pin for sentence recording trigger
// Connect button between this pin and GND (internal pullup enabled)
static const int PIN_SENTENCE_BUTTON = 5;

// MPU-6050 I2C address
#ifndef MPU6050_ADDR
#define MPU6050_ADDR 0x68
#endif

// --------- Flex calibration (raw ADC min / max) ---------
// Replace these with the values printed by extract_calib_from_dump.py
static const int FLEX_MIN[5] = {2000, 2000, 2000, 2000, 2000};
static const int FLEX_MAX[5] = {3500, 3500, 3500, 3500, 3500};

// Helper to normalize a raw flex reading into [0,1] using FLEX_MIN/MAX.
inline float normalizeFlexRaw(int idx, int raw) {
  int mn = FLEX_MIN[idx];
  int mx = FLEX_MAX[idx];
  if (mx <= mn) return 0.0f;
  float v = (float)(raw - mn) / (float)(mx - mn);
  if (v < 0.0f) v = 0.0f;
  if (v > 1.0f) v = 1.0f;
  return v;
}
