#pragma once
#include <Arduino.h>
#include "sentence_scaler_params.h"
#include "sentence_knn_model.h"

/**
 * Sentence Predictor
 * 
 * Collects 3-second windows of sensor data and predicts complete sentences.
 * Uses a circular buffer to continuously store recent sensor readings.
 */

// Configuration
#define SENTENCE_WINDOW_DURATION_MS 4000  // 4 seconds
#define SENTENCE_SAMPLE_RATE_HZ 20        // 20 Hz
#define SENTENCE_SAMPLES_PER_WINDOW 80    // 4 sec * 20 Hz
#define SENTENCE_SAMPLE_INTERVAL_MS (1000 / SENTENCE_SAMPLE_RATE_HZ)  // 50ms

// Circular buffer for sensor samples
struct SensorSample {
  float f1, f2, f3, f4, f5;  // Flex sensors (normalized 0-1)
  float gdp;                  // Gyro magnitude
  float ax, ay, az;           // Accelerometer (g)
  float gx, gy, gz;           // Gyroscope (deg/s)
};

class SentencePredictor {
private:
  SensorSample buffer[SENTENCE_SAMPLES_PER_WINDOW];
  uint8_t bufferIndex;
  uint32_t lastSampleTime;
  bool bufferFilled;
  bool isRecording;
  uint32_t recordingStartTime;

public:
  SentencePredictor() 
    : bufferIndex(0), lastSampleTime(0), bufferFilled(false), 
      isRecording(false), recordingStartTime(0) 
  {}

  // Start recording a 3-second window
  void startRecording() {
    isRecording = true;
    recordingStartTime = millis();
    bufferIndex = 0;
    bufferFilled = false;
    memset(buffer, 0, sizeof(buffer));
  }

  // Check if currently recording
  bool recording() const {
    return isRecording;
  }

  // Get recording progress (0.0 to 1.0)
  float getRecordingProgress() const {
    if (!isRecording) return 0.0f;
    uint32_t elapsed = millis() - recordingStartTime;
    return min(1.0f, (float)elapsed / (float)SENTENCE_WINDOW_DURATION_MS);
  }

  // Get remaining time in milliseconds
  uint32_t getRemainingTime() const {
    if (!isRecording) return 0;
    uint32_t elapsed = millis() - recordingStartTime;
    if (elapsed >= SENTENCE_WINDOW_DURATION_MS) return 0;
    return SENTENCE_WINDOW_DURATION_MS - elapsed;
  }

  // Add a sensor sample to the buffer
  // Returns true if window is complete and ready for prediction
  bool addSample(float f1, float f2, float f3, float f4, float f5,
                 float gdp, float ax, float ay, float az,
                 float gx, float gy, float gz) {
    
    if (!isRecording) return false;

    uint32_t now = millis();
    
    // Check if enough time has passed since last sample
    if (now - lastSampleTime < SENTENCE_SAMPLE_INTERVAL_MS) {
      return false;
    }
    
    // Store sample
    buffer[bufferIndex].f1 = f1;
    buffer[bufferIndex].f2 = f2;
    buffer[bufferIndex].f3 = f3;
    buffer[bufferIndex].f4 = f4;
    buffer[bufferIndex].f5 = f5;
    buffer[bufferIndex].gdp = gdp;
    buffer[bufferIndex].ax = ax;
    buffer[bufferIndex].ay = ay;
    buffer[bufferIndex].az = az;
    buffer[bufferIndex].gx = gx;
    buffer[bufferIndex].gy = gy;
    buffer[bufferIndex].gz = gz;
    
    lastSampleTime = now;
    bufferIndex++;
    
    // Check if window is complete
    if (bufferIndex >= SENTENCE_SAMPLES_PER_WINDOW) {
      bufferFilled = true;
      isRecording = false;  // Stop recording
      return true;  // Ready for prediction
    }
    
    // Check if 3 seconds elapsed (fallback)
    if (now - recordingStartTime >= SENTENCE_WINDOW_DURATION_MS) {
      bufferFilled = true;
      isRecording = false;
      return true;
    }
    
    return false;
  }

  // Predict sentence from current buffer
  // Returns label index and confidence score via meanDistance
  uint8_t predict(float* meanDistance) {
    if (!bufferFilled) {
      *meanDistance = 999999.0f;
      return 0;
    }

    // Build feature vector (flatten buffer)
    float features[SENTENCE_NUM_FEATURES];
    int idx = 0;
    
    for (int t = 0; t < SENTENCE_SAMPLES_PER_WINDOW; t++) {
      features[idx++] = buffer[t].f1;
      features[idx++] = buffer[t].f2;
      features[idx++] = buffer[t].f3;
      features[idx++] = buffer[t].f4;
      features[idx++] = buffer[t].f5;
      features[idx++] = buffer[t].gdp;
      features[idx++] = buffer[t].ax;
      features[idx++] = buffer[t].ay;
      features[idx++] = buffer[t].az;
      features[idx++] = buffer[t].gx;
      features[idx++] = buffer[t].gy;
      features[idx++] = buffer[t].gz;
    }

    // Standardize features
    standardizeSentenceFeatures(features);

    // KNN prediction (Euclidean distance)
    return predictSentenceKNN(features, meanDistance);
  }

  // Reset buffer
  void reset() {
    bufferIndex = 0;
    bufferFilled = false;
    isRecording = false;
  }

private:
  // KNN prediction using Euclidean distance
  uint8_t predictSentenceKNN(const float* query, float* outMeanDist) {
    const int K = SENTENCE_KNN_N_NEIGHBORS;
    const int N = SENTENCE_KNN_N_SAMPLES;
    const int D = SENTENCE_KNN_N_FEATURES;

    // Arrays to store K nearest neighbors
    float nearestDist[K];
    uint8_t nearestLabels[K];
    
    // Initialize with large distances
    for (int i = 0; i < K; i++) {
      nearestDist[i] = 1e9;
      nearestLabels[i] = 0;
    }

    // Find K nearest neighbors
    for (int i = 0; i < N; i++) {
      // Calculate Euclidean distance
      float dist = 0.0f;
      
      for (int d = 0; d < D; d++) {
        // Read from PROGMEM (training data is stored in flash)
        float train_val = pgm_read_float(&SENTENCE_TRAINING_DATA[i * D + d]);
        float diff = query[d] - train_val;
        dist += diff * diff;
      }
      
      dist = sqrtf(dist);

      // Insert into K nearest if closer than current worst
      if (dist < nearestDist[K-1]) {
        // Find insertion position
        int pos = K - 1;
        while (pos > 0 && dist < nearestDist[pos - 1]) {
          pos--;
        }
        
        // Shift and insert
        for (int j = K - 1; j > pos; j--) {
          nearestDist[j] = nearestDist[j - 1];
          nearestLabels[j] = nearestLabels[j - 1];
        }
        
        nearestDist[pos] = dist;
        nearestLabels[pos] = SENTENCE_TRAINING_LABELS[i];
      }
    }

    // Vote among K nearest neighbors
    uint8_t votes[SENTENCE_NUM_CLASSES] = {0};
    for (int i = 0; i < K; i++) {
      votes[nearestLabels[i]]++;
    }

    // Find label with most votes
    uint8_t bestLabel = 0;
    uint8_t maxVotes = 0;
    for (int i = 0; i < SENTENCE_NUM_CLASSES; i++) {
      if (votes[i] > maxVotes) {
        maxVotes = votes[i];
        bestLabel = i;
      }
    }

    // Calculate mean distance of K neighbors
    float sumDist = 0.0f;
    for (int i = 0; i < K; i++) {
      sumDist += nearestDist[i];
    }
    *outMeanDist = sumDist / K;

    return bestLabel;
  }
};
