#pragma once
#include <Arduino.h>
#include "sentence_label_names.h"
#include "sentence_scaler_params.h"
#include "sentence_knn_model_q.h" // Use quantized INT8 model for memory efficiency

// Note: Remove obsolete hardcoded sample/count defines; rely on header values
// SENTENCE_KNN_N_NEIGHBORS, SENTENCE_KNN_N_SAMPLES, SENTENCE_KNN_N_FEATURES now come from sentence_knn_model.h

/**
 * Sentence Predictor
 * 
 * Collects 4-second windows of sensor data and predicts complete sentences.
 * Uses a circular buffer to continuously store recent sensor readings.
 */

// Configuration
#define SENTENCE_WINDOW_DURATION_MS 4000  // 4 seconds recording window
#define SENTENCE_SAMPLE_RATE_HZ 20        // 20 Hz
#define SENTENCE_SAMPLES_PER_WINDOW 80    // 4 sec * 20 Hz = 80 samples
#define SENTENCE_SAMPLES_FOR_PREDICTION 80 // Use all 80 samples (4 sec) - matches training data!
#define SENTENCE_SAMPLE_INTERVAL_MS (1000 / SENTENCE_SAMPLE_RATE_HZ)  // 50ms

// Circular buffer for sensor samples
struct SensorSample {
  float f1, f2, f3, f4, f5;  // Flex sensors (RAW values, match training logs)
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
  int restLabelIndex = -1;  // resolved lazily from label names

public:
  SentencePredictor() 
    : bufferIndex(0), lastSampleTime(0), bufferFilled(false), 
      isRecording(false), recordingStartTime(0) 
  {}

  // Start recording a 4-second window
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
    
    // If we collected fewer than target samples (due to timing), resample to 80 via linear interpolation
    int collected = min((int)SENTENCE_SAMPLES_PER_WINDOW, (int)bufferIndex);
    if (collected < (int)SENTENCE_SAMPLES_FOR_PREDICTION) {
      // Precompute mapping from target index to source fractional index
      for (int t = 0; t < SENTENCE_SAMPLES_FOR_PREDICTION; t++) {
        float srcPos = (collected > 1)
          ? (float)t * (float)(collected - 1) / (float)(SENTENCE_SAMPLES_FOR_PREDICTION - 1)
          : 0.0f;
        int i0 = (int)floorf(srcPos);
        int i1 = (int)ceilf(srcPos);
        float w = srcPos - (float)i0;

        const SensorSample& s0 = buffer[i0];
        const SensorSample& s1 = buffer[i1];

        auto lerp = [w](float a, float b) { return a + w * (b - a); };

        features[idx++] = lerp(s0.f1, s1.f1);
        features[idx++] = lerp(s0.f2, s1.f2);
        features[idx++] = lerp(s0.f3, s1.f3);
        features[idx++] = lerp(s0.f4, s1.f4);
        features[idx++] = lerp(s0.f5, s1.f5);
        features[idx++] = lerp(s0.gdp, s1.gdp);
        features[idx++] = lerp(s0.ax, s1.ax);
        features[idx++] = lerp(s0.ay, s1.ay);
        features[idx++] = lerp(s0.az, s1.az);
        features[idx++] = lerp(s0.gx, s1.gx);
        features[idx++] = lerp(s0.gy, s1.gy);
        features[idx++] = lerp(s0.gz, s1.gz);
      }
    } else {
      // Exact 80 samples collected; copy directly
      for (int t = 0; t < SENTENCE_SAMPLES_FOR_PREDICTION; t++) {
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
    }

    // Standardize features
    standardizeSentenceFeatures(features);

    // KNN prediction (Manhattan distance with distance-weighted voting)
    uint8_t rawPred = predictSentenceKNN(features, meanDistance);

    // Resolve 'Rest' index lazily (or 'Unknown' if present)
    if (restLabelIndex < 0) {
      for (int i = 0; i < SENTENCE_NUM_CLASSES; ++i) {
        const char* nm = sentence_label_names[i];
        if ((nm && strcmp(nm, "Rest") == 0) || (nm && strcmp(nm, "Unknown") == 0)) {
          restLabelIndex = i;
          break;
        }
      }
      if (restLabelIndex < 0) restLabelIndex = 0; // fallback
    }

    // TEMPORARY: Raise rejection threshold until we collect int8 distance stats.
    // Previous 12000 value was for float space; int8 Manhattan distances are larger.
    const float REJECTION_MEAN_DISTANCE = 200000.0f; // debug high threshold to avoid constant Rest override

    uint8_t finalPred = rawPred;
    bool overridden = false;
    if (*meanDistance > REJECTION_MEAN_DISTANCE && restLabelIndex >= 0) {
      finalPred = (uint8_t)restLabelIndex;
      overridden = true;
    }

#ifdef SENTENCE_DEBUG
  Serial.print("{\"debug\":\"sentence_pred\",\"rawPred\":");
  Serial.print(rawPred);
  Serial.print(",\"meanD\":");
  Serial.print(*meanDistance, 2);
  Serial.print(",\"restIdx\":");
  Serial.print(restLabelIndex);
  Serial.print(",\"override\":");
  Serial.print(overridden ? 1 : 0);
  Serial.println("}");
#endif

    return finalPred;
  }

  // Reset buffer
  void reset() {
    bufferIndex = 0;
    bufferFilled = false;
    isRecording = false;
  }

private:
  // KNN prediction using Manhattan distance (L1) with distance-weighted voting
  uint8_t predictSentenceKNN(const float* queryStd, float* outMeanDist) {
    // Quantize query (already standardized) to int8
    static int8_t qQuery[SENTENCE_KNN_Q_N_FEATURES];
    quantizeSentenceFeatures(queryStd, qQuery);

    const int K = SENTENCE_KNN_Q_N_NEIGHBORS;
    const int N = SENTENCE_KNN_Q_N_SAMPLES;
    const int D = SENTENCE_KNN_Q_N_FEATURES;

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
      // Calculate Manhattan distance (must match training)
      float dist = 0.0f;
      
      for (int d = 0; d < D; d++) {
        int8_t train_q = pgm_read_byte(&SENTENCE_TRAINING_DATA_Q[i * D + d]);
        int diff = (int)qQuery[d] - (int)train_q;
        dist += (diff >= 0 ? diff : -diff); // Manhattan distance in quantized space
      }

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
        nearestLabels[pos] = pgm_read_byte(&SENTENCE_TRAINING_LABELS_Q[i]);
      }
    }

    // Distance-weighted voting (matches training weights='distance')
    const int numClasses = SENTENCE_NUM_CLASSES;
    float weightSums[numClasses];
    for (int i = 0; i < numClasses; i++) weightSums[i] = 0.0f;
    for (int i = 0; i < K; i++) {
      float w = 1.0f / (nearestDist[i] + 1e-6f); // inverse distance weighting
      weightSums[nearestLabels[i]] += w;
    }

    // Pick label with largest accumulated weight
    uint8_t bestLabel = 0;
    float bestWeight = -1.0f;
    for (int i = 0; i < numClasses; i++) {
      if (weightSums[i] > bestWeight) {
        bestWeight = weightSums[i];
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
