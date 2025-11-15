# Sentence Recognition - Setup Guide

## Overview

Your EchoSignRealtime project now supports TWO recognition modes:

1. **Gesture Mode** (existing): Instant recognition of hand gestures (One, Two, Three, etc.)
2. **Sentence Mode** (NEW): Recognition of complete sign language sentences/phrases using 4-second windows

## How It Works

### Gesture Mode
- Continuous real-time recognition
- Recognizes single gestures instantly
- Uses your existing KNN model

### Sentence Mode
- Activated by pressing a button
- Records 4 seconds of continuous sensor data
- Recognizes complete sentences like "how are you", "i eat rice"
- Uses a separate KNN model trained on 4-second windows

## Hardware Requirements

### New Component Needed
- **Push button** connected to GPIO 5 (PIN_SENTENCE_BUTTON)
- Wire one side to GPIO 5, other side to GND
- Internal pullup resistor is enabled in software

## Setup Steps

### Step 1: Collect Sentence Data

Run the sentence data collection tool:

```bash
python tools/collect_sentence_data.py --port COM15
```

**For each sentence you want to recognize:**

1. Enter the sentence name (e.g., "how_are_you", "i_eat_rice", "thank_you")
2. Watch the 3-second preview to see sensor data
3. Press ENTER to start recording
4. After 3-2-1 countdown, perform the COMPLETE sentence gesture
5. Recording automatically stops after 4 seconds
6. Repeat 5-10 times per sentence for best accuracy

**Recommended first 5 sentences:**
- how_are_you
- i_eat_rice
- thank_you
- good_morning
- nice_to_meet_you

### Step 2: Train Sentence Model

After collecting data, train the KNN model:

```bash
python tools/train_sentence_knn.py
```

This will generate:
- `data/sentence_dataset.csv` - Combined dataset
- `src/sentence_knn_model.h` - KNN model
- `src/sentence_scaler_params.h` - Feature scaling parameters
- `src/sentence_label_names.h` - Sentence labels

### Step 3: Configure Firmware

Edit `src/main.cpp` and set the prediction mode:

```cpp
// PREDICTION_MODE options:
// 0 = GESTURE MODE only
// 1 = SENTENCE MODE only  
// 2 = AUTO MODE (gesture + sentence button) <- RECOMMENDED
#define PREDICTION_MODE 2
```

**Recommended: Use PREDICTION_MODE 2 (AUTO MODE)**
- Continuous gesture recognition by default
- Press button to switch to sentence mode for 3 seconds
- Best of both worlds!

### Step 4: Compile and Upload

```bash
pio run --target upload
```

Or use PlatformIO's upload button in VS Code.

### Step 5: Test

1. Open serial monitor (115200 baud)
2. You should see: "Prediction: AUTO MODE (gesture + sentence button)"
3. Test gesture mode - perform hand gestures, see instant recognition
4. Test sentence mode:
   - Press the button (connected to GPIO 5)
   - You'll hear 3 beeps (LED turns on)
   - Perform your sentence gesture
   - After 4 seconds, you'll hear 1 long beep (LED turns off)
   - See sentence recognized in serial output

## Data Collection Tips

### For Best Results

1. **Consistency**: Perform each sentence the same way every time
2. **Timing**: Use the full 3 seconds - don't rush
3. **Natural flow**: Sign language sentences are continuous, not individual letters
4. **Repetition**: Record 5-10 samples per sentence minimum
5. **Variety**: Try slight variations (speed, size) to improve robustness

### Example Sentence Flow

For "how are you":
- **0.0-1.5s**: Sign "HOW" gesture
- **1.5-2.5s**: Sign "ARE" gesture  
- **2.5-4.0s**: Sign "YOU" gesture

Keep movements smooth and continuous throughout the 4 seconds.

## JSON Output Format

### Gesture Mode Output
```json
{
  "mode": "gesture",
  "label": "One",
  "meanD": 12.34,
  "gdp": 156.7,
  "f1": 0.45, "f2": 0.67, ...
}
```

### Sentence Recording Progress
```json
{
  "mode": "sentence",
  "recording": true,
  "progress": 0.65
}
```

### Sentence Prediction Output
```json
{
  "mode": "sentence",
  "recording": false,
  "sentence": "how_are_you",
  "confidence": 0.892,
  "meanD": 45.67
}
```

## Troubleshooting

### "Sentence mode requested but model files missing!"
- Run: `python tools/train_sentence_knn.py`
- Make sure sentence data files exist in `data/` folder
- Files should be named: `sentence_raw_<label>_01.txt`

### Button doesn't trigger sentence mode
- Check GPIO 5 connection (button between GPIO 5 and GND)
- Verify `PREDICTION_MODE` is set to 1 or 2
- Check serial output for button press events

### Poor sentence recognition accuracy
- Collect more samples (aim for 10+ per sentence)
- Be more consistent in how you perform each sentence
- Make sure you use the full 3 seconds
- Check sensor data quality during collection

### Not enough samples collected
- ESP32 might not be sending data fast enough
- Check `RUN_MODE = 0` for data collection
- Verify 115200 baud rate
- Try increasing timeout in collection script

## Sample Rate Details

- **Target**: 20 Hz (50ms between samples)
- **4-second window**: 80 samples
- **Features per sample**: 12 (f1-f5, gdp, ax-az, gx-gz)
- **Total features**: 960 (80 samples × 12 features)

## Advanced Configuration

### Change Recording Duration

Edit `src/sentence_predictor.h`:
```cpp
#define SENTENCE_WINDOW_DURATION_MS 4000  // Change to 5000 for 5 seconds
#define SENTENCE_SAMPLE_RATE_HZ 20
#define SENTENCE_SAMPLES_PER_WINDOW 80    // Update accordingly (5000ms * 20Hz / 1000 = 100)
```

### Change KNN Parameters

Edit `tools/train_sentence_knn.py`:
```python
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9],  # Add more K values
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
```

## Next Steps

1. **Expand vocabulary**: Add more sentences as needed
2. **Web UI integration**: Update web interface to show sentence mode
3. **Improve accuracy**: Collect more diverse training samples
4. **Add more features**: Consider adding sentence chaining or grammar

## Files Overview

### New Files Created
- `tools/collect_sentence_data.py` - Data collection tool
- `tools/train_sentence_knn.py` - Model training script
- `src/sentence_predictor.h` - Sentence prediction logic
- `src/sentence_knn_model.h` - Generated KNN model (after training)
- `src/sentence_scaler_params.h` - Generated scaling params (after training)
- `src/sentence_label_names.h` - Generated label names (after training)

### Modified Files
- `src/main.cpp` - Added sentence mode support
- `src/calib.h` - Added button pin definition

### Data Files
- `data/sentence_raw_*.txt` - Raw sentence recordings
- `data/sentence_dataset.csv` - Combined training dataset

## Summary

You now have a **hybrid system** that can:
- ✅ Recognize instant gestures (existing feature)
- ✅ Recognize complete sentences via button trigger (new feature)
- ✅ Seamlessly switch between modes
- ✅ Use separate optimized models for each mode

Start with 5 sentences, test the system, then expand your vocabulary!
