# EchoSignRealtime

<img src="data/WhatsApp%20Image%202025-11-29%20at%203.47.41%20PM.jpeg" alt="EchoSign Glove Front" width="600" />

Real-time gesture and sentence recognition for a wearable sign glove. Firmware runs a lightweight KNN model (float and quantized int8) and streams predictions to a Web UI with optional text-to-speech.

## Project Overview

This project implements real-time recognition using flex sensors and IMU (accelerometer/gyroscope). A KNN (k-Nearest Neighbors) classifier operates on standardized feature windows. Two modes are supported:
- Gesture mode: continuous single-frame classifications
- Sentence mode: 4-second windows aggregated, standardized, then classified

## Directory Structure

- **src/**: Main firmware sources
  - `main.cpp`: Main application entry point
  - `predictor.h`: Gesture prediction logic
  - `knn_runtime.h`: KNN runtime (distance, voting)
  - `glove_knn_model.h`: Gesture KNN model (float)
  - `scaler_params.h`: Gesture scaler (means/scales)
  - `label_names.h`: Gesture labels
  - `calib.h` and `include/Calib.h`: Calibration
  - `sentence_predictor.h`: Sentence prediction pipeline (4-second windows)
  - `sentence_knn_model.cpp` / `sentence_knn_model.h`: Sentence KNN (float)
  - `sentence_knn_model_q.h`: Sentence KNN (quantized int8 data + scales)
  - `sentence_scaler_params.h`: Sentence scaler (means/scales)
  - `sentence_label_names.h`: Sentence labels

- **data/**: Dataset and Web UI assets
  - `dataset.csv`: Combined gesture training dataset
  - `raw_*.txt`: Raw gesture sensor logs
  - `sentence_raw_*.txt`: Raw sentence windows (4-second) logs
  - `index.html`, `main.js`, `style.css`: Web UI (3D hand, status, TTS)

- **tools/**: Python utilities
  - `collect_gesture_data.py`: Collect gesture samples
  - `collect_sentence_data.py`: Collect sentence windows (4s recordings)
  - `train_knn.py`: Train gesture KNN
  - `train_sentence_knn.py`: Train sentence KNN (float + int8 export)
  - `parse_and_train.py`: Parse and train convenience script
  - `merge_logs.py`: Merge text logs
  - `extract_calib_from_dump.py`: Extract calibration
  - `web_ui.py`: Flask server + serial bridge

- **include/**: Header files
- **lib/**: Library files
- **test/**: Test files

## Getting Started

### Prerequisites

- Python 3.10+
- PlatformIO (ESP32/Arduino toolchain)
- Recommended: VS Code + PlatformIO extension
- Python packages: `flask`, `numpy`, `scikit-learn`, `pandas` (install as needed)

### Data Collection

Collect gesture samples:

```powershell
python tools/collect_gesture_data.py
```

Collect sentence samples (4-second windows). Add 10 windows per new sentence, and at least 20 windows for `Rest` to improve rejection:

```powershell
python tools/collect_sentence_data.py
```

### Model Training

Train gesture KNN from `data/dataset.csv`:

```powershell
python tools/train_knn.py data/dataset.csv
```

Train sentence KNN (exports float and quantized int8 headers used by firmware):

```powershell
python tools/train_sentence_knn.py
```

Parse and train in one step (optional):

```powershell
python tools/parse_and_train.py
```

### Web UI

Start the web interface:


## Hardware

Photos of the current glove build (flex sensors + ESP32):

<img src="data/WhatsApp%20Image%202025-11-29%20at%203.47.41%20PM.jpeg" alt="EchoSign Glove Front" width="600" />

<img src="data/WhatsApp%20Image%202025-11-29%20at%203.47.42%20PM.jpeg" alt="EchoSign Glove + UI" width="600" />

## Wiring

ESP32 + flex sensor resistor ladder and IMU connections:

<img src="data\wiring_diagram.png" alt="EchoSign Wiring Diagram (ESP32 + Flex + IMU)" width="700" />
```powershell
python tools/web_ui.py
```

Open `http://localhost:5000`.

Key features:
- 3D hand visualization driven by live flex/IMU data
- Connection status, confidence bar, recent history
- Mode buttons: `Gesture` and `Sentence`
- Voice toggle: enable text-to-speech for predictions (speaks once per change)

Sentence mode operation:
- Each trigger records 4 seconds + 0.5s gap
- Auto-triggers every 4.5s while in sentence mode
- Displays predicted sentence, confidence, and mean Manhattan distance

### Build & Upload (ESP32)

Using PlatformIO tasks (VS Code):
- `Run Web UI`: starts Flask server
- `Run Data Collection`: collect samples
- `Train KNN Model`: trains gesture KNN
- `Parse and Train`: convenience training
- `Merge Logs`: utility

CLI upload:

```powershell
C:\Users\88018\.platformio\penv\Scripts\platformio.exe run --target upload --upload-port COM15
```

Adjust `--upload-port` to your COM port.

## Supported Labels

- Gestures: One, Two, Three, Four, Five, Rest, Mid, etc.
- Sentences: Configurable; add new sentences by collecting `sentence_raw_*.txt` and retraining

## Key Features

- Real-time recognition (gesture + sentence modes)
- KNN classification: Manhattan distance, distance-weighted voting, K=3/5
- Feature standardization via exported scaler params
- Quantized int8 sentence model to reduce flash footprint
- Rejection threshold via mean distance (tunable)
- Web UI with 3D visualization and optional TTS

## Threshold Tuning (Sentence Mode)

- The predictor computes mean Manhattan distance for the KNN neighbors.
- Rejection threshold (`REJECTION_MEAN_DISTANCE`) overrides predictions to `Rest` when meanD exceeds the threshold.
- After switching to int8 quantization, distances scale up; set a higher threshold accordingly.
- Recommended workflow:
  - Log `meanD` during runtime (debug output) and collect stats.
  - Set the threshold above typical in-class meanD and below off-distribution values.

## Project Configuration

See `platformio.ini` for board, framework, and serial settings.

## Notes

- See `UPGRADE_INSTRUCTIONS.md` for upgrade/migration info.
- See `SENTENCE_MODE_GUIDE.md` for sentence workflows.

## License

MIT License. See `LICENSE` for full text.

## Author

Tasir Mahtab Haque
