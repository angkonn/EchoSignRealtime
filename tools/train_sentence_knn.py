"""
EchoSign – Sentence KNN Trainer
--------------------------------
Trains a KNN model for sentence/phrase recognition using 4-second windows.

This creates a SEPARATE model from the gesture model:
  - gesture model: single instant gestures (One, Two, Three, etc.)
  - sentence model: 4-second continuous sequences (how_are_you, i_eat_rice, etc.)

Input:  data/sentence_raw_*.txt files
Output: - data/sentence_dataset.csv
        - src/sentence_knn_model.h
        - src/sentence_scaler_params.h
        - src/sentence_label_names.h

Usage:
  python train_sentence_knn.py
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import List, Tuple, Optional, Dict, Union, Sequence

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")

# Output files
DATASET_CSV = os.path.join(DATA_DIR, "sentence_dataset.csv")
MODEL_HEADER = os.path.join(SRC_DIR, "sentence_knn_model.h")
SCALER_HEADER = os.path.join(SRC_DIR, "sentence_scaler_params.h")
LABELS_HEADER = os.path.join(SRC_DIR, "sentence_label_names.h")
MODEL_HEADER_INT8 = os.path.join(SRC_DIR, "sentence_knn_model_q.h")

# Feature columns (12 features per sample, 80 samples = 960 features total)
# Order: f1, f2, f3, f4, f5, gdp, ax, ay, az, gx, gy, gz (repeated for each time step)
FEATURES_PER_SAMPLE = 12
SAMPLES_PER_WINDOW = 80  # 4 seconds at 20 Hz
TOTAL_FEATURES = FEATURES_PER_SAMPLE * SAMPLES_PER_WINDOW


def parse_sensor_line(line: str) -> Dict[str, float]:
    """Parse a sensor data line into dictionary"""
    # Format: FLEX: f1 f2 f3 f4 f5 | ACC: ax ay az | GYRO: gx gy gz | GDP=val
    data: Dict[str, float] = {}
    
    try:
        # Extract flex values
        flex_match = re.search(r'FLEX:\s+([\d\s]+)\s+\|', line)
        if flex_match:
            flex_vals = list(map(int, flex_match.group(1).split()))
            if len(flex_vals) == 5:
                data['f1'], data['f2'], data['f3'], data['f4'], data['f5'] = flex_vals
        
        # Extract accelerometer
        acc_match = re.search(r'ACC:\s+([-\d\s]+)\s+\|', line)
        if acc_match:
            acc_vals = list(map(int, acc_match.group(1).split()))
            if len(acc_vals) == 3:
                data['ax'], data['ay'], data['az'] = acc_vals
        
        # Extract gyroscope
        gyro_match = re.search(r'GYRO:\s+([-\d\s]+)\s+\|', line)
        if gyro_match:
            gyro_vals = list(map(int, gyro_match.group(1).split()))
            if len(gyro_vals) == 3:
                data['gx'], data['gy'], data['gz'] = gyro_vals
        
        # Extract GDP
        gdp_match = re.search(r'GDP=([\d.]+)', line)
        if gdp_match:
            data['gdp'] = float(gdp_match.group(1))
        else:
            # Calculate GDP if not provided
            if 'gx' in data and 'gy' in data and 'gz' in data:
                gx, gy, gz = data['gx'], data['gy'], data['gz']
                data['gdp'] = np.sqrt(gx**2 + gy**2 + gz**2)
    
    except Exception as e:
        print(f"Warning: Could not parse line: {line[:50]}... Error: {e}")
        return {}
    
    # Check if we got all required features
    required = ['f1', 'f2', 'f3', 'f4', 'f5', 'gdp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
    if all(k in data for k in required):
        return data
    else:
        return {}


def load_sentence_file(file_path: str) -> Tuple[Optional[str], List[Dict[str, float]]]:
    """Load a sentence data file and return label + samples"""
    label: Optional[str] = None
    samples: List[Dict[str, float]] = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Extract label from header
            if line.startswith('# sentence_label='):
                label = line.split('=', 1)[1].strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse sensor data
            if line.startswith('FLEX:'):
                sample = parse_sensor_line(line)
                if sample:
                    samples.append(sample)
    
    return label, samples


def create_feature_window(samples: List[Dict[str, float]], target_size: int = SAMPLES_PER_WINDOW) -> Optional[np.ndarray]:
    """Convert list of samples into a flattened feature vector"""
    # Resample to exactly target_size samples
    if len(samples) == 0:
        return None
    
    # Convert to array
    feature_names = ['f1', 'f2', 'f3', 'f4', 'f5', 'gdp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
    data_matrix = np.array([[s[k] for k in feature_names] for s in samples])
    
    # Resample to target size using linear interpolation
    original_indices = np.linspace(0, len(samples) - 1, len(samples))
    target_indices = np.linspace(0, len(samples) - 1, target_size)
    
    resampled = np.zeros((target_size, FEATURES_PER_SAMPLE))
    for i in range(FEATURES_PER_SAMPLE):
        resampled[:, i] = np.interp(target_indices, original_indices, data_matrix[:, i])
    
    # Flatten to 1D feature vector
    feature_vector = resampled.flatten()
    
    return feature_vector


def load_all_sentence_data() -> Optional[pd.DataFrame]:
    """Load all sentence data files and create dataset"""
    print("\n" + "="*60)
    print("LOADING SENTENCE DATA")
    print("="*60)
    
    all_features: List[np.ndarray] = []
    all_labels: List[str] = []
    
    # Find all sentence data files
    sentence_files = []
    for filename in os.listdir(DATA_DIR):
        if filename.startswith('sentence_raw_') and filename.endswith('.txt'):
            sentence_files.append(os.path.join(DATA_DIR, filename))
    
    if not sentence_files:
        print("\n✗ ERROR: No sentence data files found!")
        print(f"  Expected files like: sentence_raw_<label>_01.txt in {DATA_DIR}")
        print(f"\n  Please run: python collect_sentence_data.py first")
        return None
    
    print(f"\nFound {len(sentence_files)} sentence data files")
    
    # Process each file
    file_count = {}
    for file_path in sorted(sentence_files):
        filename = os.path.basename(file_path)
        label, samples = load_sentence_file(file_path)
        
        if not label or not samples:
            print(f"  ⚠ Skipping {filename}: no label or no data")
            continue
        
        # Create feature window
        features = create_feature_window(samples, target_size=SAMPLES_PER_WINDOW)
        if features is None or len(features) != TOTAL_FEATURES:
            print(f"  ⚠ Skipping {filename}: invalid feature window")
            continue
        
        all_features.append(features)
        all_labels.append(label)
        
        # Count files per label
        file_count[label] = file_count.get(label, 0) + 1
        
        print(f"  ✓ {filename}: {len(samples)} samples → {len(features)} features, label='{label}'")
    
    if not all_features:
        print("\n✗ ERROR: No valid data loaded!")
        return None
    
    # Create DataFrame
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total windows: {len(all_features)}")
    print(f"Features per window: {TOTAL_FEATURES}")
    print(f"Labels: {len(set(all_labels))}")
    print(f"\nSamples per sentence:")
    for label in sorted(file_count.keys()):
        count = file_count[label]
        status = "✓" if count >= 5 else "⚠"
        print(f"  {status} {label}: {count} samples")
    
    if any(count < 3 for count in file_count.values()):
        print(f"\n⚠ WARNING: Some sentences have < 3 samples. Recommend 5-10 per sentence!")
    
    # Create column names
    column_names = []
    for t in range(SAMPLES_PER_WINDOW):
        for feat in ['f1', 'f2', 'f3', 'f4', 'f5', 'gdp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']:
            column_names.append(f"{feat}_t{t}")
    column_names.append('label')
    
    # Create DataFrame
    df = pd.DataFrame(all_features, columns=column_names[:-1])
    df['label'] = all_labels
    
    return df


def export_sentence_scaler(scaler: StandardScaler, out_path: str) -> None:
    """Export scaler parameters to C++ header"""
    mean = np.asarray(scaler.mean_, dtype=float)
    scale = np.asarray(scaler.scale_, dtype=float)
    num_features = len(mean)
    
    lines = [
        "#pragma once",
        "#include <Arduino.h>",
        "",
        f"#define SENTENCE_NUM_FEATURES {num_features}",
        "",
        "// Scaler mean values",
        f"static const float SENTENCE_SCALER_MEAN[SENTENCE_NUM_FEATURES] = {{",
    ]
    
    # Write mean values (10 per line for readability)
    for i in range(0, num_features, 10):
        chunk = mean[i:i+10]
        vals = ", ".join(f"{v:.6f}f" for v in chunk)
        lines.append(f"  {vals},")
    lines[-1] = lines[-1].rstrip(',')  # Remove trailing comma
    lines.append("};")
    lines.append("")
    
    # Write scale values
    lines.append("// Scaler scale values")
    lines.append(f"static const float SENTENCE_SCALER_SCALE[SENTENCE_NUM_FEATURES] = {{")
    for i in range(0, num_features, 10):
        chunk = scale[i:i+10]
        vals = ", ".join(f"{v:.6f}f" for v in chunk)
        lines.append(f"  {vals},")
    lines[-1] = lines[-1].rstrip(',')
    lines.append("};")
    lines.append("")
    
    # Write standardization function
    lines.extend([
        "inline void standardizeSentenceFeatures(float feat[SENTENCE_NUM_FEATURES]) {",
        "  for (int i = 0; i < SENTENCE_NUM_FEATURES; ++i) {",
        "    feat[i] = (feat[i] - SENTENCE_SCALER_MEAN[i]) / SENTENCE_SCALER_SCALE[i];",
        "  }",
        "}",
        ""
    ])
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Wrote scaler to: {out_path}")


def export_sentence_labels(le: LabelEncoder, out_path: str) -> None:
    """Export label names to C++ header"""
    classes = list(le.classes_)
    num_classes = len(classes)
    
    lines = [
        "#pragma once",
        "",
        f"#define SENTENCE_NUM_CLASSES {num_classes}",
        "",
        "static const char* const sentence_label_names[SENTENCE_NUM_CLASSES] = {",
    ]
    
    for i, name in enumerate(classes):
        safe = str(name).replace("\\", "\\\\").replace('"', '\\"')
        comma = "," if i < num_classes - 1 else ""
        lines.append(f'  "{safe}"{comma}')
    
    lines.append("};")
    lines.append("")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Wrote labels to: {out_path}")


def export_sentence_knn_model(knn: KNeighborsClassifier, X_scaled: np.ndarray, 
                              y_enc: np.ndarray, out_path: str) -> None:  # type: ignore
    """Export KNN model to C++ header"""
    # Ensure labels are a numpy array of ints (fixes type checker issues)
    y_arr = np.asarray(y_enc, dtype=int)
    n_samples, n_features = X_scaled.shape
    k: int = getattr(knn, 'n_neighbors', 3)
    
    lines = [
        "#pragma once",
        "#include <Arduino.h>",
        "",
        f"#define SENTENCE_KNN_N_NEIGHBORS {k}",
        f"#define SENTENCE_KNN_N_SAMPLES {n_samples}",
        f"#define SENTENCE_KNN_N_FEATURES {n_features}",
        "",
        "// Training data (flattened)",
        f"static const float SENTENCE_TRAINING_DATA[SENTENCE_KNN_N_SAMPLES * SENTENCE_KNN_N_FEATURES] PROGMEM = {{",
    ]
    
    # Write training data (5 values per line)
    flat_data = X_scaled.flatten()
    for i in range(0, len(flat_data), 5):
        chunk = flat_data[i:i+5]
        vals = ", ".join(f"{v:.6f}f" for v in chunk)
        lines.append(f"  {vals},")
    lines[-1] = lines[-1].rstrip(',')
    lines.append("};")
    lines.append("")
    
    # Write labels
    lines.append("// Training labels")
    lines.append(f"static const uint8_t SENTENCE_TRAINING_LABELS[SENTENCE_KNN_N_SAMPLES] = {{")
    for i in range(0, n_samples, 20):
        chunk = y_arr[i:i+20]
        vals = ", ".join(str(int(v)) for v in chunk)
        lines.append(f"  {vals},")
    lines[-1] = lines[-1].rstrip(',')
    lines.append("};")
    lines.append("")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Wrote model to: {out_path}")


def export_sentence_knn_model_int8(X_scaled: np.ndarray, y_enc: np.ndarray, out_path: str) -> None:
    """Export quantized (int8) KNN training data + per-feature scales.

    We perform symmetric per-feature quantization on the standardized feature space.
    For each feature column j, scale_j = 127 / maxAbs_j, where maxAbs_j = max(|X[:,j]|).
    Quantized value q = clip(round(x * scale_j), -128, 127).
    Distance is computed directly on int8 values (Manhattan), preserving relative ordering.
    """
    y_arr = np.asarray(y_enc, dtype=int)
    n_samples, n_features = X_scaled.shape

    # Compute per-feature symmetric scales
    max_abs = np.max(np.abs(X_scaled), axis=0)
    # Avoid divide by zero
    max_abs[max_abs == 0] = 1e-6
    scales = 127.0 / max_abs  # multiply float -> int8

    # Quantize
    q_data = np.rint(np.clip(X_scaled * scales, -128, 127)).astype(np.int8)

    lines: List[str] = [
        "#pragma once",
        "#include <Arduino.h>",
        "",
        f"#define SENTENCE_KNN_Q_N_NEIGHBORS 3",  # matches best params; runtime can override if desired
        f"#define SENTENCE_KNN_Q_N_SAMPLES {n_samples}",
        f"#define SENTENCE_KNN_Q_N_FEATURES {n_features}",
        "",
        "// Per-feature quantization scales (multiply standardized float to get int8)",
        "static const float SENTENCE_Q_SCALES[SENTENCE_KNN_Q_N_FEATURES] PROGMEM = {",
    ]
    # Write scales (10 per line)
    for i in range(0, n_features, 10):
        chunk = scales[i:i+10]
        vals = ", ".join(f"{v:.6f}f" for v in chunk)
        lines.append(f"  {vals},")
    lines[-1] = lines[-1].rstrip(',')
    lines.append("};")
    lines.append("")

    # Write quantized training data
    lines.append("// Quantized training data (int8)")
    lines.append("static const int8_t SENTENCE_TRAINING_DATA_Q[SENTENCE_KNN_Q_N_SAMPLES * SENTENCE_KNN_Q_N_FEATURES] PROGMEM = {")
    flat_q = q_data.flatten()
    for i in range(0, len(flat_q), 32):  # 32 per line for compactness
        chunk = flat_q[i:i+32]
        vals = ", ".join(str(int(v)) for v in chunk)
        lines.append(f"  {vals},")
    lines[-1] = lines[-1].rstrip(',')
    lines.append("};")
    lines.append("")

    # Write labels
    lines.append("// Training labels")
    lines.append("static const uint8_t SENTENCE_TRAINING_LABELS_Q[SENTENCE_KNN_Q_N_SAMPLES] PROGMEM = {")
    for i in range(0, n_samples, 32):
        chunk = y_arr[i:i+32]
        vals = ", ".join(str(int(v)) for v in chunk)
        lines.append(f"  {vals},")
    lines[-1] = lines[-1].rstrip(',')
    lines.append("};")
    lines.append("")

    # Helper inline for quantizing a query vector already standardized
    lines.extend([
        "inline void quantizeSentenceFeatures(const float* inFeat, int8_t* outQ) {",
        "  for (int i = 0; i < SENTENCE_KNN_Q_N_FEATURES; ++i) {",
        "    float q = inFeat[i] * pgm_read_float(&SENTENCE_Q_SCALES[i]);",
        "    if (q > 127.0f) q = 127.0f; else if (q < -128.0f) q = -128.0f;",
        "    outQ[i] = (int8_t)lrintf(q);",
        "  }",
        "}",
        "",
    ])

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"✓ Wrote INT8 model to: {out_path}")


def main() -> None:
    print("\n" + "="*60)
    print("  EchoSign - Sentence KNN Trainer")
    print("="*60)
    
    # Load data
    df = load_all_sentence_data()
    if df is None:
        return
    
    # Save dataset
    df.to_csv(DATASET_CSV, index=False)
    print(f"\n✓ Saved dataset to: {DATASET_CSV}")
    
    # Prepare features and labels
    X = df.drop('label', axis=1).values
    # Convert pandas Series to a NumPy array with explicit dtype to satisfy type checkers
    y = df['label'].to_numpy(dtype=str)
    
    # Encode labels
    le = LabelEncoder()
    y_enc: np.ndarray = le.fit_transform(y)  # type: ignore
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    print(f"\n{'='*60}")
    print("TRAINING MODEL")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X.shape[1]}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train KNN with grid search
    print("\nTraining KNN model (this may take a minute)...")
    # Ensure training metric matches firmware (Manhattan/L1) and use distance weights
    param_grid = {
        'n_neighbors': [3, 5],  # Try K=3 and K=5
        'weights': ['distance'],
        'metric': ['manhattan']
    }
    
    knn: KNeighborsClassifier = KNeighborsClassifier()
    grid_search: GridSearchCV = GridSearchCV(knn, param_grid, cv=min(3, len(X_train)), 
                               scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_knn: KNeighborsClassifier = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}")
    
    # Evaluate
    y_pred = best_knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Export to C++
    print(f"\n{'='*60}")
    print("EXPORTING TO C++")
    print(f"{'='*60}")
    
    # Use full dataset for deployment
    X_all_scaled = scaler.transform(X)
    
    export_sentence_scaler(scaler, SCALER_HEADER)
    export_sentence_labels(le, LABELS_HEADER)
    # Float model (for fallback/reference)
    export_sentence_knn_model(best_knn, X_all_scaled, y_enc, MODEL_HEADER)
    # INT8 quantized model (primary for deployment)
    export_sentence_knn_model_int8(X_all_scaled, y_enc, MODEL_HEADER_INT8)
    
    print(f"\n{'='*60}")
    print("✓ TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nGenerated files:")
    print(f"  - {DATASET_CSV}")
    print(f"  - {SCALER_HEADER}")
    print(f"  - {LABELS_HEADER}")
    print(f"  - {MODEL_HEADER}")
    print(f"  - {MODEL_HEADER_INT8}")
    print(f"\nNext steps:")
    print(f"  1. Update firmware to include sentence prediction mode")
    print(f"  2. Add button to trigger sentence recording")
    print(f"  3. Compile and flash to ESP32")
    print(f"  4. Test your sentences!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
