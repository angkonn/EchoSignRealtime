#!/usr/bin/env python
"""
EchoSign – Sentence Data Collector
-----------------------------------
Collects 3-second windows of continuous sensor data for complete sentences/phrases.
Each 3-second window represents ONE complete sentence (e.g., "how are you").

Usage:
  python collect_sentence_data.py --port COM15

The tool will:
  1. Prompt for sentence name
  2. Show live preview of sensor data
  3. Wait for button press to START 3-second recording
  4. Automatically stop after 3 seconds
  5. Save to: data/sentence_raw_<label>_<id>.txt

Requirements:
  - ESP32 must be in DATA COLLECTION mode (RUN_MODE = 0)
  - Button connected to trigger recording
"""

import argparse
import os
import sys
import time
import serial

BANNER = """
╔═══════════════════════════════════════════════════════════╗
║     EchoSign – Sentence/Phrase Data Collector            ║
║     Collects 4-second windows for complete sentences     ║
╚═══════════════════════════════════════════════════════════╝

Expected ESP32 output format:
  FLEX: f1 f2 f3 f4 f5 | ACC: ax ay az | GYRO: gx gy gz | GDP=val

WORKFLOW:
  1. Enter sentence name (e.g., "how_are_you", "i_eat_rice")
  2. Preview sensor data (3 sec) - get ready
  3. Press ENTER to start 4-second recording
  4. Perform the complete sentence gesture sequence
  5. Recording automatically stops after 4 seconds
  6. File saved to: data/sentence_raw_<label>_<id>.txt

Repeat 5-10 times per sentence for best model accuracy!
"""

# Recording parameters
RECORD_DURATION_SEC = 4.0
SAMPLE_RATE_HZ = 20  # 20 Hz = 50ms between samples
EXPECTED_SAMPLES = int(RECORD_DURATION_SEC * SAMPLE_RATE_HZ)  # ~80 samples


def make_slug(label: str) -> str:
    """Convert label to filesystem-safe slug"""
    slug = label.strip().replace(" ", "_").lower()
    slug = "".join(c for c in slug if c.isalnum() or c in "_-")
    return slug or "sentence"


def preview_data(ser: serial.Serial, duration: float = 3.0) -> None:
    """Show live sensor data preview without recording"""
    print(f"\n{'='*60}")
    print(f"PREVIEW MODE - Get your hands ready!")
    print(f"Showing data for {duration:.1f} seconds...")
    print(f"{'='*60}\n")

    end_time = time.time() + duration
    count = 0
    
    while time.time() < end_time:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line or not line.startswith("FLEX:"):
            continue
        
        count += 1
        # Print every 5th sample to avoid flooding
        if count % 5 == 0:
            print(f"  {line}")
    
    print(f"\nPreview complete. Saw {count} samples.")
    print(f"{'='*60}\n")


def record_sentence(ser: serial.Serial, duration: float = RECORD_DURATION_SEC) -> list:
    """Record exactly 'duration' seconds of sensor data"""
    print(f"\n{'*'*60}")
    print(f"  RECORDING IN PROGRESS - Perform your sentence!")
    print(f"  Duration: {duration:.1f} seconds")
    print(f"{'*'*60}\n")
    
    # Send START command to ESP32
    try:
        ser.write(b"S")
    except Exception as e:
        print(f"Warning: Could not send START command: {e}")
    
    samples = []
    start_time = time.time()
    last_update = start_time
    
    ser.reset_input_buffer()
    
    while True:
        elapsed = time.time() - start_time
        
        # Stop after duration
        if elapsed >= duration:
            break
        
        # Progress indicator every 0.5 sec
        if time.time() - last_update >= 0.5:
            progress = (elapsed / duration) * 100
            print(f"  Recording: {elapsed:.1f}s / {duration:.1f}s ({progress:.0f}%) - {len(samples)} samples", end="\r")
            last_update = time.time()
        
        # Read sensor line
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line or not line.startswith("FLEX:"):
            continue
        
        samples.append(line)
    
    # Send STOP command to ESP32
    try:
        ser.write(b"E")
    except Exception as e:
        print(f"\nWarning: Could not send STOP command: {e}")
    
    print(f"\n\n{'*'*60}")
    print(f"  RECORDING COMPLETE!")
    print(f"  Collected {len(samples)} samples in {duration:.1f} seconds")
    print(f"  Average rate: {len(samples)/duration:.1f} Hz")
    print(f"{'*'*60}\n")
    
    return samples


def save_sentence_data(samples: list, label: str, file_path: str) -> None:
    """Save recorded samples to file"""
    with open(file_path, "w", encoding="utf-8") as f:
        # Write header
        f.write(f"# sentence_label={label}\n")
        f.write(f"# timestamp={time.time()}\n")
        f.write(f"# duration_sec={RECORD_DURATION_SEC}\n")
        f.write(f"# sample_rate_hz={SAMPLE_RATE_HZ}\n")
        f.write(f"# total_samples={len(samples)}\n")
        f.write("# format: FLEX: f1 f2 f3 f4 f5 | ACC: ax ay az | GYRO: gx gy gz | GDP=val\n")
        f.write("#\n")
        
        # Write data
        for line in samples:
            f.write(line + "\n")
    
    print(f"✓ Saved to: {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect 3-second sentence data for sign language recognition"
    )
    parser.add_argument("--port", default="COM15", help="Serial port (default: COM15)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--timeout", type=float, default=0.5, help="Serial timeout")
    args = parser.parse_args()

    print(BANNER)

    # Setup data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    # Open serial connection
    try:
        ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
        print(f"✓ Connected to {args.port} at {args.baud} baud\n")
    except Exception as e:
        print(f"✗ ERROR: Could not open serial port {args.port}")
        print(f"  Reason: {e}")
        print(f"\nTroubleshooting:")
        print(f"  - Check if device is connected")
        print(f"  - Verify correct port (use Device Manager on Windows)")
        print(f"  - Close other programs using the serial port")
        sys.exit(1)

    time.sleep(2)
    ser.reset_input_buffer()

    # Main collection loop
    session_count = 0
    
    while True:
        print("\n" + "="*60)
        print("NEW SENTENCE RECORDING SESSION")
        print("="*60)
        
        # Get sentence label
        label = input("\nEnter sentence name (e.g., 'how_are_you' or blank to quit): ").strip()
        if not label:
            print("\n✓ Exiting. Thank you!")
            break
        
        slug = make_slug(label)
        
        # Find next available session ID
        session_id = 1
        while True:
            filename = f"sentence_raw_{slug}_{session_id:02d}.txt"
            file_path = os.path.join(data_dir, filename)
            if not os.path.exists(file_path):
                break
            session_id += 1
        
        print(f"\nSentence: '{label}'")
        print(f"Will save to: {filename}")
        print(f"Recording duration: {RECORD_DURATION_SEC} seconds")
        print(f"Target samples: ~{EXPECTED_SAMPLES} samples @ {SAMPLE_RATE_HZ} Hz")
        
        # Show preview
        preview_data(ser, duration=3.0)
        
        # Wait for user ready
        input("Press ENTER when ready to START 3-second recording...")
        
        # Countdown
        print("\nStarting in...")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        print("  GO!\n")
        
        # Record
        samples = record_sentence(ser, duration=RECORD_DURATION_SEC)
        
        # Check sample count
        if len(samples) < EXPECTED_SAMPLES * 0.5:  # Less than 50% of expected
            print(f"\n⚠ WARNING: Only got {len(samples)} samples (expected ~{EXPECTED_SAMPLES})")
            print("  This might indicate:")
            print("  - ESP32 not sending data fast enough")
            print("  - Serial connection issues")
            retry = input("  Save anyway? (y/n): ").strip().lower()
            if retry != 'y':
                print("  Discarding this recording.\n")
                continue
        
        # Save
        save_sentence_data(samples, label, file_path)
        session_count += 1
        
        print(f"\n{'='*60}")
        print(f"Session {session_count} complete!")
        print(f"{'='*60}")
        
        # Ask to continue
        print(f"\nRecommendation: Record 5-10 sessions per sentence for best accuracy")
        again = input("Record another session? (y=yes, n=quit, any other key=same sentence again): ").strip().lower()
        
        if again == 'n':
            break
        elif again != 'y':
            # Same sentence, decrement to reuse this session
            continue

    # Cleanup
    ser.close()
    print(f"\n{'='*60}")
    print(f"✓ Collection complete! Recorded {session_count} sessions.")
    print(f"  Data saved in: {data_dir}")
    print(f"\nNext steps:")
    print(f"  1. Run: python tools/train_sentence_knn.py")
    print(f"  2. Flash firmware with sentence mode enabled")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
