import os
import threading
import time
import json
from flask import Flask, jsonify, Response, send_from_directory
from flask_cors import CORS
import serial

# -------- CONFIG --------
SERIAL_PORT = "COM15"
BAUDRATE = 115200

# Path to data folder with HTML/CSS/JS
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Shared state
class State:
    def __init__(self):
        self.latest_data = {}
        self.latest_pred_line = "---"
        self.ser = None
        self.lock = threading.Lock()

STATE = State()


# -------- SERIAL READER THREAD --------
def serial_reader():
    global STATE
    while True:
        try:
            if STATE.ser is None or not STATE.ser.is_open:
                print(f"[Serial] Opening {SERIAL_PORT} @ {BAUDRATE}")
                STATE.ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1.0)
                time.sleep(2)
                if STATE.ser is not None:
                    try:
                        STATE.ser.reset_input_buffer()
                    except Exception:
                        pass

            if STATE.ser is None:
                # No serial port available yet; wait a bit and retry
                time.sleep(0.1)
                continue

            raw = STATE.ser.readline()
            if not raw:
                # nothing read this iteration (timeout)
                continue

            try:
                line = raw.decode("utf-8", errors="ignore").strip()
            except Exception:
                # Fallback: ensure we have a string even if decode fails
                line = str(raw).strip()

            # Parse PRED: lines (legacy format)
            if line.startswith("PRED:"):
                with STATE.lock:
                    STATE.latest_pred_line = line
                    # Try to extract gesture name
                    try:
                        parts = line.split(":")
                        if len(parts) > 1:
                            gesture = parts[1].split("(")[0].strip()
                            STATE.latest_data["label"] = gesture
                    except:
                        pass
            
            # Parse JSON lines (compact format from main.cpp)
            elif line.startswith("{") and line.endswith("}"):
                try:
                    data = json.loads(line)
                    with STATE.lock:
                        STATE.latest_data = data
                        # Update legacy pred line for compatibility
                        if "label" in data:
                            meanD = data.get("meanD", 0)
                            STATE.latest_pred_line = f"PRED: {data['label']} (meanD={meanD:.2f})"
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            print("[Serial] Error:", e)
            if STATE.ser:
                try:
                    STATE.ser.close()
                except:
                    pass
            STATE.ser = None
            time.sleep(2)


# -------- FLASK WEB SERVER --------
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    """Serve the premium HTML interface"""
    return send_from_directory(DATA_DIR, "index.html")

@app.route("/style.css")
def style():
    """Serve CSS file"""
    return send_from_directory(DATA_DIR, "style.css")

@app.route("/main.js")
def main_js():
    """Serve JavaScript file"""
    return send_from_directory(DATA_DIR, "main.js")

@app.route("/hand.glb")
def hand_model():
    """Serve 3D hand model"""
    return send_from_directory(DATA_DIR, "hand.glb")

@app.route("/data")
def api_data():
    """REST endpoint for real-time data (fallback for WebSocket)"""
    with STATE.lock:
        data = STATE.latest_data.copy()
    
    # Ensure we have at least a label field
    if not data:
        data = {"label": "unknown", "gdp": 0}
    
    return jsonify(data)

@app.route("/api/pred")
def api_pred():
    """Legacy API endpoint for compatibility"""
    with STATE.lock:
        line = STATE.latest_pred_line
        data = STATE.latest_data.copy()
    
    # Parse gesture name
    gesture = data.get("label", "---")
    if gesture == "---" and line.startswith("PRED:"):
        try:
            gesture = line.split(":")[1].split("(")[0].strip()
        except:
            pass
    
    return jsonify(pred=line, label=gesture)


def main():
    t = threading.Thread(target=serial_reader, daemon=True)
    t.start()
    print("Web UI running at http://127.0.0.1:5000/")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
