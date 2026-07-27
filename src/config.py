"""
Central configuration for the drowsiness detection app.
Every tunable value lives here — main.py imports from this module
instead of redefining its own copies (that was the bug in the old version:
main.py had its own hardcoded thresholds that silently ignored this file).
"""
import os

# --- Sequence / model settings ---
SEQ_LEN = 15

# --- Eye Aspect Ratio (EAR) ---
EAR_THRESH = 0.20
EYE_CLOSED_SECONDS = 1.0   # how long eyes must stay closed before flagging drowsy

# --- Mouth Aspect Ratio (MAR) / Yawn ---
MAR_THRESH = 0.70
YAWN_MODEL_CONF_THRESH = 0.7
YAWN_FRAMES = 8

# --- LSTM fatigue prediction ---
PRED_THRESH = 0.5
HOLD_FRAMES = 10
PRED_BUFFER_SIZE = 5

# --- Logging ---
LOG_INTERVAL = 5

# --- Model paths (relative to project root) ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DROWSY_MODEL_PATH = os.path.join(MODEL_DIR, "drowsiness_model.h5")
YAWN_MODEL_PATH = os.path.join(MODEL_DIR, "yawn_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.save")

# --- Arduino / buzzer ---
# Configurable via environment variable so it isn't hardcoded per-machine.
# Set ARDUINO_PORT in a .env file or your shell, e.g. export ARDUINO_PORT=COM5
ARDUINO_PORT = os.environ.get("ARDUINO_PORT", None)
ARDUINO_BAUDRATE = int(os.environ.get("ARDUINO_BAUDRATE", "9600"))
