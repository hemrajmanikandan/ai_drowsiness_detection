"""
================================================================================
AI-Based Drowsiness Detection System
Configuration File (config.py)

This file contains all configurable parameters used throughout the project.
Do NOT hardcode values inside other modules. Instead, import them from here.

Example:
    from config import EAR_THRESH, CAMERA_INDEX

Author : Hemraj
================================================================================
"""

import os

# ==============================================================================
# PROJECT PATHS
# ==============================================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
ASSET_DIR = os.path.join(BASE_DIR, "assets")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# MODEL PATHS
# ==============================================================================

DROWSY_MODEL_PATH = os.path.join(MODEL_DIR, "drowsiness_model.h5")
YAWN_MODEL_PATH = os.path.join(MODEL_DIR, "yawn_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.save")

# ==============================================================================
# CAMERA SETTINGS
# ==============================================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

FPS = 30

# Flip image horizontally (webcam selfie view)
FLIP_FRAME = True

# ==============================================================================
# AI MODEL SETTINGS
# ==============================================================================

SEQ_LEN = 15

PRED_THRESH = 0.50

PRED_BUFFER_SIZE = 5

HOLD_FRAMES = 10

# ==============================================================================
# EYE (EAR) SETTINGS
# ==============================================================================

EAR_THRESH = 0.20

# Eyes must remain closed for this many seconds
EYE_CLOSED_SECONDS = 1.0

# ==============================================================================
# MOUTH (MAR) SETTINGS
# ==============================================================================

MAR_THRESH = 0.70

YAWN_MODEL_CONF_THRESH = 0.70

# Number of consecutive frames before confirming a yawn
YAWN_FRAMES = 8

# ==============================================================================
# PERCLOS SETTINGS
# ==============================================================================

# Percentage of eye closure indicating fatigue
PERCLOS_THRESH = 0.40

PERCLOS_WINDOW_SECONDS = 60

# ==============================================================================
# HEAD POSE SETTINGS
# ==============================================================================

HEAD_DOWN_THRESHOLD = 25

HEAD_TILT_THRESHOLD = 20

POSE_SMOOTHING_WINDOW = 5

# ==============================================================================
# MEDIAPIPE SETTINGS
# ==============================================================================

MIN_DETECTION_CONFIDENCE = 0.5

MIN_TRACKING_CONFIDENCE = 0.5

MAX_NUM_FACES = 1

# ==============================================================================
# DISPLAY SETTINGS
# ==============================================================================

SHOW_FPS = True

SHOW_EAR = True

SHOW_MAR = True

SHOW_PERCLOS = True

SHOW_POSE = True

SHOW_YAWN = True

SHOW_LANDMARKS = True

SHOW_PREDICTION = True

# ==============================================================================
# COLORS (BGR FORMAT)
# ==============================================================================

GREEN = (0, 255, 0)

RED = (0, 0, 255)

BLUE = (255, 0, 0)

YELLOW = (0, 255, 255)

WHITE = (255, 255, 255)

BLACK = (0, 0, 0)

CYAN = (255, 255, 0)

# ==============================================================================
# FONT SETTINGS
# ==============================================================================

FONT = "FONT_HERSHEY_SIMPLEX"

FONT_SCALE = 0.6

FONT_THICKNESS = 2

# ==============================================================================
# LOGGING SETTINGS
# ==============================================================================

SAVE_LOGS = True

LOG_INTERVAL = 5

LOG_FILE = os.path.join(LOG_DIR, "drowsiness_log.csv")

# ==============================================================================
# VIDEO RECORDING
# ==============================================================================

SAVE_VIDEO = False

OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "output.mp4")

VIDEO_CODEC = "mp4v"

# ==============================================================================
# AUDIO / BUZZER SETTINGS
# ==============================================================================

ENABLE_BUZZER = True

BUZZER_DELAY = 1

ENABLE_ALARM_SOUND = False

ALARM_SOUND = os.path.join(ASSET_DIR, "alarm.wav")

# ==============================================================================
# ARDUINO SETTINGS
# ==============================================================================

ARDUINO_PORT = os.environ.get("ARDUINO_PORT", None)

ARDUINO_BAUDRATE = int(os.environ.get("ARDUINO_BAUDRATE", "9600"))

# ==============================================================================
# PERFORMANCE SETTINGS
# ==============================================================================

USE_GPU = True

SKIP_FRAMES = 0

MAX_RUNTIME_FPS = 30

# ==============================================================================
# DEBUG SETTINGS
# ==============================================================================

DEBUG = False

PRINT_MODEL_OUTPUT = False

PRINT_POSE_VALUES = False

PRINT_EAR_MAR = False

# ==============================================================================
# CSV HEADERS
# ==============================================================================

CSV_COLUMNS = [
    "Timestamp",
    "EAR",
    "MAR",
    "PERCLOS",
    "Pitch",
    "Yaw",
    "Roll",
    "Fatigue Probability",
    "Prediction"
]

# ==============================================================================
# END OF CONFIGURATION
# ==============================================================================