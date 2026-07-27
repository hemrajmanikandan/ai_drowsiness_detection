import sys
import os

# Fix import path (VERY IMPORTANT)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from src.features.eye import eye_aspect_ratio
from src.features.mouth import mouth_aspect_ratio

# =========================
# CONFIG
# =========================
DATASET_PATH = "data/dataset_raw/Driver Drowsiness Dataset (DDD)"
OUTPUT_CSV = "data/dataset.csv"

# =========================
# MEDIAPIPE INIT
# =========================
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

# =========================
# STORAGE
# =========================
rows = []
total_images = 0
face_detected_count = 0

# =========================
# PROCESS IMAGE
# =========================
def process_image(path, label):
    global face_detected_count

    img = cv2.imread(path)

    if img is None:
        print(f"[WARNING] Could not read image: {path}")
        return

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print(f"[SKIP] No face detected: {path}")
        return

    face_detected_count += 1

    landmarks = results.multi_face_landmarks[0].landmark
    points = np.array([(lm.x * w, lm.y * h) for lm in landmarks])

    try:
        # Eye landmarks
        left_eye = points[[33, 160, 158, 133, 153, 144]]
        right_eye = points[[263, 387, 385, 362, 380, 373]]

        # Mouth landmarks
        mouth = points[[61,146,91,181,84,17,314,405,321,375,291,308]]

        # Features
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
        mar = mouth_aspect_ratio(mouth)

        perclos = 1 if ear < 0.2 else 0
        pose = 0  # placeholder (can upgrade later)
        yawn = 1 if mar > 0.6 else 0

        rows.append([ear, mar, perclos, pose, yawn, label])

    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {path} → {e}")


# =========================
# WALK DATASET (RECURSIVE)
# =========================
for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        total_images += 1
        file_path = os.path.join(root, file)

        # Label detection (SMART)
        folder_name = root.lower()

        if "drowsy" in folder_name:
            label = 1
        else:
            label = 0

        process_image(file_path, label)


# =========================
# SAVE DATASET
# =========================
df = pd.DataFrame(rows, columns=[
    "ear", "mar", "perclos", "pose", "yawn_detected", "label"
])

df.to_csv(OUTPUT_CSV, index=False)

# =========================
# SUMMARY
# =========================
print("\n==========================")
print("📊 DATASET SUMMARY")
print("==========================")
print(f"Total images scanned: {total_images}")
print(f"Faces detected: {face_detected_count}")
print(f"Final samples: {len(rows)}")

if len(rows) == 0:
    print("❌ ERROR: Dataset is empty!")
    print("👉 Check dataset path or face detection")
else:
    print("✅ Dataset created successfully!")