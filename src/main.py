"""
Real-time driver drowsiness detection.
Run from the project root with:  python -m src.main
"""
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import joblib
from tensorflow.keras.models import load_model

from src import config
from src.features.eye import eye_aspect_ratio
from src.features.mouth import mouth_aspect_ratio
from src.alerts.arduino import BuzzerAlert


def main():
    # =========================
    # LOAD MODELS
    # =========================
    drowsy_model = load_model(config.DROWSY_MODEL_PATH)
    yawn_model = load_model(config.YAWN_MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)

    sequence = deque(maxlen=config.SEQ_LEN)
    pred_buffer = deque(maxlen=config.PRED_BUFFER_SIZE)

    buzzer = BuzzerAlert()

    # =========================
    # MEDIAPIPE
    # =========================
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    cap = cv2.VideoCapture(0)

    # =========================
    # STATE
    # =========================
    sleep_start_time = None
    drowsy_state = False
    yawn_counter = 0
    drowsy_hold = 0
    yawn_hold = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                points = np.array([(lm.x * w, lm.y * h) for lm in landmarks])

                # --- Eyes ---
                left_eye = points[[33, 160, 158, 133, 153, 144]]
                right_eye = points[[263, 387, 385, 362, 380, 373]]
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

                if ear < config.EAR_THRESH:
                    if sleep_start_time is None:
                        sleep_start_time = time.time()
                else:
                    sleep_start_time = None
                    drowsy_state = False

                if sleep_start_time is not None:
                    elapsed = time.time() - sleep_start_time
                    if elapsed >= config.EYE_CLOSED_SECONDS:
                        drowsy_state = True

                # --- Mouth / yawn ---
                mouth = points[[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]]
                mar = mouth_aspect_ratio(mouth)

                x_min, x_max = int(np.min(mouth[:, 0])), int(np.max(mouth[:, 0]))
                y_min, y_max = int(np.min(mouth[:, 1])), int(np.max(mouth[:, 1]))
                mouth_crop = frame[y_min:y_max, x_min:x_max]

                yawning = False
                if mouth_crop.size != 0:
                    try:
                        mouth_crop = cv2.resize(mouth_crop, (64, 64)) / 255.0
                        mouth_crop = np.expand_dims(mouth_crop, axis=0)
                        pred_yawn = yawn_model.predict(mouth_crop, verbose=0)[0][0]

                        if pred_yawn > config.YAWN_MODEL_CONF_THRESH and mar > config.MAR_THRESH:
                            yawn_counter += 1
                        else:
                            yawn_counter = max(0, yawn_counter - 1)

                        yawning = yawn_counter >= config.YAWN_FRAMES
                    except Exception:
                        pass

                # --- LSTM fatigue features ---
                features = [ear, mar, int(drowsy_state), 0.0, int(yawning)]
                features = scaler.transform([features])[0]
                sequence.append(features)

                if len(sequence) == config.SEQ_LEN:
                    input_data = np.array(sequence).reshape(1, config.SEQ_LEN, 5)
                    pred = drowsy_model.predict(input_data, verbose=0)[0][0]
                    pred_buffer.append(pred)
                    smooth_pred = np.mean(pred_buffer)

                    if drowsy_state or smooth_pred > config.PRED_THRESH:
                        drowsy_hold += 1
                    else:
                        drowsy_hold = max(0, drowsy_hold - 1)

                    if yawning:
                        yawn_hold += 1
                    else:
                        yawn_hold = max(0, yawn_hold - 1)

                    drowsy_state = drowsy_hold >= config.HOLD_FRAMES

                buzzer.set(drowsy_state)

                # --- Display ---
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if drowsy_state:
                    cv2.putText(frame, "DROWSY ALERT!", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    cv2.putText(frame, "NORMAL", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "NO FACE", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Drowsiness System", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
