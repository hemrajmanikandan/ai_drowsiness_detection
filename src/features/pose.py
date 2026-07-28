import cv2
import numpy as np
from collections import deque
import time


class HeadPoseEstimator:
    """
    Advanced Head Pose Estimator

    Features:
    - 3D Head Pose Estimation
    - Pitch, Yaw, Roll
    - Temporal Smoothing
    - Ready for Distraction Detection
    """

    def __init__(self, frame_width, frame_height):
        import time

        self.distraction_start_time = None
        self.distraction_duration = 0
        self.alert_threshold = 10      # seconds

        self.frame_width = frame_width
        self.frame_height = frame_height

        # Camera parameters
        self.focal_length = frame_width

        self.camera_matrix = np.array([
            [self.focal_length, 0, frame_width / 2],
            [0, self.focal_length, frame_height / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros((4, 1))

        # -----------------------------
        # 3D Face Model
        # -----------------------------
        self.model_points = np.array([

            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye corner
            (225.0, 170.0, -135.0),   # Right eye corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner

        ], dtype=np.float64)

        # -----------------------------
        # History Buffers
        # -----------------------------
        self.pitch_history = deque(maxlen=10)
        self.yaw_history = deque(maxlen=10)
        self.roll_history = deque(maxlen=10)

        self.previous_pitch = 0
        self.previous_yaw = 0
        self.previous_roll = 0
    # ==========================================================
    # Get required MediaPipe landmarks
    # ==========================================================

    def get_image_points(self, landmarks):

        h = self.frame_height
        w = self.frame_width

        image_points = np.array([

            (
                landmarks.landmark[1].x * w,
                landmarks.landmark[1].y * h
            ),

            (
                landmarks.landmark[152].x * w,
                landmarks.landmark[152].y * h
            ),

            (
                landmarks.landmark[33].x * w,
                landmarks.landmark[33].y * h
            ),

            (
                landmarks.landmark[263].x * w,
                landmarks.landmark[263].y * h
            ),

            (
                landmarks.landmark[61].x * w,
                landmarks.landmark[61].y * h
            ),

            (
                landmarks.landmark[291].x * w,
                landmarks.landmark[291].y * h
            )

        ], dtype=np.float64)

        return image_points

    # ==========================================================
    # Estimate Head Pose
    # ==========================================================

    def estimate_pose(self, landmarks):

        image_points = self.get_image_points(landmarks)

        success, rotation_vector, translation_vector = cv2.solvePnP(

            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE

        )

        if not success:

            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
        pitch = ((angles[0] + 180) % 360) - 180
        yaw = ((angles[1] + 180) % 360) - 180
        roll = ((angles[2] + 180) % 360) - 180

        return {

            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "rotation_vector": rotation_vector,
            "translation_vector": translation_vector,
            "rotation_matrix": rotation_matrix

        }


    # ==========================================================
    # Smooth Pose Angles
    # ==========================================================
    def smooth_pose(self, pitch, yaw, roll):

        self.pitch_history.append(pitch)
        self.yaw_history.append(yaw)
        self.roll_history.append(roll)

        alpha = 0.2

        self.previous_pitch = alpha * pitch + (1-alpha) * self.previous_pitch
        self.previous_yaw = alpha * yaw + (1-alpha) * self.previous_yaw
        self.previous_roll = alpha * roll + (1-alpha) * self.previous_roll

        return (
            self.previous_pitch,
            self.previous_yaw,
            self.previous_roll
        )

    # ==========================================================
    # Calculate Head Stability
    # ==========================================================
    def calculate_stability(self):

        if len(self.pitch_history) < 2:
            return 100

        pitch_var = np.var(self.pitch_history)
        yaw_var = np.var(self.yaw_history)
        roll_var = np.var(self.roll_history)

        total_variance = pitch_var + yaw_var + roll_var

        stability = max(0, 100 - total_variance)

        return stability

    # ==========================================================
    # Detect Driver Distraction
    # ==========================================================
    def detect_distraction(self, pitch, yaw, roll):

        state = "NORMAL"

        if yaw > 20:
            state = "LOOKING_RIGHT"

        elif yaw < -20:
            state = "LOOKING_LEFT"

        elif pitch > 20:
            state = "LOOKING_DOWN"

        elif pitch < -15:
            state = "LOOKING_UP"

        elif abs(roll) > 20:
            state = "HEAD_TILTED"

        return state

    # ==========================================================
    # Calculate Pose Score
    # ==========================================================
    def calculate_pose_score(self, pitch, yaw, roll):

        score = 100

        score -= abs(pitch) * 2
        score -= abs(yaw) * 1.5
        score -= abs(roll)

        score = max(0, min(score, 100))

        return score

    # ==========================================================
    # Complete Pose Analysis
    # ==========================================================
    def analyze(self, landmarks):

        pose = self.estimate_pose(landmarks)

        if pose is None:
            return None

        pitch = pose["pitch"]
        yaw = pose["yaw"]
        roll = pose["roll"]

        pitch, yaw, roll = self.smooth_pose(
            pitch,
            yaw,
            roll
        )

        stability = self.calculate_stability()

        pose_score = self.calculate_pose_score(
            pitch,
            yaw,
            roll
        )

        state = self.detect_distraction(
            pitch,
            yaw,
            roll
        )

        return {

            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "pose_score": pose_score,
            "stability": stability,
            "state": state,
            "rotation_vector": pose["rotation_vector"],
            "translation_vector": pose["translation_vector"],
        }

    # ==========================================================
    # Check if Driver is Distracted
    # =====================================================

    def update_distraction_timer(self, state):

        if state != "NORMAL":

            if self.distraction_start_time is None:
                self.distraction_start_time = time.time()

            self.distraction_duration = (
                time.time() - self.distraction_start_time
            )

        else:

            self.distraction_start_time = None
            self.distraction_duration = 0

        return self.distraction_duration >= self.alert_threshold
    
    # ==========================================================
    # Draw Pose Axis
    # ==========================================================
    
    def draw_pose_axis(self,
                       frame,
                       rotation_vector,
                       translation_vector):
    
        axis = np.float32([
            [50,0,0],
            [0,50,0],
            [0,0,50]
        ])
    
        nose_end, _ = cv2.projectPoints(
            axis,
            rotation_vector,
            translation_vector,
            self.camera_matrix,
            self.dist_coeffs
        )
    
        h = self.frame_height
        w = self.frame_width
    
        origin = (
            int(w/2),
            int(h/2)
        )
    
        x_axis = tuple(
            nose_end[0].ravel().astype(int)
        )
    
        y_axis = tuple(
            nose_end[1].ravel().astype(int)
        )
    
        z_axis = tuple(
            nose_end[2].ravel().astype(int)
        )
    
        cv2.line(frame, origin, x_axis, (0,0,255), 3)
        cv2.line(frame, origin, y_axis, (0,255,0), 3)
        cv2.line(frame, origin, z_axis, (255,0,0), 3)
    
    
    # ==========================================================
    # Draw Information
    # ==========================================================
    
    def draw_information(self,
                         frame,
                         result):
    
        color = (0,255,0)
    
        if result["state"] != "NORMAL":
    
            color = (0,165,255)
    
        distracted = self.update_distraction_timer(
            result["state"]
        )
    
        if distracted:
    
            color = (0,0,255)
    
            cv2.putText(
                frame,
                "DRIVER DISTRACTED",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3
            )
    
        cv2.putText(
            frame,
            f"Pitch : {result['pitch']:.2f}",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
    
        cv2.putText(
            frame,
            f"Yaw : {result['yaw']:.2f}",
            (20,110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
    
        cv2.putText(
            frame,
            f"Roll : {result['roll']:.2f}",
            (20,140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
    
        cv2.putText(
            frame,
            f"Pose Score : {result['pose_score']:.0f}",
            (20,170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
    
        cv2.putText(
            frame,
            f"Stability : {result['stability']:.0f}",
            (20,200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
    
        cv2.putText(
            frame,
            f"State : {result['state']}",
            (20,230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )
        
    