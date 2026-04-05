"""
Real-time inference engine for webcam/video drowsiness detection.

Key design decisions:
- Face/eye detection via Haar cascades (lightweight, no GPU needed)
- Processes both eyes independently and averages predictions
- Integrates FatigueTracker for temporal state assessment
- Overlay HUD showing fatigue score, state, and alert status
"""
import cv2
import numpy as np
import tensorflow as tf
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.models.fatigue_tracker import FatigueTracker


# Color scheme for driver states
STATE_COLORS = {
    "ALERT":             (0, 200, 0),      # Green
    "MILD_FATIGUE":      (0, 200, 200),    # Yellow
    "MODERATE_FATIGUE":  (0, 140, 255),    # Orange
    "SEVERE_DROWSINESS": (0, 0, 255),      # Red
}


class DrowsinessDetector:
    """
    Real-time drowsiness detection system combining:
    1. Face/eye detection (Haar cascades)
    2. Eye state classification (MobileNetV2)
    3. Temporal fatigue tracking (FatigueTracker)
    """

    def __init__(self, model_path: str = None):
        # Load model
        if model_path is None:
            model_path = str(config.MODEL_DIR / "drowsiness_detector.keras")
        self.model = tf.keras.models.load_model(model_path)
        print(f"  Model loaded from {model_path}")

        # Haar cascade detectors
        haar_dir = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            haar_dir + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            haar_dir + "haarcascade_eye.xml"
        )

        # Fatigue tracker
        self.tracker = FatigueTracker()

        # Performance tracking
        self._fps_history = []

    def preprocess_eye(self, eye_roi: np.ndarray) -> np.ndarray:
        """Convert raw eye ROI to model input format."""
        eye = cv2.resize(eye_roi, (config.MODEL_INPUT_SIZE, config.MODEL_INPUT_SIZE))
        if len(eye.shape) == 2:
            eye = cv2.cvtColor(eye, cv2.COLOR_GRAY2RGB)
        elif eye.shape[2] == 3:
            eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            eye = cv2.cvtColor(eye_gray, cv2.COLOR_GRAY2RGB)
        eye = eye.astype(np.float32) / 255.0
        return eye

    def predict_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single video frame.

        Returns dict with detections, predictions, and fatigue state.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        eye_predictions = []
        eye_rois = []

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            face_upper = face_gray[:h//2, :]  # Eyes are in upper half

            eyes = self.eye_cascade.detectMultiScale(
                face_upper, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
            )

            for (ex, ey, ew, eh) in eyes[:2]:  # Max 2 eyes
                eye_roi = face_upper[ey:ey+eh, ex:ex+ew]
                eye_input = self.preprocess_eye(eye_roi)
                eye_rois.append({
                    "roi": eye_roi,
                    "bbox": (x + ex, y + ey, ew, eh),
                })

                pred = self.model.predict(
                    np.expand_dims(eye_input, 0), verbose=0
                )[0, 0]
                eye_predictions.append(pred)

        # Average prediction across detected eyes
        if eye_predictions:
            avg_prob = np.mean(eye_predictions)
        else:
            avg_prob = 0.0  # No eyes detected → assume alert

        # Update fatigue tracker
        result = self.tracker.update(avg_prob)

        return {
            "faces": faces,
            "eyes": eye_rois,
            "eye_predictions": eye_predictions,
            "avg_prob": avg_prob,
            "fatigue_result": result,
        }

    def draw_overlay(self, frame: np.ndarray, detection: dict) -> np.ndarray:
        """Draw HUD overlay with detection results and fatigue status."""
        result = detection["fatigue_result"]
        color = STATE_COLORS.get(result.driver_state, (255, 255, 255))

        # Draw face boxes
        for (x, y, w, h) in detection["faces"]:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Draw eye boxes
        for eye in detection["eyes"]:
            ex, ey, ew, eh = eye["bbox"]
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 200, 0), 1)

        # ── HUD Panel ──────────────────────────────────────────────────
        h, w = frame.shape[:2]
        panel_h = 140
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # State label
        cv2.putText(frame, f"State: {result.driver_state}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Fatigue score bar
        bar_x, bar_y, bar_w, bar_h = 10, 50, 250, 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (100, 100, 100), -1)
        fill_w = int(bar_w * result.fatigue_score)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      color, -1)
        cv2.putText(frame, f"Fatigue: {result.fatigue_score:.2f}",
                    (bar_x + bar_w + 10, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Drowsy probability
        cv2.putText(frame, f"P(drowsy): {result.smoothed_prob:.2f}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Confidence
        cv2.putText(frame, f"Confidence: {result.confidence:.2f}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # FPS
        if self._fps_history:
            fps = np.mean(self._fps_history[-30:])
            cv2.putText(frame, f"FPS: {fps:.0f}",
                        (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 1)

        # Alert banner
        if result.is_alert:
            cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 200), -1)
            cv2.putText(frame, "!! DROWSINESS ALERT !!",
                        (w // 2 - 180, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        return frame

    def run_webcam(self):
        """Run real-time inference on webcam feed."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return

        print("Starting webcam inference (press 'q' to quit)...")
        self.tracker.reset()

        while True:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            detection = self.predict_frame(frame)
            frame = self.draw_overlay(frame, detection)

            dt = time.time() - t_start
            self._fps_history.append(1.0 / max(dt, 1e-6))

            cv2.imshow("Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        summary = self.tracker.get_session_summary()
        print("\n Session Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    def run_video(self, video_path: str, output_path: str = None):
        """Run inference on a video file and optionally save output."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        self.tracker.reset()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing {frame_count} frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detection = self.predict_frame(frame)
            frame = self.draw_overlay(frame, detection)

            if writer:
                writer.write(frame)

        cap.release()
        if writer:
            writer.release()
            print(f"  Output saved to {output_path}")

        summary = self.tracker.get_session_summary()
        return summary
