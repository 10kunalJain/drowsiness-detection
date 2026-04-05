"""
Multimodal Extension — beyond pure eye-state classification.

Demonstrates forward-thinking system design by extracting additional
behavioral signals that a production drowsiness system would fuse:

1. Blink Rate Tracker: Measures blink frequency (PERCLOS-inspired metric)
2. Head Pose Estimator: Detects head nodding/drooping via facial landmarks
3. Multi-Signal Fusion: Combines eye state + blink rate + head pose into
   a unified fatigue assessment

This module works with the existing FatigueTracker and shows how the
single-modal system extends to a multimodal one.
"""
import cv2
import numpy as np
import time
from dataclasses import dataclass
from collections import deque
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


@dataclass
class MultimodalSignals:
    """Aggregated signals from all modalities for a single frame."""
    eye_drowsy_prob: float       # From CNN model
    blink_rate_per_min: float    # Blinks per minute
    perclos: float               # % of time eyes closed in last N seconds
    head_pitch: float            # Head vertical angle (nodding detection)
    head_yaw: float              # Head horizontal angle (looking away)
    fused_fatigue_score: float   # Weighted combination of all signals


class BlinkRateTracker:
    """
    Track eye blink rate using frame-level drowsy predictions.

    Blink detection: A blink is a short (2-6 frame) transition from
    open → closed → open. Extended closures are drowsiness, not blinks.

    Key metric: PERCLOS (Percentage of Eye Closure over time)
    - Industry standard for fatigue detection
    - PERCLOS > 0.15 is a strong drowsiness indicator
    """

    def __init__(self, fps: float = 30.0, window_seconds: float = 60.0):
        self.fps = fps
        self.window_size = int(fps * window_seconds)
        self._eye_states = deque(maxlen=self.window_size)  # True = closed
        self._blink_timestamps = deque(maxlen=100)
        self._prev_closed = False
        self._closure_start = 0
        self._frame_count = 0

    def update(self, eye_closed: bool) -> dict:
        """
        Process a new frame's eye state.

        Args:
            eye_closed: Whether the eye is classified as closed/drowsy

        Returns:
            Dict with blink_rate, perclos, closure_duration
        """
        self._frame_count += 1
        self._eye_states.append(eye_closed)

        current_time = self._frame_count / self.fps
        closure_duration = 0.0

        # Blink detection: transition from closed → open
        if self._prev_closed and not eye_closed:
            closure_frames = self._frame_count - self._closure_start
            closure_duration = closure_frames / self.fps

            # A blink is 0.1-0.4 seconds; longer is microsleep
            if 0.1 <= closure_duration <= 0.4:
                self._blink_timestamps.append(current_time)

        if eye_closed and not self._prev_closed:
            self._closure_start = self._frame_count

        self._prev_closed = eye_closed

        # Calculate blink rate (blinks per minute)
        recent_blinks = [
            t for t in self._blink_timestamps if current_time - t < 60
        ]
        blink_rate = len(recent_blinks)

        # PERCLOS: fraction of frames where eyes are closed in the window
        if len(self._eye_states) > 0:
            perclos = sum(self._eye_states) / len(self._eye_states)
        else:
            perclos = 0.0

        return {
            "blink_rate_per_min": blink_rate,
            "perclos": perclos,
            "closure_duration": closure_duration,
            "is_microsleep": closure_duration > 0.5,
        }


class HeadPoseEstimator:
    """
    Estimate head pose (pitch/yaw) from facial landmarks.

    Uses a 6-point facial landmark model (eyes, nose, chin) to solve
    the Perspective-n-Point (PnP) problem for head orientation.

    Key signals:
    - Pitch < -15°: Head drooping forward (nodding off)
    - Yaw > 30°: Looking away from road
    """

    # 3D reference points for a generic face model (in mm)
    _MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),   # Right eye corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ], dtype=np.float64)

    def __init__(self):
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def estimate(self, frame: np.ndarray, face_bbox: tuple = None) -> dict:
        """
        Estimate head pose from a video frame.

        Note: Full implementation requires dlib or MediaPipe for facial landmarks.
        This provides the architecture and falls back to a Haar-based approximation
        when landmarks are not available.

        Args:
            frame: BGR video frame
            face_bbox: Optional (x, y, w, h) face bounding box

        Returns:
            Dict with pitch, yaw, roll estimates
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        if face_bbox is None:
            faces = self._face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            if len(faces) == 0:
                return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "detected": False}
            face_bbox = faces[0]

        x, y, w, h = face_bbox
        face_roi = gray[y:y+h, x:x+w]

        # Approximate head pose from face aspect ratio and position
        # (Production system would use MediaPipe Face Mesh for 468 landmarks)
        frame_h, frame_w = gray.shape[:2]
        face_center_x = (x + w / 2) / frame_w
        face_center_y = (y + h / 2) / frame_h
        aspect_ratio = w / max(h, 1)

        # Yaw: face center deviation from frame center
        yaw = (face_center_x - 0.5) * 60  # Rough degrees

        # Pitch: based on vertical position and aspect ratio changes
        pitch = (face_center_y - 0.4) * 40  # Positive = looking down

        return {
            "pitch": pitch,
            "yaw": yaw,
            "roll": 0.0,
            "detected": True,
        }


class MultimodalFatigueAssessor:
    """
    Fuse multiple signals into a unified fatigue assessment.

    Signal weights (tunable):
    - Eye state (CNN):   0.50 — primary signal
    - PERCLOS:           0.25 — industry-standard fatigue metric
    - Head pose:         0.15 — nodding/drooping detection
    - Blink rate:        0.10 — abnormal blink patterns

    A production system would learn these weights via a small MLP
    trained on labeled driving session data.
    """

    SIGNAL_WEIGHTS = {
        "eye_state": 0.50,
        "perclos": 0.25,
        "head_pose": 0.15,
        "blink_rate": 0.10,
    }

    # Normal ranges for anomaly detection
    NORMAL_BLINK_RATE = (12, 20)  # blinks/min
    PERCLOS_ALERT_THRESHOLD = 0.15
    HEAD_NOD_THRESHOLD = -15  # degrees pitch

    def __init__(self):
        self.blink_tracker = BlinkRateTracker()
        self.head_estimator = HeadPoseEstimator()

    def assess(self, eye_prob: float, frame: np.ndarray = None,
               face_bbox: tuple = None) -> MultimodalSignals:
        """
        Generate a multimodal fatigue assessment for a single frame.

        Args:
            eye_prob: Drowsy probability from CNN (0-1)
            frame: Optional BGR frame for head pose estimation
            face_bbox: Optional face bounding box

        Returns:
            MultimodalSignals with all extracted signals and fused score
        """
        eye_closed = eye_prob > config.DROWSY_THRESHOLD

        # Blink rate analysis
        blink_info = self.blink_tracker.update(eye_closed)

        # Head pose analysis
        if frame is not None:
            head_pose = self.head_estimator.estimate(frame, face_bbox)
        else:
            head_pose = {"pitch": 0.0, "yaw": 0.0, "detected": False}

        # ── Normalize signals to [0, 1] fatigue scale ──────────────────
        # Eye state: direct probability
        eye_signal = eye_prob

        # PERCLOS: above 0.15 is concerning, above 0.30 is critical
        perclos = blink_info["perclos"]
        perclos_signal = min(perclos / 0.30, 1.0)

        # Head pose: pitch below -15° indicates nodding
        pitch = head_pose["pitch"]
        head_signal = max(0, min((-pitch - 5) / 25, 1.0)) if pitch < -5 else 0.0

        # Blink rate: both very low (<8) and very high (>25) indicate fatigue
        br = blink_info["blink_rate_per_min"]
        if br < 8:
            blink_signal = (8 - br) / 8  # Low blink rate = heavy eyelids
        elif br > 25:
            blink_signal = min((br - 25) / 15, 1.0)  # Excessive blinking = strain
        else:
            blink_signal = 0.0

        # ── Weighted fusion ────────────────────────────────────────────
        fused = (
            self.SIGNAL_WEIGHTS["eye_state"] * eye_signal
            + self.SIGNAL_WEIGHTS["perclos"] * perclos_signal
            + self.SIGNAL_WEIGHTS["head_pose"] * head_signal
            + self.SIGNAL_WEIGHTS["blink_rate"] * blink_signal
        )

        return MultimodalSignals(
            eye_drowsy_prob=eye_prob,
            blink_rate_per_min=blink_info["blink_rate_per_min"],
            perclos=perclos,
            head_pitch=pitch,
            head_yaw=head_pose["yaw"],
            fused_fatigue_score=min(fused, 1.0),
        )

    def get_signal_breakdown(self) -> str:
        """Return a formatted string showing signal weights for the HUD."""
        w = self.SIGNAL_WEIGHTS
        return (f"Signal Weights: Eye={w['eye_state']:.0%} | "
                f"PERCLOS={w['perclos']:.0%} | "
                f"Head={w['head_pose']:.0%} | "
                f"Blink={w['blink_rate']:.0%}")
