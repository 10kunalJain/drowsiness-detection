"""
Production API Abstraction Layer.

Clean, documented interface for the drowsiness detection system.
This is what an integration engineer or downstream service would use —
no ML internals leak through.

Usage:
    from src.api import DrowsinessAPI

    api = DrowsinessAPI()

    # Single image prediction
    result = api.predict_eye(eye_image)
    # → {"state": "DROWSY", "probability": 0.87, "fatigue_score": 0.65,
    #    "uncertainty": 0.04, "reliable": True, "driver_state": "MODERATE_FATIGUE"}

    # Video frame (full face)
    result = api.predict_frame(frame)
    # → same as above, plus face/eye bounding boxes

    # Session management
    api.start_session()
    for frame in video:
        result = api.predict_frame(frame)
    summary = api.end_session()
"""
import numpy as np
import tensorflow as tf
import cv2
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.models.fatigue_tracker import FatigueTracker
from src.models.uncertainty import MCDropoutEstimator


@dataclass
class PredictionResult:
    """Structured output from the drowsiness detection system."""
    # Core prediction
    state: str                    # "DROWSY" or "NATURAL"
    probability: float            # P(drowsy) from model
    confidence: float             # Model confidence (0-1)

    # Uncertainty estimation
    uncertainty: float            # Epistemic uncertainty (MC Dropout std)
    reliable: bool                # Whether to trust this prediction

    # Temporal context (only available during sessions)
    fatigue_score: float          # Cumulative fatigue metric (0-1)
    driver_state: str             # ALERT / MILD_FATIGUE / MODERATE_FATIGUE / SEVERE_DROWSINESS
    alert_active: bool            # Whether drowsiness alert is triggered

    # Metadata
    eyes_detected: int            # Number of eyes found in frame
    processing_time_ms: float     # End-to-end latency

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        return (f"PredictionResult(state={self.state}, prob={self.probability:.2f}, "
                f"fatigue={self.fatigue_score:.2f}, uncertainty={self.uncertainty:.3f}, "
                f"driver_state={self.driver_state})")


class DrowsinessAPI:
    """
    Production-ready API for driver drowsiness detection.

    Combines:
    - MobileNetV2 eye-state classifier
    - MC Dropout uncertainty estimation
    - Temporal fatigue tracking
    - Face/eye detection pipeline

    Thread-safety: NOT thread-safe (fatigue tracker has state).
    Create one instance per driver/camera stream.
    """

    def __init__(self, model_path: str = None, enable_uncertainty: bool = True,
                 mc_samples: int = 15):
        """
        Initialize the drowsiness detection API.

        Args:
            model_path: Path to .keras model file (uses default if None)
            enable_uncertainty: Enable MC Dropout uncertainty estimation
            mc_samples: Number of MC forward passes (higher = more accurate but slower)
        """
        if model_path is None:
            model_path = str(config.MODEL_DIR / "drowsiness_detector.keras")

        self._model = tf.keras.models.load_model(model_path)

        self._uncertainty_enabled = enable_uncertainty
        if enable_uncertainty:
            self._mc_estimator = MCDropoutEstimator(self._model, n_samples=mc_samples)

        self._tracker = FatigueTracker()
        self._session_active = False

        # Haar cascades for face/eye detection
        haar_dir = cv2.data.haarcascades
        self._face_cascade = cv2.CascadeClassifier(
            haar_dir + "haarcascade_frontalface_default.xml"
        )
        self._eye_cascade = cv2.CascadeClassifier(
            haar_dir + "haarcascade_eye.xml"
        )

    # ── Core API ───────────────────────────────────────────────────────

    def predict_eye(self, eye_image: np.ndarray) -> PredictionResult:
        """
        Predict drowsiness from a single eye image.

        Args:
            eye_image: Eye crop — grayscale or RGB, any size
                       (will be resized and normalized internally)

        Returns:
            PredictionResult with prediction, uncertainty, and fatigue state
        """
        import time
        t0 = time.perf_counter()

        preprocessed = self._preprocess(eye_image)

        # Get prediction with uncertainty
        if self._uncertainty_enabled:
            unc_result = self._mc_estimator.predict_with_uncertainty(preprocessed)
            prob = unc_result.mean_prob
            uncertainty = unc_result.std_prob
            reliable = unc_result.is_reliable
            confidence = unc_result.confidence
        else:
            prob = float(self._model.predict(
                np.expand_dims(preprocessed, 0), verbose=0
            ).flatten()[0])
            uncertainty = 0.0
            reliable = True
            confidence = abs(prob - 0.5) * 2

        # Update fatigue tracker
        fatigue_result = self._tracker.update(prob)

        elapsed = (time.perf_counter() - t0) * 1000

        return PredictionResult(
            state=config.CLASS_NAMES[int(prob > 0.5)],
            probability=round(prob, 4),
            confidence=round(confidence, 4),
            uncertainty=round(uncertainty, 4),
            reliable=reliable,
            fatigue_score=round(fatigue_result.fatigue_score, 4),
            driver_state=fatigue_result.driver_state,
            alert_active=fatigue_result.is_alert,
            eyes_detected=1,
            processing_time_ms=round(elapsed, 2),
        )

    def predict_frame(self, frame: np.ndarray) -> PredictionResult:
        """
        Predict drowsiness from a full video frame (face → eyes → classify).

        Args:
            frame: BGR video frame from camera

        Returns:
            PredictionResult (averaged across detected eyes)
        """
        import time
        t0 = time.perf_counter()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        eye_probs = []
        eye_uncertainties = []

        for (x, y, w, h) in faces:
            face_upper = gray[y:y+h//2, x:x+w]
            eyes = self._eye_cascade.detectMultiScale(
                face_upper, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
            )
            for (ex, ey, ew, eh) in eyes[:2]:
                eye_roi = face_upper[ey:ey+eh, ex:ex+ew]
                preprocessed = self._preprocess(eye_roi)

                if self._uncertainty_enabled:
                    unc = self._mc_estimator.predict_with_uncertainty(preprocessed)
                    eye_probs.append(unc.mean_prob)
                    eye_uncertainties.append(unc.std_prob)
                else:
                    p = float(self._model.predict(
                        np.expand_dims(preprocessed, 0), verbose=0
                    ).flatten()[0])
                    eye_probs.append(p)
                    eye_uncertainties.append(0.0)

        # Average across eyes
        if eye_probs:
            prob = float(np.mean(eye_probs))
            uncertainty = float(np.mean(eye_uncertainties))
        else:
            prob = 0.0
            uncertainty = 0.0

        reliable = uncertainty < 0.12
        confidence = max(0, 1 - uncertainty / 0.5)

        fatigue_result = self._tracker.update(prob)
        elapsed = (time.perf_counter() - t0) * 1000

        return PredictionResult(
            state=config.CLASS_NAMES[int(prob > 0.5)],
            probability=round(prob, 4),
            confidence=round(confidence, 4),
            uncertainty=round(uncertainty, 4),
            reliable=reliable,
            fatigue_score=round(fatigue_result.fatigue_score, 4),
            driver_state=fatigue_result.driver_state,
            alert_active=fatigue_result.is_alert,
            eyes_detected=len(eye_probs),
            processing_time_ms=round(elapsed, 2),
        )

    # ── Session Management ─────────────────────────────────────────────

    def start_session(self):
        """Start a new monitoring session (resets fatigue tracker)."""
        self._tracker.reset()
        self._session_active = True

    def end_session(self) -> dict:
        """End session and return summary statistics."""
        self._session_active = False
        return self._tracker.get_session_summary()

    # ── Internal ───────────────────────────────────────────────────────

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Normalize any eye image to model input format."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (config.MODEL_INPUT_SIZE, config.MODEL_INPUT_SIZE))
        image = image.astype(np.float32) / 255.0
        image = np.stack([image] * 3, axis=-1)  # Grayscale → RGB
        return image
