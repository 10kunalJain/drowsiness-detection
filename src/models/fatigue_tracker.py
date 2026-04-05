"""
Fatigue State Tracker — temporal modeling layer on top of frame-level predictions.

This is the innovation layer that elevates this project beyond basic classification:

1. **Temporal Smoothing**: Sliding window over raw predictions to filter noise
2. **Fatigue Score**: Exponentially-weighted cumulative drowsiness metric (0→1)
3. **Driver State Machine**: 4-level progression from ALERT → SEVERE_DROWSINESS
4. **Confidence Gating**: Ignore low-confidence predictions to reduce false alarms

This models the real-world insight that drowsiness is a *progressive state*,
not a binary flip. A single frame of closed eyes might be a blink; sustained
closure over multiple frames is genuine drowsiness.
"""
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


@dataclass
class FrameResult:
    """Result from processing a single frame."""
    frame_id: int
    raw_prob: float          # Raw model output (drowsy probability)
    smoothed_prob: float     # Temporally smoothed probability
    fatigue_score: float     # Cumulative fatigue metric
    driver_state: str        # Current state label
    is_alert: bool           # Whether to trigger an alert
    confidence: float        # Model confidence (distance from 0.5)


class FatigueTracker:
    """
    Stateful tracker that converts frame-level drowsy probabilities
    into a progressive fatigue assessment.
    """

    def __init__(
        self,
        window_size: int = config.WINDOW_SIZE,
        drowsy_threshold: float = config.DROWSY_THRESHOLD,
        fatigue_decay: float = config.FATIGUE_DECAY,
        fatigue_boost: float = config.FATIGUE_BOOST,
        alert_threshold: float = config.ALERT_FATIGUE_THRESHOLD,
        confidence_gate: float = 0.15,  # Ignore predictions within ±0.15 of 0.5
    ):
        self.window_size = window_size
        self.drowsy_threshold = drowsy_threshold
        self.fatigue_decay = fatigue_decay
        self.fatigue_boost = fatigue_boost
        self.alert_threshold = alert_threshold
        self.confidence_gate = confidence_gate

        self.reset()

    def reset(self):
        """Reset tracker state (e.g., between sessions)."""
        self._history = deque(maxlen=self.window_size)
        self._fatigue_score = 0.0
        self._frame_count = 0
        self._alert_active = False
        self._results_log: list[FrameResult] = []

    def update(self, drowsy_prob: float) -> FrameResult:
        """
        Process a new frame prediction and return the current driver assessment.

        Args:
            drowsy_prob: Raw sigmoid output from the model (0=natural, 1=drowsy)

        Returns:
            FrameResult with smoothed prediction, fatigue score, and state
        """
        self._frame_count += 1
        confidence = abs(drowsy_prob - 0.5) * 2  # Scale to 0-1

        # Confidence gating: only add high-confidence predictions to window
        if confidence >= self.confidence_gate:
            self._history.append(drowsy_prob)

        # Temporal smoothing: weighted moving average (recent frames matter more)
        if len(self._history) > 0:
            weights = np.exp(np.linspace(-1, 0, len(self._history)))
            weights /= weights.sum()
            smoothed = np.average(list(self._history), weights=weights)
        else:
            smoothed = drowsy_prob

        # Update fatigue score with exponential dynamics
        is_drowsy = smoothed > self.drowsy_threshold
        if is_drowsy:
            self._fatigue_score = min(1.0,
                self._fatigue_score * self.fatigue_decay + self.fatigue_boost)
        else:
            self._fatigue_score = max(0.0,
                self._fatigue_score * self.fatigue_decay)

        # Determine driver state from thresholds
        driver_state = self._get_state(self._fatigue_score)

        # Alert logic with hysteresis to prevent flickering
        if self._fatigue_score >= self.alert_threshold:
            self._alert_active = True
        elif self._fatigue_score < self.alert_threshold * 0.7:  # Hysteresis
            self._alert_active = False

        result = FrameResult(
            frame_id=self._frame_count,
            raw_prob=drowsy_prob,
            smoothed_prob=float(smoothed),
            fatigue_score=self._fatigue_score,
            driver_state=driver_state,
            is_alert=self._alert_active,
            confidence=confidence,
        )
        self._results_log.append(result)
        return result

    def _get_state(self, fatigue_score: float) -> str:
        """Map fatigue score to driver state."""
        for state, (low, high) in config.STATE_THRESHOLDS.items():
            if low <= fatigue_score < high:
                return state
        return "SEVERE_DROWSINESS"

    @property
    def fatigue_score(self) -> float:
        return self._fatigue_score

    @property
    def history(self) -> list[FrameResult]:
        return list(self._results_log)

    def get_session_summary(self) -> dict:
        """Generate summary statistics for the current session."""
        if not self._results_log:
            return {}

        probs = [r.smoothed_prob for r in self._results_log]
        fatigue_scores = [r.fatigue_score for r in self._results_log]
        states = [r.driver_state for r in self._results_log]

        return {
            "total_frames": len(self._results_log),
            "avg_drowsy_prob": np.mean(probs),
            "max_fatigue_score": max(fatigue_scores),
            "time_in_states": {
                state: states.count(state) / len(states)
                for state in config.STATE_THRESHOLDS
            },
            "alert_triggered_count": sum(1 for r in self._results_log if r.is_alert),
            "peak_fatigue_frame": fatigue_scores.index(max(fatigue_scores)),
        }
