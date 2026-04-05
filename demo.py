"""
Demo Recording Script — designed to produce a compelling portfolio video.

This script runs a polished webcam demo with all system features visible:
- Real-time eye detection + drowsy probability
- Fatigue score bar with color-coded state progression
- MC Dropout uncertainty indicator
- Multimodal signal panel (PERCLOS, blink rate, head pose)
- Alert banner with sound trigger
- Live FPS counter + session timer

Usage:
    python demo.py                     # Webcam demo (default)
    python demo.py --video input.mp4   # Run on video file
    python demo.py --save              # Auto-save to outputs/demo.mp4
    python demo.py --simulate          # Simulate drowsiness for demo recording
"""
import argparse
import os
import time
import numpy as np
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import config
from src.models.fatigue_tracker import FatigueTracker
from src.models.uncertainty import MCDropoutEstimator
from src.models.multimodal import MultimodalFatigueAssessor


# ── Color Palette ──────────────────────────────────────────────────────
COLORS = {
    "ALERT":             (0, 200, 0),
    "MILD_FATIGUE":      (0, 200, 200),
    "MODERATE_FATIGUE":  (0, 140, 255),
    "SEVERE_DROWSINESS": (0, 0, 255),
    "bg_dark":           (20, 20, 20),
    "bg_panel":          (30, 30, 40),
    "text_primary":      (240, 240, 240),
    "text_secondary":    (160, 160, 170),
    "bar_bg":            (60, 60, 70),
    "uncertainty_low":   (0, 200, 0),
    "uncertainty_med":   (0, 200, 200),
    "uncertainty_high":  (0, 0, 255),
    "accent":            (255, 180, 0),
}


class DemoRunner:
    """Full-featured demo with all system capabilities visible."""

    def __init__(self, model_path=None, mc_samples=15):
        model_path = model_path or str(config.MODEL_DIR / "drowsiness_detector.keras")

        if not os.path.exists(model_path):
            # Try improved model
            alt = str(config.MODEL_DIR / "drowsiness_detector_improved.keras")
            if os.path.exists(alt):
                model_path = alt
            else:
                print(f"ERROR: No model found. Run 'python train.py' first.")
                print(f"  Looked in: {model_path}")
                raise FileNotFoundError(model_path)

        self.model = tf.keras.models.load_model(model_path)
        print(f"  Model loaded: {model_path}")

        self.mc_estimator = MCDropoutEstimator(self.model, n_samples=mc_samples)
        self.tracker = FatigueTracker()
        self.multimodal = MultimodalFatigueAssessor()

        # Haar cascades
        haar = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(haar + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(haar + "haarcascade_eye.xml")

        # Stats
        self._fps_buf = []
        self._start_time = None

    def preprocess_eye(self, eye_roi):
        """Convert raw eye ROI to model input."""
        eye = cv2.resize(eye_roi, (config.MODEL_INPUT_SIZE, config.MODEL_INPUT_SIZE))
        if len(eye.shape) == 2:
            eye = cv2.cvtColor(eye, cv2.COLOR_GRAY2RGB)
        elif eye.shape[2] == 3:
            eye = cv2.cvtColor(cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
        return eye.astype(np.float32) / 255.0

    def process_frame(self, frame, use_uncertainty=True):
        """Process one frame through the full pipeline."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        eye_probs, eye_uncertainties = [], []
        eye_bboxes = []

        for (fx, fy, fw, fh) in faces:
            face_upper = gray[fy:fy+fh//2, fx:fx+fw]
            eyes = self.eye_cascade.detectMultiScale(face_upper, 1.1, 3, minSize=(20, 20))

            for (ex, ey, ew, eh) in eyes[:2]:
                eye_roi = face_upper[ey:ey+eh, ex:ex+ew]
                eye_input = self.preprocess_eye(eye_roi)
                eye_bboxes.append((fx+ex, fy+ey, ew, eh))

                if use_uncertainty:
                    unc = self.mc_estimator.predict_with_uncertainty(eye_input)
                    eye_probs.append(unc.mean_prob)
                    eye_uncertainties.append(unc.std_prob)
                else:
                    p = float(self.model.predict(np.expand_dims(eye_input, 0), verbose=0)[0, 0])
                    eye_probs.append(p)
                    eye_uncertainties.append(0.0)

        avg_prob = float(np.mean(eye_probs)) if eye_probs else 0.0
        avg_unc = float(np.mean(eye_uncertainties)) if eye_uncertainties else 0.0

        fatigue_result = self.tracker.update(avg_prob)
        multimodal = self.multimodal.assess(avg_prob, frame,
                                            faces[0] if len(faces) > 0 else None)

        return {
            "faces": faces,
            "eye_bboxes": eye_bboxes,
            "prob": avg_prob,
            "uncertainty": avg_unc,
            "fatigue": fatigue_result,
            "multimodal": multimodal,
            "n_eyes": len(eye_probs),
        }

    def draw_hud(self, frame, result):
        """Draw the full demo HUD overlay."""
        h, w = frame.shape[:2]
        fat = result["fatigue"]
        mm = result["multimodal"]
        state = fat.driver_state
        color = COLORS.get(state, COLORS["text_primary"])

        # ── Top panel background ───────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 170), COLORS["bg_panel"], -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # ── Row 1: State + Probability ─────────────────────────────────
        cv2.putText(frame, state.replace("_", " "),
                    (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        prob_text = f"P(drowsy): {result['prob']:.2f}"
        cv2.putText(frame, prob_text,
                    (w - 220, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    COLORS["text_secondary"], 1)

        # ── Row 2: Fatigue Score Bar ───────────────────────────────────
        bar_x, bar_y, bar_w, bar_h = 15, 48, w - 30, 22
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      COLORS["bar_bg"], -1)
        # State zone markers
        for sname, (lo, hi) in config.STATE_THRESHOLDS.items():
            x1 = bar_x + int(bar_w * lo)
            x2 = bar_x + int(bar_w * hi)
            c = COLORS.get(sname, (100, 100, 100))
            cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h),
                          tuple(max(0, v - 100) for v in c), -1)
        # Fill
        fill_w = int(bar_w * fat.fatigue_score)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      color, -1)
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (80, 80, 80), 1)
        # Label
        cv2.putText(frame, f"Fatigue: {fat.fatigue_score:.2f}",
                    (bar_x + 5, bar_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    COLORS["text_primary"], 1)

        # ── Row 3: Uncertainty Indicator ───────────────────────────────
        unc = result["uncertainty"]
        if unc < 0.05:
            unc_label, unc_color = "LOW", COLORS["uncertainty_low"]
        elif unc < 0.12:
            unc_label, unc_color = "MEDIUM", COLORS["uncertainty_med"]
        else:
            unc_label, unc_color = "HIGH", COLORS["uncertainty_high"]

        cv2.putText(frame, f"Uncertainty: {unc:.3f} [{unc_label}]",
                    (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, unc_color, 1)

        # Reliability badge
        if unc_label == "HIGH":
            cv2.putText(frame, "UNRELIABLE - CAUTION MODE",
                        (w - 310, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        COLORS["uncertainty_high"], 2)

        # ── Row 4: Multimodal Signals ──────────────────────────────────
        signals = [
            f"PERCLOS: {mm.perclos:.2f}",
            f"Blink/min: {mm.blink_rate_per_min}",
            f"Head pitch: {mm.head_pitch:.1f}deg",
            f"Fused: {mm.fused_fatigue_score:.2f}",
        ]
        for i, sig in enumerate(signals):
            x = 15 + i * (w // 4)
            cv2.putText(frame, sig, (x, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["text_secondary"], 1)

        # ── Row 5: Eyes detected + FPS + Timer ─────────────────────────
        eyes_text = f"Eyes: {result['n_eyes']}"
        cv2.putText(frame, eyes_text, (15, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["text_secondary"], 1)

        if self._fps_buf:
            fps = np.mean(self._fps_buf[-30:])
            cv2.putText(frame, f"FPS: {fps:.0f}",
                        (w - 100, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        COLORS["text_secondary"], 1)

        if self._start_time:
            elapsed = time.time() - self._start_time
            mins, secs = divmod(int(elapsed), 60)
            cv2.putText(frame, f"Session: {mins:02d}:{secs:02d}",
                        (w // 2 - 50, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        COLORS["text_secondary"], 1)

        # ── Face/Eye Bounding Boxes ────────────────────────────────────
        for (fx, fy, fw, fh) in result["faces"]:
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), color, 2)
        for (ex, ey, ew, eh) in result["eye_bboxes"]:
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), COLORS["accent"], 1)

        # ── Alert Banner ───────────────────────────────────────────────
        if fat.is_alert:
            # Flashing effect
            flash = int(time.time() * 4) % 2 == 0
            banner_color = (0, 0, 220) if flash else (0, 0, 180)
            cv2.rectangle(frame, (0, h - 65), (w, h), banner_color, -1)
            cv2.putText(frame, "DROWSINESS ALERT — PULL OVER",
                        (w // 2 - 250, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        return frame

    def run(self, source=0, save_path=None, simulate=False):
        """
        Run the demo.

        Args:
            source: 0 for webcam, or path to video file
            save_path: If set, save output video to this path
            simulate: If True, periodically simulate drowsiness by darkening
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"ERROR: Cannot open {'webcam' if source == 0 else source}")
            return

        self._start_time = time.time()
        self.tracker.reset()

        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
            print(f"  Recording to: {save_path}")

        print("\n  Demo running — press 'q' to quit")
        print("  Press 's' to take a screenshot")
        if simulate:
            print("  Simulation mode: auto-darkening every 10s to trigger drowsiness")

        frame_count = 0
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Simulation mode: periodically darken the frame to mimic drowsiness
            if simulate:
                elapsed = time.time() - self._start_time
                cycle = elapsed % 20  # 20-second cycle
                if 10 < cycle < 17:  # Dark phase for 7 seconds
                    darkness = 0.3 + 0.4 * ((cycle - 10) / 7)
                    frame = (frame * (1 - darkness)).astype(np.uint8)

            result = self.process_frame(frame, use_uncertainty=True)
            frame = self.draw_hud(frame, result)

            dt = time.perf_counter() - t0
            self._fps_buf.append(1.0 / max(dt, 1e-6))

            if writer:
                writer.write(frame)

            cv2.imshow("Driver Drowsiness Detection — Demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                ss_path = f"outputs/screenshot_{frame_count}.png"
                cv2.imwrite(ss_path, frame)
                print(f"  Screenshot saved: {ss_path}")

        cap.release()
        if writer:
            writer.release()
            print(f"\n  Video saved to: {save_path}")
        cv2.destroyAllWindows()

        # Print session summary
        summary = self.tracker.get_session_summary()
        print("\n  Session Summary:")
        for k, v in summary.items():
            if isinstance(v, dict):
                print(f"    {k}:")
                for sk, sv in v.items():
                    print(f"      {sk}: {sv:.2%}" if isinstance(sv, float) else f"      {sk}: {sv}")
            elif isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Drowsiness Detection Demo")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file (default: webcam)")
    parser.add_argument("--save", action="store_true",
                        help="Save output to outputs/demo.mp4")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model file")
    parser.add_argument("--simulate", action="store_true",
                        help="Simulate drowsiness for demo recording")
    parser.add_argument("--mc-samples", type=int, default=10,
                        help="MC Dropout samples (lower = faster, default 10)")
    args = parser.parse_args()

    print("=" * 50)
    print("  DROWSINESS DETECTION — DEMO MODE")
    print("=" * 50)

    runner = DemoRunner(model_path=args.model, mc_samples=args.mc_samples)

    source = args.video if args.video else 0
    save_path = "outputs/demo.mp4" if args.save else None

    runner.run(source=source, save_path=save_path, simulate=args.simulate)


if __name__ == "__main__":
    main()
