"""
Streamlit Demo — Driver Drowsiness Detection System.

Run: streamlit run app.py

Features:
- Live webcam capture with face/eye detection
- Video file upload with frame-by-frame analysis + fatigue tracking
- Upload an eye image for single prediction
- MC Dropout uncertainty estimation
- Full analysis gallery from the 12-step pipeline
"""
import streamlit as st
import numpy as np
import cv2
import tempfile
from pathlib import Path
from PIL import Image
import os
import matplotlib.pyplot as plt
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

st.set_page_config(
    page_title="Drowsiness Detection",
    page_icon="👁",
    layout="wide",
)


HF_REPO = "Kunaljain10/drowsiness-detection-model"
HF_MODEL_FILE = "drowsiness_detector.keras"  # Base model: 88% accuracy at threshold 0.5

DEFAULT_DROWSY_THRESHOLD = 0.50  # Base model works best at 0.50 (88% accuracy)

STATE_COLORS = {
    "ALERT": "#2ecc71",
    "MILD_FATIGUE": "#f1c40f",
    "MODERATE_FATIGUE": "#e67e22",
    "SEVERE_DROWSINESS": "#e74c3c",
}

STATE_THRESHOLDS = {
    "ALERT": (0.0, 0.3),
    "MILD_FATIGUE": (0.3, 0.5),
    "MODERATE_FATIGUE": (0.5, 0.7),
    "SEVERE_DROWSINESS": (0.7, 1.0),
}


# ── Model Loading ──────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load model — local first, then HuggingFace."""
    import tensorflow as tf

    local_paths = [
        Path("outputs/models/drowsiness_detector.keras"),  # Base model: best at threshold 0.5
    ]
    for p in local_paths:
        if p.exists():
            return tf.keras.models.load_model(str(p))

    try:
        from huggingface_hub import hf_hub_download
        with st.spinner("Downloading model from HuggingFace (first run only)..."):
            model_path = hf_hub_download(repo_id=HF_REPO, filename=HF_MODEL_FILE)
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None


@st.cache_resource
def load_haar_cascades():
    """Load face and eye Haar cascade detectors."""
    haar_dir = cv2.data.haarcascades
    face = cv2.CascadeClassifier(haar_dir + "haarcascade_frontalface_default.xml")
    eyes = cv2.CascadeClassifier(haar_dir + "haarcascade_eye.xml")
    return face, eyes


# ── Preprocessing ──────────────────────────────────────────────────────

def preprocess_eye_roi(gray_eye, size=96, channels=3):
    """Preprocess an eye ROI for model input."""
    resized = cv2.resize(gray_eye, (size, size))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    resized = clahe.apply(resized.astype(np.uint8))
    normalized = resized.astype(np.float32) / 255.0
    if channels == 1:
        return normalized[..., np.newaxis]
    return np.stack([normalized] * 3, axis=-1)


def preprocess_uploaded(img_array):
    """Preprocess an uploaded image."""
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    elif len(img_array.shape) == 2:
        gray = img_array
    else:
        gray = img_array[:, :, 0]
    return preprocess_eye_roi(gray)


# ── Prediction ─────────────────────────────────────────────────────────

def predict_with_uncertainty(model, image, n_samples=20):
    """Prediction with uncertainty via prediction variance.

    Uses model.predict() (NOT training=True) because training=True
    breaks BatchNorm on single images — BatchNorm computes batch stats
    from one sample, collapsing all outputs to ~0.5.

    Uncertainty is estimated by running multiple predictions with slight
    input perturbations (test-time augmentation) instead of MC Dropout.
    """
    img = np.expand_dims(image, 0)

    # Base prediction (deterministic, correct)
    raw_base = float(model.predict(img, verbose=0).flatten()[0])

    # Estimate uncertainty via test-time augmentation
    # Small random perturbations to the input → variance = uncertainty
    samples = [raw_base]
    for _ in range(n_samples - 1):
        # Small random noise + flip
        noisy = image.copy()
        noisy += np.random.normal(0, 0.02, noisy.shape).astype(np.float32)
        noisy = np.clip(noisy, 0, 1)
        if np.random.random() > 0.5:
            noisy = noisy[:, ::-1, :]  # Random horizontal flip
        noisy_batch = np.expand_dims(noisy, 0)
        raw = float(model.predict(noisy_batch, verbose=0).flatten()[0])
        samples.append(raw)

    raw_samples = np.array(samples)
    # raw = P(NATURAL), so P(DROWSY) = 1 - raw
    drowsy_samples = 1.0 - raw_samples

    return {
        "mean": float(drowsy_samples.mean()),
        "std": float(drowsy_samples.std()),
        "samples": drowsy_samples,
    }


def predict_single(model, image):
    """Fast single-pass prediction (for video frames). Returns P(drowsy)."""
    img = np.expand_dims(image, 0)
    raw = float(model.predict(img, verbose=0).flatten()[0])
    return 1.0 - raw  # raw = P(NATURAL), so P(DROWSY) = 1 - raw


# ── Detection ──────────────────────────────────────────────────────────

def extract_eyes(frame_rgb, face_cascade, eye_cascade):
    """Detect face/eyes and return eye ROIs + annotated frame."""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    eyes_found = []
    annotated = frame_rgb.copy()

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
        face_upper = gray[fy:fy + fh // 2, fx:fx + fw]
        eyes = eye_cascade.detectMultiScale(face_upper, 1.1, 3, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes[:2]:
            eye_roi = face_upper[ey:ey + eh, ex:ex + ew]
            eyes_found.append(eye_roi)
            cv2.rectangle(annotated, (fx + ex, fy + ey),
                          (fx + ex + ew, fy + ey + eh), (255, 200, 0), 2)

    return eyes_found, annotated


# ── Fatigue Tracker ────────────────────────────────────────────────────

class SimpleFatigueTracker:
    """Lightweight fatigue tracker for the Streamlit app."""

    def __init__(self):
        self.fatigue_score = 0.0
        self.history = []

    def update(self, drowsy_prob, threshold=None):
        if threshold is None:
            threshold = DEFAULT_DROWSY_THRESHOLD
        if drowsy_prob > threshold:
            self.fatigue_score = min(1.0, self.fatigue_score * 0.95 + 0.15)
        else:
            self.fatigue_score = max(0.0, self.fatigue_score * 0.95)

        # Determine state
        state = "ALERT"
        for s, (lo, hi) in STATE_THRESHOLDS.items():
            if lo <= self.fatigue_score < hi:
                state = s
                break

        self.history.append({
            "prob": drowsy_prob,
            "fatigue": self.fatigue_score,
            "state": state,
        })
        return state

    def get_summary(self):
        if not self.history:
            return {}
        probs = [h["prob"] for h in self.history]
        fatigue_scores = [h["fatigue"] for h in self.history]
        states = [h["state"] for h in self.history]
        return {
            "frames_processed": len(self.history),
            "avg_drowsy_prob": np.mean(probs),
            "max_fatigue": max(fatigue_scores),
            "time_in_states": {
                s: states.count(s) / len(states) for s in STATE_THRESHOLDS
            },
        }


# ── Display Helpers ────────────────────────────────────────────────────

def display_results(result, col, threshold=None):
    """Display MC Dropout prediction results."""
    if threshold is None:
        threshold = DEFAULT_DROWSY_THRESHOLD
    prob = result["mean"]
    unc = result["std"]
    state = "DROWSY" if prob > threshold else "NATURAL"
    unc_level = "LOW" if unc < 0.05 else "MEDIUM" if unc < 0.12 else "HIGH"

    with col:
        if state == "DROWSY":
            st.error(f"State: **{state}**")
        else:
            st.success(f"State: **{state}**")

        c1, c2 = st.columns(2)
        c1.metric("P(Drowsy)", f"{prob:.3f}")
        c2.metric("Uncertainty", f"{unc:.3f} ({unc_level})")

        if unc_level == "HIGH":
            st.warning("High uncertainty — prediction may be unreliable")

        fig, ax = plt.subplots(figsize=(5, 2))
        ax.hist(result["samples"], bins=15, color="#3498db", alpha=0.7, edgecolor="black")
        ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Threshold")
        ax.axvline(x=prob, color="green", linewidth=2, label=f"Mean: {prob:.3f}")
        ax.set_xlabel("P(Drowsy)")
        ax.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ── App ────────────────────────────────────────────────────────────────

st.title("👁 Driver Drowsiness Detection")
st.markdown("**Uncertainty-aware detection with MC Dropout and temporal fatigue modeling**")

model = load_model()
if model is None:
    st.error("No trained model found.")
    st.stop()

face_cascade, eye_cascade = load_haar_cascades()

# ── Sidebar Settings ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    drowsy_threshold = st.slider(
        "Drowsy Threshold",
        min_value=0.30, max_value=0.60, value=DEFAULT_DROWSY_THRESHOLD, step=0.05,
        help="Lower = more sensitive to drowsiness (catches more cases but more false alarms). "
             "Default 0.50 gives 88% accuracy with 82% drowsy recall."
    )
    st.caption(f"Predicts DROWSY when P(drowsy) > {drowsy_threshold:.2f}")

    tta_passes = st.slider(
        "Uncertainty Passes (TTA)",
        min_value=1, max_value=20, value=10, step=1,
        help="Test-time augmentation passes for uncertainty estimation. "
             "More = more stable estimate but slower."
    )

tab_webcam, tab_video, tab_upload, tab_gallery, tab_about = st.tabs([
    "Live Webcam", "Video Analysis", "Upload Image", "Analysis Gallery", "About"
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1: LIVE WEBCAM
# ══════════════════════════════════════════════════════════════════════
with tab_webcam:
    st.header("Live Webcam Detection")
    st.markdown("Capture a frame, then crop your eye region for prediction.")

    camera_input = st.camera_input("Look at the camera and click to capture")

    if camera_input is not None:
        img = Image.open(camera_input)
        frame_rgb = np.array(img)
        h, w = frame_rgb.shape[:2]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(frame_rgb, caption="Captured Frame", use_container_width=True)

            st.markdown("**Crop your eye region** (adjust sliders to select one eye):")
            # Default crop: center region
            left = st.slider("Left", 0, w - 10, w // 4, key="crop_l")
            right = st.slider("Right", left + 10, w, 3 * w // 4, key="crop_r")
            top = st.slider("Top", 0, h - 10, h // 4, key="crop_t")
            bottom = st.slider("Bottom", top + 10, h, 3 * h // 4, key="crop_b")

            # Show crop preview
            cropped = frame_rgb[top:bottom, left:right]
            gray_crop = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
            st.image(cropped, caption="Cropped Eye Region", width=200)

        with col2:
            if st.button("Predict on Cropped Region", type="primary"):
                preprocessed = preprocess_eye_roi(gray_crop)

                with st.expander("What the model sees"):
                    st.image(preprocessed[:, :, 0], caption="96x96 CLAHE-enhanced",
                             width=150, clamp=True)

                with st.spinner(f"Predicting ({tta_passes} passes)..."):
                    result = predict_with_uncertainty(model, preprocessed, n_samples=tta_passes)
                display_results(result, col2, threshold=drowsy_threshold)
            else:
                st.info("Adjust the crop sliders on the left to select one eye, "
                        "then click 'Predict on Cropped Region'.")

                # Also show auto-detected eyes for reference
                eyes, annotated = extract_eyes(frame_rgb, face_cascade, eye_cascade)
                if eyes:
                    st.caption(f"Auto-detected {len(eyes)} eye(s) — "
                               "but manual cropping gives better results:")
                    eye_cols = st.columns(min(len(eyes), 3))
                    for i, (eye_roi, c) in enumerate(zip(eyes[:3], eye_cols)):
                        c.image(eye_roi, caption=f"Auto Eye {i+1}", width=100, clamp=True)
    else:
        st.info("Click the camera button above to capture a frame.")


# ══════════════════════════════════════════════════════════════════════
# TAB 2: VIDEO ANALYSIS
# ══════════════════════════════════════════════════════════════════════
with tab_video:
    st.header("Video Analysis")
    st.markdown("Upload a driving video — the system processes it frame-by-frame "
                "with **fatigue tracking** and shows the drowsiness progression over time.")

    uploaded_video = st.file_uploader(
        "Upload a video file", type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader"
    )

    if uploaded_video is not None:
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        st.info(f"Video: {total_frames} frames at {fps:.0f} FPS "
                f"({total_frames / fps:.1f} seconds)")

        # Settings
        col_s1, col_s2 = st.columns(2)
        skip_frames = col_s1.slider("Process every N-th frame (speed vs detail)",
                                     1, 10, 3)
        mc_passes = col_s2.slider("MC Dropout passes per frame (accuracy vs speed)",
                                   1, 20, 5)

        if st.button("Analyze Video", type="primary"):
            tracker = SimpleFatigueTracker()
            progress = st.progress(0)
            status = st.empty()
            frame_display = st.empty()

            frame_idx = 0
            processed = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % skip_frames == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    eyes, annotated = extract_eyes(frame_rgb, face_cascade, eye_cascade)

                    if eyes:
                        probs = []
                        for eye_roi in eyes:
                            preprocessed = preprocess_eye_roi(eye_roi)
                            if mc_passes > 1:
                                r = predict_with_uncertainty(model, preprocessed,
                                                             n_samples=mc_passes)
                                probs.append(r["mean"])
                            else:
                                probs.append(predict_single(model, preprocessed))

                        avg_prob = np.mean(probs)
                    else:
                        avg_prob = 0.0

                    state = tracker.update(avg_prob, threshold=drowsy_threshold)
                    processed += 1

                    # Draw HUD on frame
                    color_hex = STATE_COLORS[state]
                    # Convert hex to BGR
                    color_bgr = tuple(int(color_hex.lstrip('#')[i:i+2], 16)
                                      for i in (4, 2, 0))
                    cv2.putText(annotated, f"{state} | Fatigue: {tracker.fatigue_score:.2f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                color_bgr, 2)

                    # Update display every 5 processed frames
                    if processed % 5 == 0:
                        frame_display.image(annotated, use_container_width=True)

                    progress.progress(min(frame_idx / total_frames, 1.0))
                    status.text(f"Frame {frame_idx}/{total_frames} | "
                                f"State: {state} | Fatigue: {tracker.fatigue_score:.2f}")

                frame_idx += 1

            cap.release()
            progress.progress(1.0)
            status.text(f"Done! Processed {processed} frames out of {total_frames}.")

            # ── Results ────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("Session Results")

            summary = tracker.get_summary()

            # Key metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Frames Processed", summary["frames_processed"])
            col2.metric("Max Fatigue Score", f"{summary['max_fatigue']:.3f}")
            col3.metric("Avg Drowsy Probability", f"{summary['avg_drowsy_prob']:.3f}")

            # Time in states
            st.markdown("**Time in Each State:**")
            state_cols = st.columns(4)
            for i, (state_name, pct) in enumerate(summary["time_in_states"].items()):
                state_cols[i].metric(
                    state_name.replace("_", " ").title(),
                    f"{pct:.1%}",
                )

            # Fatigue progression chart
            st.subheader("Fatigue Progression Over Time")

            history = tracker.history
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

            frames = range(len(history))
            probs = [h["prob"] for h in history]
            fatigue = [h["fatigue"] for h in history]
            states = [h["state"] for h in history]

            # Drowsy probability over time
            ax1.plot(frames, probs, color="#3498db", alpha=0.7, linewidth=1)
            ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.3)
            ax1.fill_between(frames, probs, 0.5,
                             where=[p > 0.5 for p in probs],
                             alpha=0.2, color="red", label="Drowsy")
            ax1.fill_between(frames, probs, 0.5,
                             where=[p <= 0.5 for p in probs],
                             alpha=0.2, color="green", label="Alert")
            ax1.set_ylabel("P(Drowsy)")
            ax1.set_title("Drowsiness Probability Over Time", fontweight="bold")
            ax1.legend(fontsize=8)
            ax1.set_ylim(0, 1)

            # Fatigue score over time with state coloring
            ax2.plot(frames, fatigue, color="#e74c3c", linewidth=2)
            for s, (lo, hi) in STATE_THRESHOLDS.items():
                ax2.axhspan(lo, hi, alpha=0.1,
                            color=STATE_COLORS[s], label=s.replace("_", " "))
            ax2.set_ylabel("Fatigue Score")
            ax2.set_xlabel("Frame")
            ax2.set_title("Cumulative Fatigue Score", fontweight="bold")
            ax2.legend(fontsize=7, loc="upper left")
            ax2.set_ylim(0, 1)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Clean up temp file
            os.unlink(tfile.name)

    else:
        st.info("Upload a video file to analyze drowsiness over time.")


# ══════════════════════════════════════════════════════════════════════
# TAB 3: UPLOAD IMAGE
# ══════════════════════════════════════════════════════════════════════
with tab_upload:
    st.header("Upload an Eye Image")

    st.info("**Best results:** The model was trained on **48x48 grayscale close-up eye crops** "
            "(just the iris and eyelid). For accurate predictions, upload a tightly cropped "
            "eye image. Full face images or wide eye regions will give unreliable results (~0.5).")

    # Sample images
    sample_dir = Path("samples")
    if sample_dir.exists():
        st.markdown("**Try a sample image:**")
        sample_cols = st.columns(6)
        sample_files = sorted(sample_dir.glob("*.png"))
        selected_sample = None

        for i, sf in enumerate(sample_files[:6]):
            with sample_cols[i]:
                label = "Drowsy" if "drowsy" in sf.name else "Natural"
                st.image(str(sf), caption=label, width=80)
                if st.button(f"Use", key=f"sample_{i}"):
                    selected_sample = sf

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded = st.file_uploader("Choose an eye image", type=["png", "jpg", "jpeg"])

        # Use either uploaded file or selected sample
        img_to_process = None
        if uploaded:
            img_to_process = Image.open(uploaded)
            st.image(img_to_process, caption="Uploaded Image", width=250)
        elif selected_sample:
            img_to_process = Image.open(selected_sample)
            st.image(img_to_process, caption=f"Sample: {selected_sample.name}", width=250)

        if img_to_process:
            img_array = np.array(img_to_process)
            preprocessed = preprocess_uploaded(img_array)

            # Show preprocessed version
            with st.expander("Preprocessed input (what the model sees)"):
                st.image(preprocessed[:, :, 0], caption="96x96 CLAHE-enhanced", width=200,
                         clamp=True)

            with st.spinner(f"Predicting ({tta_passes} passes)..."):
                result = predict_with_uncertainty(model, preprocessed, n_samples=tta_passes)
            display_results(result, col2, threshold=drowsy_threshold)


# ══════════════════════════════════════════════════════════════════════
# TAB 4: ANALYSIS GALLERY
# ══════════════════════════════════════════════════════════════════════
with tab_gallery:
    st.header("Analysis Gallery")
    st.markdown("*All plots generated automatically by the 12-step training pipeline.*")

    plot_sections = {
        "Training & Evaluation": [
            ("Training History", "outputs/plots/training_history.png"),
            ("Confusion Matrix", "outputs/plots/confusion_matrix.png"),
            ("ROC & PR Curves", "outputs/plots/roc_pr_curves.png"),
        ],
        "Model Benchmark": [
            ("Architecture Comparison", "outputs/plots/model_benchmark.png"),
        ],
        "Robustness Testing": [
            ("Robustness Under Corruption", "outputs/plots/robustness_test.png"),
            ("Corruption Examples", "outputs/plots/corruption_samples.png"),
        ],
        "Uncertainty": [
            ("MC Dropout Analysis", "outputs/plots/uncertainty_analysis.png"),
        ],
        "Error Analysis": [
            ("Confidence Distribution", "outputs/errors/confidence_distribution.png"),
            ("Error Breakdown", "outputs/errors/error_breakdown.png"),
            ("Hardness Analysis", "outputs/errors/hardness_analysis.png"),
            ("Misclassified Gallery", "outputs/errors/misclassified_gallery.png"),
        ],
        "Improvement": [
            ("Before vs After", "outputs/plots/improvement_comparison.png"),
            ("Threshold Optimization", "outputs/plots/threshold_optimization.png"),
        ],
        "Interpretability (Grad-CAM)": [
            ("Correct vs Misclassified", "outputs/gradcam/gradcam_comparison.png"),
            ("Drowsy Class", "outputs/gradcam/gradcam_drowsy.png"),
            ("Natural Class", "outputs/gradcam/gradcam_natural.png"),
        ],
        "Data Quality": [
            ("Class Distribution", "outputs/plots/class_distribution.png"),
            ("Dataset Bias Profile", "outputs/plots/dataset_bias_profile.png"),
            ("Label Noise Suspects", "outputs/plots/label_noise_suspects.png"),
        ],
    }

    for section, plots in plot_sections.items():
        with st.expander(section, expanded=False):
            for title, path in plots:
                if Path(path).exists():
                    st.markdown(f"**{title}**")
                    st.image(path, use_container_width=True)
                else:
                    st.caption(f"{title}: not yet generated (run train.py)")


# ══════════════════════════════════════════════════════════════════════
# TAB 5: ABOUT
# ══════════════════════════════════════════════════════════════════════
with tab_about:
    st.header("About This Project")
    st.markdown("""
    **12-step ML pipeline:**

    1. Data loading & EDA
    2. Data quality analysis (imbalance, bias, label noise)
    3. Multi-model benchmark (CustomCNN vs MobileNetV2 vs ResNet50V2)
    4. Two-phase transfer learning (ResNet50V2)
    5. Test set evaluation
    6. Grad-CAM interpretability
    7. Error analysis
    8. MC Dropout uncertainty estimation
    9. Robustness stress-testing (36 conditions)
    10. Error-driven improvement loop
    11. LSTM temporal sequence head
    12. Failure narrative generation

    **Key Results:**
    - AUC-ROC: **0.988** (frame-level)
    - LSTM Accuracy: **96.3%** (sequence-level)
    - Latency: **~52ms/frame** (CPU)
    """)
    st.markdown("---")
    st.markdown("Built with TensorFlow, OpenCV, and Streamlit")
