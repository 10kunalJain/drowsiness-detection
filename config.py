"""
Central configuration for Driver Drowsiness Detection System.
All hyperparameters, paths, and system settings in one place.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "Drowsy_datset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"
ERROR_DIR = OUTPUT_DIR / "errors"

# ── Data ───────────────────────────────────────────────────────────────
IMG_SIZE = 48                   # Original image size
MODEL_INPUT_SIZE = 96           # 96x96 for pretrained backbone compatibility
NUM_CHANNELS = 3                # 3-channel for pretrained backbones
VAL_SPLIT = 0.15                # Hold out 15% of training data for validation
RANDOM_SEED = 42
USE_CLAHE = True                # Apply CLAHE histogram equalization
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = 4

# ── Classes ────────────────────────────────────────────────────────────
CLASS_NAMES = ["DROWSY", "NATURAL"]
NUM_CLASSES = 2

# ── Model ──────────────────────────────────────────────────────────────
MODEL_TYPE = "resnet50v2"       # "custom_cnn", "mobilenetv2", or "resnet50v2"

# ── Training ───────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 50                     # Increased from 30
LEARNING_RATE = 1e-3            # For classifier head (frozen backbone)
FINE_TUNE_LR = 1e-5             # For unfrozen backbone layers (MobileNetV2 only)
FINE_TUNE_AT_EPOCH = 15         # Unfreeze backbone after this epoch
EARLY_STOP_PATIENCE = 10        # More patience
LR_REDUCE_PATIENCE = 4
LR_REDUCE_FACTOR = 0.5
LABEL_SMOOTHING = 0.1           # Prevents overconfident predictions
MIXUP_ALPHA = 0.0               # Disabled — eye image interpolation is not meaningful

# ── Temporal Smoothing (Real-Time Inference) ───────────────────────────
WINDOW_SIZE = 15                # Sliding window for temporal smoothing (frames)
DROWSY_THRESHOLD = 0.6          # Probability threshold for drowsy classification
FATIGUE_DECAY = 0.95            # Exponential decay for fatigue score
FATIGUE_BOOST = 0.15            # Boost per drowsy frame
ALERT_FATIGUE_THRESHOLD = 0.7   # Trigger alert above this fatigue score

# ── Driver State Machine ──────────────────────────────────────────────
# States: ALERT → MILD_FATIGUE → MODERATE_FATIGUE → SEVERE_DROWSINESS
STATE_THRESHOLDS = {
    "ALERT":              (0.0, 0.3),
    "MILD_FATIGUE":       (0.3, 0.5),
    "MODERATE_FATIGUE":   (0.5, 0.7),
    "SEVERE_DROWSINESS":  (0.7, 1.0),
}

# ── Benchmarking ──────────────────────────────────────────────────────
BENCHMARK_MODELS = {
    "CustomCNN": {
        "type": "custom_cnn",
    },
    "MobileNetV2": {
        "type": "mobilenetv2",
    },
    "ResNet50V2": {
        "type": "resnet50v2",
    },
}
BENCHMARK_EPOCHS = 15            # More epochs for fair comparison

# ── Grad-CAM ──────────────────────────────────────────────────────────
GRADCAM_LAYER = "conv5_block3_out"  # Last conv block in ResNet50V2
NUM_GRADCAM_SAMPLES = 20
