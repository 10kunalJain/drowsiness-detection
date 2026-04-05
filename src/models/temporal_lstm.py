"""
LSTM Temporal Sequence Head — learned temporal modeling.

Instead of hand-crafted smoothing (sliding window, exponential decay),
this module LEARNS the temporal patterns that distinguish:
- A blink (2-3 frames of closed eyes) from drowsiness (sustained closure)
- Progressive drowsiness onset from sudden changes
- Normal eye movement patterns from fatigue-related patterns

Architecture:
    MobileNetV2 (frozen) → features per frame → LSTM → fatigue prediction

The key insight: a sequence of [0.8, 0.2, 0.8, 0.2] (flickering) has a
very different meaning than [0.3, 0.5, 0.7, 0.9] (progressive onset).
A sliding average treats them similarly; an LSTM can learn the difference.

Training approach:
- Generate synthetic sequences from the frame-level dataset
- Use the trained CNN to extract features, then train only the LSTM head
- This avoids needing temporally-annotated video data
"""
import numpy as np
import tensorflow as tf
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


# ── Sequence Generation ───────────────────────────────────────────────

def generate_synthetic_sequences(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    seq_length: int = 15,
    n_sequences: int = 2000,
    seed: int = config.RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic temporal sequences from individual frames.

    Strategy: simulate realistic driving sequences by sampling frames with
    temporal structure (not random), creating patterns like:
    - Sustained alert (all natural)
    - Sustained drowsy (all drowsy)
    - Transition: alert → drowsy (progressive onset)
    - Blink pattern: short drowsy burst within alert sequence
    - Recovery: drowsy → alert

    Each sequence gets a label based on its drowsy frame ratio:
    - < 30% drowsy frames → "alert" (0)
    - ≥ 30% drowsy frames → "drowsy" (1)

    The LSTM should learn that the PATTERN matters, not just the ratio.
    """
    rng = np.random.RandomState(seed)

    drowsy_idx = np.where(y == 0)[0]  # DROWSY class indices
    natural_idx = np.where(y == 1)[0]  # NATURAL class indices

    # Extract CNN features for all images (after GAP + BatchNorm, before classifier)
    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("bn_head").output,
    )
    features = feature_extractor.predict(X, batch_size=config.BATCH_SIZE, verbose=0)
    feature_dim = features.shape[-1]

    sequences = []
    labels = []

    patterns = [
        "sustained_alert",
        "sustained_drowsy",
        "onset",       # alert → drowsy transition
        "blink",       # short drowsy burst in alert
        "recovery",    # drowsy → alert
        "mixed",       # random mix
    ]

    per_pattern = n_sequences // len(patterns)

    for pattern in patterns:
        for _ in range(per_pattern):
            seq = np.zeros((seq_length, feature_dim))

            if pattern == "sustained_alert":
                # All natural frames
                idxs = rng.choice(natural_idx, seq_length, replace=True)
                seq_label = 0  # Not drowsy

            elif pattern == "sustained_drowsy":
                # All drowsy frames
                idxs = rng.choice(drowsy_idx, seq_length, replace=True)
                seq_label = 1  # Drowsy

            elif pattern == "onset":
                # First half alert, second half drowsy (progressive)
                split = rng.randint(seq_length // 3, 2 * seq_length // 3)
                alert_part = rng.choice(natural_idx, split, replace=True)
                drowsy_part = rng.choice(drowsy_idx, seq_length - split, replace=True)
                idxs = np.concatenate([alert_part, drowsy_part])
                seq_label = 1 if (seq_length - split) / seq_length > 0.3 else 0

            elif pattern == "blink":
                # Mostly alert with a 2-4 frame drowsy burst (blink)
                idxs = rng.choice(natural_idx, seq_length, replace=True)
                blink_start = rng.randint(2, seq_length - 4)
                blink_len = rng.randint(2, 5)
                blink_idxs = rng.choice(drowsy_idx, blink_len, replace=True)
                idxs[blink_start:blink_start + blink_len] = blink_idxs[:min(blink_len, seq_length - blink_start)]
                seq_label = 0  # Blinks are NOT drowsiness

            elif pattern == "recovery":
                # First half drowsy, second half alert
                split = rng.randint(seq_length // 3, 2 * seq_length // 3)
                drowsy_part = rng.choice(drowsy_idx, split, replace=True)
                alert_part = rng.choice(natural_idx, seq_length - split, replace=True)
                idxs = np.concatenate([drowsy_part, alert_part])
                seq_label = 0  # Recovering → not currently drowsy

            else:  # mixed
                n_drowsy = rng.randint(0, seq_length)
                drowsy_frames = rng.choice(drowsy_idx, n_drowsy, replace=True)
                alert_frames = rng.choice(natural_idx, seq_length - n_drowsy, replace=True)
                idxs = np.concatenate([drowsy_frames, alert_frames])
                rng.shuffle(idxs)
                seq_label = 1 if n_drowsy / seq_length > 0.5 else 0

            seq = features[idxs]
            sequences.append(seq)
            labels.append(seq_label)

    return np.array(sequences), np.array(labels)


# ── LSTM Model ────────────────────────────────────────────────────────

def build_temporal_model(
    feature_dim: int,
    seq_length: int = 15,
    lstm_units: int = 64,
) -> tf.keras.Model:
    """
    Build LSTM temporal head for sequence-level drowsiness prediction.

    Architecture:
        Input (seq_length, feature_dim) → LSTM(64) → Dense(32) → Sigmoid

    This is intentionally lightweight — the CNN backbone does the heavy
    feature extraction; the LSTM just learns temporal patterns.
    """
    inputs = tf.keras.Input(shape=(seq_length, feature_dim), name="feature_sequence")

    x = tf.keras.layers.LSTM(
        lstm_units,
        return_sequences=False,  # Only need final hidden state
        dropout=0.3,
        recurrent_dropout=0.2,
        name="temporal_lstm",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn_temporal")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="fc_temporal")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_temporal")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="drowsy_sequence")(x)

    model = tf.keras.Model(inputs, outputs, name="TemporalDrowsinessHead")
    return model


def train_temporal_model(
    cnn_model: tf.keras.Model,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    seq_length: int = 15,
    n_sequences: int = 3000,
    epochs: int = 20,
) -> tuple[tf.keras.Model, dict]:
    """
    Train the LSTM temporal head.

    Steps:
    1. Generate synthetic sequences from frame-level data
    2. Build lightweight LSTM model
    3. Train on sequences
    4. Evaluate temporal vs frame-level performance
    """
    print("\n" + "=" * 60)
    print("TRAINING LSTM TEMPORAL HEAD")
    print("=" * 60)

    # Generate sequences for train and validation
    print("  Generating synthetic sequences...")
    X_seq_train, y_seq_train = generate_synthetic_sequences(
        cnn_model, X_train, y_train,
        seq_length=seq_length, n_sequences=n_sequences, seed=42,
    )
    X_seq_val, y_seq_val = generate_synthetic_sequences(
        cnn_model, X_val, y_val,
        seq_length=seq_length, n_sequences=n_sequences // 3, seed=99,
    )

    feature_dim = X_seq_train.shape[-1]
    print(f"  Sequence shape: {X_seq_train.shape}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Train sequences: {len(X_seq_train)} (drowsy: {y_seq_train.sum()})")
    print(f"  Val sequences: {len(X_seq_val)}")

    # Build and train
    temporal_model = build_temporal_model(feature_dim, seq_length)
    temporal_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    temporal_model.summary()

    history = temporal_model.fit(
        X_seq_train, y_seq_train,
        validation_data=(X_seq_val, y_seq_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=5,
                restore_best_weights=True, mode="max"
            ),
        ],
        verbose=1,
    )

    # Save
    save_path = config.MODEL_DIR / "temporal_lstm.keras"
    temporal_model.save(save_path)
    print(f"  Saved temporal model: {save_path}")

    # Evaluate
    val_metrics = temporal_model.evaluate(X_seq_val, y_seq_val, verbose=0)
    print(f"\n  Temporal LSTM Results:")
    print(f"    Val Accuracy: {val_metrics[1]:.4f}")
    print(f"    Val AUC:      {val_metrics[2]:.4f}")

    return temporal_model, history.history


class TemporalPredictor:
    """
    Real-time temporal predictor that buffers CNN features
    and runs the LSTM when the buffer is full.

    Usage in inference loop:
        predictor = TemporalPredictor(cnn_model, lstm_model)
        for frame in video:
            eye_image = extract_eye(frame)
            result = predictor.predict(eye_image)
            # result is None until buffer fills, then gives temporal prediction
    """

    def __init__(self, cnn_model: tf.keras.Model,
                 lstm_model: tf.keras.Model,
                 seq_length: int = 15):
        self.seq_length = seq_length

        # Build feature extractor from CNN (after GAP + BatchNorm)
        self._feature_extractor = tf.keras.Model(
            inputs=cnn_model.input,
            outputs=cnn_model.get_layer("bn_head").output,
        )
        self._lstm = lstm_model
        self._buffer = []

    def predict(self, eye_image: np.ndarray) -> dict:
        """
        Process one frame and return temporal prediction when buffer is full.

        Returns:
            Dict with 'temporal_prob', 'frame_prob', 'buffer_ready'
            or None if buffer not yet full
        """
        if eye_image.ndim == 3:
            eye_image = np.expand_dims(eye_image, 0)

        # Extract features
        features = self._feature_extractor.predict(eye_image, verbose=0)[0]
        self._buffer.append(features)

        # Also get frame-level prediction for comparison
        frame_prob = None  # Could add if needed

        if len(self._buffer) >= self.seq_length:
            # Run LSTM on the recent window
            seq = np.array(self._buffer[-self.seq_length:])[np.newaxis]
            temporal_prob = float(self._lstm.predict(seq, verbose=0).flatten()[0])

            return {
                "temporal_prob": temporal_prob,
                "buffer_ready": True,
                "buffer_size": len(self._buffer),
            }
        else:
            return {
                "temporal_prob": None,
                "buffer_ready": False,
                "buffer_size": len(self._buffer),
            }

    def reset(self):
        """Clear the frame buffer."""
        self._buffer = []
