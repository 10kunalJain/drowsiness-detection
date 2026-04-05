"""
Training pipeline — supports both custom CNN (end-to-end) and transfer learning (two-phase).

Custom CNN: trains all layers from scratch with Mixup augmentation
MobileNetV2: Phase 1 frozen backbone → Phase 2 fine-tune top layers

Includes proper callbacks, metric logging, and evaluation.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve,
)
import seaborn as sns
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.data.dataset import get_augmentation_layer, mixup_batch
from src.models.drowsiness_model import (
    build_model, compile_model, unfreeze_backbone, save_model,
)


def get_callbacks(phase: str = "phase1") -> list:
    """Build training callbacks."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.LR_REDUCE_FACTOR,
            patience=config.LR_REDUCE_PATIENCE,
            min_lr=1e-7,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(config.MODEL_DIR / f"best_{phase}.keras"),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
        ),
    ]


def _build_train_dataset(X, y):
    """Build a tf.data pipeline with augmentation and Mixup."""
    augment = get_augmentation_layer()

    train_ds = tf.data.Dataset.from_tensor_slices((X, y))
    train_ds = train_ds.shuffle(len(X), seed=config.RANDOM_SEED)
    train_ds = train_ds.batch(config.BATCH_SIZE)
    train_ds = train_ds.map(
        lambda x, y: (augment(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    # Apply Mixup
    if config.MIXUP_ALPHA > 0:
        train_ds = train_ds.map(
            lambda x, y: mixup_batch(x, y, config.MIXUP_ALPHA),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds


def train(data: dict, class_weights: dict = None) -> tuple[tf.keras.Model, dict]:
    """
    Full training pipeline. Adapts to model type:
    - custom_cnn: trains end-to-end for all epochs
    - mobilenetv2/resnet: two-phase (frozen → fine-tune)
    """
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = _build_train_dataset(data["X_train"], data["y_train"])
    val_ds = tf.data.Dataset.from_tensor_slices((data["X_val"], data["y_val"]))
    val_ds = val_ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = build_model()
    is_custom = config.MODEL_TYPE == "custom_cnn"
    print(f"  Model type: {config.MODEL_TYPE} | Params: {model.count_params():,}")

    if is_custom:
        # ── Custom CNN: train end-to-end ───────────────────────────────
        print("\n" + "="*60)
        print(f"TRAINING: Custom CNN (end-to-end, {config.EPOCHS} epochs)")
        print("="*60)
        model = compile_model(model)
        model.summary()

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.EPOCHS,
            class_weight=class_weights,
            callbacks=get_callbacks("custom_cnn"),
            verbose=1,
        )
        combined_history = history.history

    else:
        # ── Transfer Learning: two-phase ───────────────────────────────
        print("\n" + "="*60)
        print("PHASE 1: Training classifier head (backbone frozen)")
        print("="*60)
        model = compile_model(model)
        model.summary(print_fn=lambda x: None)

        history1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.FINE_TUNE_AT_EPOCH,
            class_weight=class_weights,
            callbacks=get_callbacks("phase1"),
            verbose=1,
        )

        print("\n" + "="*60)
        print("PHASE 2: Fine-tuning backbone (top layers unfrozen)")
        print("="*60)
        model = unfreeze_backbone(model)
        model = compile_model(model, learning_rate=config.FINE_TUNE_LR)

        remaining = config.EPOCHS - config.FINE_TUNE_AT_EPOCH
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=remaining,
            class_weight=class_weights,
            callbacks=get_callbacks("phase2"),
            verbose=1,
        )

        combined_history = {}
        for key in history1.history:
            combined_history[key] = history1.history[key] + history2.history[key]

    # Save final model
    save_model(model)

    return model, combined_history


def evaluate(model: tf.keras.Model, data: dict) -> dict:
    """Comprehensive evaluation on test set."""
    config.PLOT_DIR.mkdir(parents=True, exist_ok=True)

    X_test, y_test = data["X_test"], data["y_test"]
    y_prob = model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    report = classification_report(
        y_test, y_pred, target_names=config.CLASS_NAMES, output_dict=True,
    )
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=config.CLASS_NAMES))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.PLOT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ROC + PR Curves
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, color="#2980b9", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve", fontweight="bold")
    axes[0].legend()

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(recall, precision, color="#e74c3c", lw=2)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve", fontweight="bold")

    plt.tight_layout()
    plt.savefig(config.PLOT_DIR / "roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  AUC-ROC: {roc_auc:.4f}")

    return {
        "report": report,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def plot_training_history(history: dict):
    """Plot training curves for loss, accuracy, and AUC."""
    config.PLOT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
        ("auc", "AUC-ROC"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for ax, (metric, title) in zip(axes, metrics):
        if metric in history:
            ax.plot(history[metric], label="Train", color="#2980b9")
        if f"val_{metric}" in history:
            ax.plot(history[f"val_{metric}"], label="Validation", color="#e74c3c")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.PLOT_DIR / "training_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved training_history.png")
