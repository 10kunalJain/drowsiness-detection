"""
Failure → Improvement Loop — close the feedback cycle from error analysis.

This is what separates a portfolio project from a Kaggle notebook:
you don't just find errors, you *fix* them and measure the improvement.

Strategies applied:
1. Targeted augmentation for failure modes (low-light, low-contrast)
2. Optimal threshold search (move away from default 0.5)
3. Class weight adjustment based on error-type asymmetry
4. Hard example mining (retrain with emphasis on difficult samples)
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report,
)
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                           metric: str = "f1",
                           save_dir: Path = None) -> float:
    """
    Search for the optimal classification threshold instead of using default 0.5.

    In safety-critical systems like drowsiness detection, we may want to
    favor recall (catching all drowsy cases) over precision.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        metric: Optimization target — "f1", "recall", or "balanced"
        save_dir: Where to save the threshold analysis plot

    Returns:
        Optimal threshold value
    """
    save_dir = save_dir or config.PLOT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    thresholds = np.arange(0.20, 0.85, 0.01)
    f1_scores = []
    precisions = []
    recalls = []
    specificities = []

    for t in thresholds:
        y_pred = (y_prob > t).astype(int)
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    f1_scores = np.array(f1_scores)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    specificities = np.array(specificities)

    # Choose threshold based on objective
    if metric == "f1":
        best_idx = np.argmax(f1_scores)
    elif metric == "recall":
        # Maximize recall while keeping precision above 0.7
        valid = precisions > 0.7
        if valid.any():
            masked_recall = np.where(valid, recalls, 0)
            best_idx = np.argmax(masked_recall)
        else:
            best_idx = np.argmax(recalls)
    else:  # balanced
        balanced = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(balanced)

    optimal_threshold = thresholds[best_idx]

    print(f"\n  Threshold Optimization (target: {metric})")
    print(f"  Default (0.50): F1={f1_scores[np.argmin(np.abs(thresholds-0.5))]:.4f}")
    print(f"  Optimal ({optimal_threshold:.2f}): F1={f1_scores[best_idx]:.4f}")
    print(f"    Precision: {precisions[best_idx]:.4f}")
    print(f"    Recall: {recalls[best_idx]:.4f}")

    # Plot threshold analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(thresholds, f1_scores, label="F1", color="#2980b9", lw=2)
    axes[0].plot(thresholds, precisions, label="Precision", color="#27ae60", lw=1.5, ls="--")
    axes[0].plot(thresholds, recalls, label="Recall", color="#e74c3c", lw=1.5, ls="--")
    axes[0].axvline(x=0.5, color="gray", ls=":", alpha=0.5, label="Default (0.5)")
    axes[0].axvline(x=optimal_threshold, color="#f39c12", ls="-", lw=2,
                    label=f"Optimal ({optimal_threshold:.2f})")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Threshold vs Metrics", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # F1 improvement
    default_f1 = f1_scores[np.argmin(np.abs(thresholds - 0.5))]
    optimal_f1 = f1_scores[best_idx]
    improvement = optimal_f1 - default_f1

    labels = ["Default (0.5)", f"Optimal ({optimal_threshold:.2f})"]
    values = [default_f1, optimal_f1]
    bars = axes[1].bar(labels, values, color=["#95a5a6", "#f39c12"], alpha=0.8)
    for bar, val in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("F1-Score")
    axes[1].set_title(f"F1 Improvement: +{improvement:.4f}", fontweight="bold",
                     color="green" if improvement > 0 else "red")
    axes[1].set_ylim(min(values) - 0.05, max(values) + 0.03)

    plt.suptitle("Threshold Optimization Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "threshold_optimization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved threshold_optimization.png")

    return optimal_threshold


def build_targeted_augmentation():
    """
    Build augmentation pipeline specifically targeting common failure modes:
    - Low-light conditions (brightness shifts)
    - Partial occlusions (random erasing)
    - Low contrast (contrast stretching)

    This is the 'improvement' step: error analysis revealed these weaknesses,
    now we augment specifically for them.
    """
    return tf.keras.Sequential([
        # Standard augmentation
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom((-0.1, 0.1)),
        # Targeted for failure modes
        tf.keras.layers.RandomBrightness(0.25),   # Wider than before (was 0.15)
        tf.keras.layers.RandomContrast(0.3),       # Wider than before (was 0.15)
    ], name="targeted_augmentation")


def retrain_with_improvements(model: tf.keras.Model, data: dict,
                              class_weights: dict = None,
                              optimal_threshold: float = 0.5,
                              epochs: int = 10) -> tuple:
    """
    Retrain model with improvements identified from error analysis:
    1. Targeted augmentation for failure modes
    2. Class weights for imbalance
    3. Hard example mining (upweight difficult samples)

    Returns:
        (improved_model, before_metrics, after_metrics)
    """
    print("\n" + "=" * 60)
    print("RETRAINING WITH IMPROVEMENTS")
    print("=" * 60)

    # ── Before metrics (baseline) ──────────────────────────────────────
    y_prob_before = model.predict(
        data["X_test"], batch_size=config.BATCH_SIZE, verbose=0
    ).flatten()
    y_pred_before = (y_prob_before > 0.5).astype(int)
    before = {
        "auc": roc_auc_score(data["y_test"], y_prob_before),
        "f1": f1_score(data["y_test"], y_pred_before, average="weighted"),
        "recall_drowsy": recall_score(data["y_test"], y_pred_before, pos_label=0),
    }
    print(f"  Before: AUC={before['auc']:.4f}, F1={before['f1']:.4f}, "
          f"Recall(drowsy)={before['recall_drowsy']:.4f}")

    # ── Hard example mining ────────────────────────────────────────────
    # Find samples the model gets wrong or is uncertain about
    train_probs = model.predict(
        data["X_train"], batch_size=config.BATCH_SIZE, verbose=0
    ).flatten()
    train_pred = (train_probs > 0.5).astype(int)
    is_hard = (train_pred != data["y_train"]) | (np.abs(train_probs - 0.5) < 0.15)
    hard_ratio = is_hard.mean()
    print(f"  Hard examples: {is_hard.sum()} / {len(is_hard)} ({100*hard_ratio:.1f}%)")

    # Create sample weights: hard examples get 2x weight
    sample_weights = np.ones(len(data["y_train"]))
    sample_weights[is_hard] = 2.0

    # ── Build improved training pipeline ───────────────────────────────
    targeted_augment = build_targeted_augmentation()

    train_ds = tf.data.Dataset.from_tensor_slices(
        (data["X_train"], data["y_train"], sample_weights)
    )
    train_ds = train_ds.shuffle(len(data["X_train"]), seed=config.RANDOM_SEED)
    train_ds = train_ds.batch(config.BATCH_SIZE)
    train_ds = train_ds.map(
        lambda x, y, w: (targeted_augment(x, training=True), y, w),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((data["X_val"], data["y_val"]))
    val_ds = val_ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Recompile with lower learning rate (fine-tuning an already-trained model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.FINE_TUNE_LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    print(f"  Retraining for {epochs} epochs with targeted augmentation + hard mining...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=4,
                restore_best_weights=True, mode="max"
            ),
        ],
        verbose=1,
    )

    # ── After metrics ──────────────────────────────────────────────────
    y_prob_after = model.predict(
        data["X_test"], batch_size=config.BATCH_SIZE, verbose=0
    ).flatten()

    # Use optimized threshold for final predictions
    y_pred_after = (y_prob_after > optimal_threshold).astype(int)
    after = {
        "auc": roc_auc_score(data["y_test"], y_prob_after),
        "f1": f1_score(data["y_test"], y_pred_after, average="weighted"),
        "recall_drowsy": recall_score(data["y_test"], y_pred_after, pos_label=0),
    }
    print(f"\n  After:  AUC={after['auc']:.4f}, F1={after['f1']:.4f}, "
          f"Recall(drowsy)={after['recall_drowsy']:.4f}")

    # ── Plot improvement ───────────────────────────────────────────────
    _plot_improvement(before, after)

    print("\n  Classification Report (after improvement):")
    y_pred_final = (y_prob_after > optimal_threshold).astype(int)
    print(classification_report(
        data["y_test"], y_pred_final, target_names=config.CLASS_NAMES
    ))

    return model, before, after


def _plot_improvement(before: dict, after: dict):
    """Visualize before/after improvement comparison."""
    config.PLOT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = ["auc", "f1", "recall_drowsy"]
    labels = ["AUC-ROC", "F1-Score", "Recall (Drowsy)"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.3

    before_vals = [before[m] for m in metrics]
    after_vals = [after[m] for m in metrics]

    bars1 = ax.bar(x - width/2, before_vals, width, label="Before", color="#95a5a6", alpha=0.8)
    bars2 = ax.bar(x + width/2, after_vals, width, label="After Improvement", color="#27ae60", alpha=0.8)

    for bar, val in zip(bars1, before_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=9)
    for bar, val in zip(bars2, after_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")

    # Show delta
    for i, (b, a) in enumerate(zip(before_vals, after_vals)):
        delta = a - b
        color = "green" if delta > 0 else "red"
        ax.text(i, max(a, b) + 0.025, f"{'+'if delta>0 else ''}{delta:.4f}",
                ha="center", fontsize=10, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance: Before vs After Error-Driven Improvements",
                fontweight="bold")
    ax.legend()
    ax.set_ylim(min(before_vals + after_vals) - 0.05, 1.08)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.PLOT_DIR / "improvement_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved improvement_comparison.png")
