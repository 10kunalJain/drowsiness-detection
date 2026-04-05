"""
Error analysis module — systematic investigation of model failure modes.

Goes beyond accuracy to understand *why* and *where* the model fails:
1. Misclassification visualization and clustering
2. Confidence distribution analysis
3. Hardness profiling (which samples are consistently hard)
4. Per-class failure mode breakdown
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


def analyze_errors(images: np.ndarray, labels: np.ndarray,
                   predictions: np.ndarray, save_dir: Path = None):
    """
    Full error analysis pipeline.

    Args:
        images: Test set images (N, H, W, 3)
        labels: Ground truth labels (N,)
        predictions: Model probabilities (N,)
        save_dir: Directory to save plots
    """
    save_dir = save_dir or config.ERROR_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    y_pred_class = (predictions > 0.5).astype(int)
    correct = y_pred_class == labels
    incorrect = ~correct

    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    print(f"  Total test samples: {len(labels)}")
    print(f"  Correct: {correct.sum()} ({100*correct.mean():.1f}%)")
    print(f"  Errors:  {incorrect.sum()} ({100*incorrect.mean():.1f}%)")

    # ── 1. Confidence Distribution ─────────────────────────────────────
    _plot_confidence_distribution(predictions, labels, correct, save_dir)

    # ── 2. Error Type Breakdown ────────────────────────────────────────
    _plot_error_breakdown(labels, y_pred_class, predictions, save_dir)

    # ── 3. Misclassified Samples Gallery ───────────────────────────────
    _plot_misclassified_gallery(images, labels, predictions, incorrect, save_dir)

    # ── 4. Hardness Analysis ───────────────────────────────────────────
    _plot_hardness_analysis(predictions, labels, save_dir)

    print(f"\n  All error analysis plots saved to {save_dir}")


def _plot_confidence_distribution(predictions, labels, correct, save_dir):
    """Confidence distribution for correct vs incorrect predictions."""
    confidence = np.abs(predictions - 0.5) * 2  # Scale to [0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # By correctness
    axes[0].hist(confidence[correct], bins=30, alpha=0.6,
                 label="Correct", color="#2ecc71", density=True)
    axes[0].hist(confidence[~correct], bins=30, alpha=0.6,
                 label="Incorrect", color="#e74c3c", density=True)
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Confidence: Correct vs Incorrect", fontweight="bold")
    axes[0].legend()

    # By class
    for class_idx, class_name, color in zip([0, 1], config.CLASS_NAMES,
                                             ["#e74c3c", "#2ecc71"]):
        mask = labels == class_idx
        axes[1].hist(predictions[mask], bins=30, alpha=0.5,
                     label=class_name, color=color, density=True)
    axes[1].axvline(x=0.5, color="black", linestyle="--", alpha=0.5, label="Threshold")
    axes[1].set_xlabel("P(Drowsy)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Prediction Distribution by Class", fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_dir / "confidence_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved confidence_distribution.png")


def _plot_error_breakdown(labels, y_pred_class, predictions, save_dir):
    """Breakdown of error types: false positives vs false negatives."""
    # False Positives: predicted DROWSY but actually NATURAL (label=1 pred, label=0 true)
    fp_mask = (y_pred_class == 0) & (labels == 1)  # Predicted drowsy, actually natural
    fn_mask = (y_pred_class == 1) & (labels == 0)  # Predicted natural, actually drowsy

    # For drowsy=0: FP means model says natural(1) when it's drowsy(0)
    # Actually: class 0=DROWSY, class 1=NATURAL
    # FP for drowsy detection: model says drowsy but it's natural
    missed_drowsy = (y_pred_class == 1) & (labels == 0)  # Said natural, was drowsy
    false_alarm = (y_pred_class == 0) & (labels == 1)    # Said drowsy, was natural

    fig, ax = plt.subplots(figsize=(6, 4))
    error_types = ["Missed Drowsy\n(False Negative)", "False Alarm\n(False Positive)"]
    counts = [missed_drowsy.sum(), false_alarm.sum()]
    colors = ["#e74c3c", "#f39c12"]
    bars = ax.bar(error_types, counts, color=colors)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_title("Error Type Breakdown", fontweight="bold")
    ax.text(0.5, -0.15,
            "Missed Drowsy is MORE dangerous than False Alarm in production",
            ha="center", transform=ax.transAxes, fontsize=9, fontstyle="italic",
            color="red")
    plt.tight_layout()
    plt.savefig(save_dir / "error_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved error_breakdown.png")


def _plot_misclassified_gallery(images, labels, predictions, incorrect, save_dir):
    """Visualize the most confident misclassifications."""
    if incorrect.sum() == 0:
        print("  No misclassifications to visualize!")
        return

    error_images = images[incorrect]
    error_labels = labels[incorrect]
    error_preds = predictions[incorrect]

    # Sort by confidence (most confident errors first — these are the most concerning)
    error_confidence = np.abs(error_preds - 0.5)
    sorted_idx = np.argsort(error_confidence)[::-1]

    n = min(16, len(error_images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i in range(n):
        idx = sorted_idx[i]
        axes[i].imshow(error_images[idx][:, :, 0], cmap="gray")
        true_label = config.CLASS_NAMES[int(error_labels[idx])]
        pred_label = config.CLASS_NAMES[int(error_preds[idx] > 0.5)]
        axes[i].set_title(
            f"True: {true_label}\nPred: {pred_label} ({error_preds[idx]:.2f})",
            fontsize=8,
            color="red",
        )
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Most Confident Misclassifications (sorted by confidence)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "misclassified_gallery.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved misclassified_gallery.png")


def _plot_hardness_analysis(predictions, labels, save_dir):
    """
    Analyze prediction 'hardness' — samples near the decision boundary
    are harder for the model and likely represent ambiguous cases.
    """
    distance_from_boundary = np.abs(predictions - 0.5)

    fig, ax = plt.subplots(figsize=(8, 4))

    bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
    bin_labels = ["Very Hard\n(<0.05)", "Hard\n(0.05-0.1)", "Medium\n(0.1-0.2)",
                  "Easy\n(0.2-0.3)", "Very Easy\n(>0.3)"]
    counts = []
    accuracies = []

    for i in range(len(bins) - 1):
        mask = (distance_from_boundary >= bins[i]) & (distance_from_boundary < bins[i+1])
        counts.append(mask.sum())
        if mask.sum() > 0:
            y_pred = (predictions[mask] > 0.5).astype(int)
            acc = (y_pred == labels[mask]).mean()
            accuracies.append(acc)
        else:
            accuracies.append(0)

    x = np.arange(len(bin_labels))
    bars = ax.bar(x, counts, color="#3498db", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_ylabel("Sample Count", color="#3498db")
    ax.set_title("Sample Hardness Distribution", fontweight="bold")

    ax2 = ax.twinx()
    ax2.plot(x, accuracies, "o-", color="#e74c3c", linewidth=2, markersize=8)
    ax2.set_ylabel("Accuracy", color="#e74c3c")
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_dir / "hardness_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved hardness_analysis.png")
