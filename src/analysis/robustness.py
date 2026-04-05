"""
Distribution Shift & Robustness Testing.

A model that works on clean test data may collapse in the real world.
This module systematically degrades inputs to answer:

  "At what point does the model break, and how gracefully does it fail?"

Simulated corruptions (mapped to real driving conditions):
1. Brightness shift     → night driving, tunnel entry/exit
2. Gaussian blur        → camera out of focus, vibration
3. Gaussian noise       → low-light sensor noise, cheap cameras
4. Contrast reduction   → fog, glare, washed-out dashcam
5. Occlusion            → sunglasses, hand on face, hair

For each corruption at multiple severity levels, we measure:
- Accuracy drop
- AUC drop
- Uncertainty increase (if MC Dropout available)
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score
from pathlib import Path
from scipy.ndimage import gaussian_filter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


# ── Corruption Functions ───────────────────────────────────────────────

def _apply_brightness_shift(images: np.ndarray, severity: float) -> np.ndarray:
    """Shift brightness. severity: 0.0 (none) to 1.0 (very dark)."""
    factor = 1.0 - severity * 0.7  # At max: reduce to 30% brightness
    return np.clip(images * factor, 0, 1)


def _apply_darkness(images: np.ndarray, severity: float) -> np.ndarray:
    """Simulate very low-light / night conditions."""
    factor = 1.0 - severity * 0.85
    noisy = images * factor + np.random.normal(0, severity * 0.05, images.shape)
    return np.clip(noisy, 0, 1)


def _apply_gaussian_blur(images: np.ndarray, severity: float) -> np.ndarray:
    """Blur images. severity: 0.0 (sharp) to 1.0 (heavily blurred)."""
    sigma = severity * 3.0
    if sigma < 0.1:
        return images
    blurred = np.zeros_like(images)
    for i in range(len(images)):
        for c in range(images.shape[-1]):
            blurred[i, :, :, c] = gaussian_filter(images[i, :, :, c], sigma=sigma)
    return blurred


def _apply_gaussian_noise(images: np.ndarray, severity: float) -> np.ndarray:
    """Add Gaussian noise. severity: 0.0 (clean) to 1.0 (very noisy)."""
    std = severity * 0.2
    noise = np.random.normal(0, std, images.shape)
    return np.clip(images + noise, 0, 1)


def _apply_contrast_reduction(images: np.ndarray, severity: float) -> np.ndarray:
    """Reduce contrast. severity: 0.0 (normal) to 1.0 (flat gray)."""
    mean = images.mean(axis=(1, 2, 3), keepdims=True)
    factor = 1.0 - severity * 0.85
    return np.clip(mean + (images - mean) * factor, 0, 1)


def _apply_occlusion(images: np.ndarray, severity: float) -> np.ndarray:
    """Random rectangular occlusion (simulates sunglasses, hand, hair)."""
    occluded = images.copy()
    h, w = images.shape[1], images.shape[2]
    occ_size = int(max(h, w) * severity * 0.5)
    if occ_size < 2:
        return occluded
    for i in range(len(occluded)):
        y = np.random.randint(0, max(1, h - occ_size))
        x = np.random.randint(0, max(1, w - occ_size))
        occluded[i, y:y+occ_size, x:x+occ_size, :] = 0  # Black patch
    return occluded


CORRUPTIONS = {
    "Brightness Shift": _apply_brightness_shift,
    "Low Light":        _apply_darkness,
    "Gaussian Blur":    _apply_gaussian_blur,
    "Gaussian Noise":   _apply_gaussian_noise,
    "Low Contrast":     _apply_contrast_reduction,
    "Occlusion":        _apply_occlusion,
}


# ── Main Robustness Test ──────────────────────────────────────────────

def run_robustness_test(model: tf.keras.Model,
                        X_test: np.ndarray, y_test: np.ndarray,
                        severity_levels: list[float] = None,
                        save_dir: Path = None) -> dict:
    """
    Run full robustness test across all corruptions and severity levels.

    Args:
        model: Trained model
        X_test: Clean test images
        y_test: Test labels
        severity_levels: List of severity values (0-1)
        save_dir: Where to save results

    Returns:
        Dict mapping corruption_name → {severity → {accuracy, auc}}
    """
    if severity_levels is None:
        severity_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    save_dir = save_dir or config.PLOT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ROBUSTNESS & DISTRIBUTION SHIFT TESTING")
    print("=" * 60)

    # Baseline (clean)
    y_prob_clean = model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=0).flatten()
    y_pred_clean = (y_prob_clean > 0.5).astype(int)
    baseline_acc = accuracy_score(y_test, y_pred_clean)
    baseline_auc = roc_auc_score(y_test, y_prob_clean)
    print(f"  Baseline — Accuracy: {baseline_acc:.4f}, AUC: {baseline_auc:.4f}")

    results = {}

    for corr_name, corr_fn in CORRUPTIONS.items():
        print(f"\n  Testing: {corr_name}")
        results[corr_name] = {}

        for severity in severity_levels:
            # Apply corruption
            X_corrupted = corr_fn(X_test.copy(), severity).astype(np.float32)

            # Predict
            y_prob = model.predict(
                X_corrupted, batch_size=config.BATCH_SIZE, verbose=0
            ).flatten()
            y_pred = (y_prob > 0.5).astype(int)

            acc = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_prob)
            except ValueError:
                auc = 0.5  # All same prediction

            results[corr_name][severity] = {
                "accuracy": acc,
                "auc": auc,
                "acc_drop": baseline_acc - acc,
                "auc_drop": baseline_auc - auc,
            }

            if severity > 0:
                drop = baseline_acc - acc
                symbol = "!!" if drop > 0.1 else "!" if drop > 0.05 else " "
                print(f"    severity={severity:.1f}: "
                      f"acc={acc:.4f} (drop={drop:+.4f}) {symbol}")

    # Generate plots
    _plot_robustness_results(results, severity_levels, baseline_acc, baseline_auc, save_dir)
    _plot_corruption_samples(X_test, save_dir)

    # Summary: most vulnerable corruptions
    print(f"\n  {'─'*50}")
    print(f"  VULNERABILITY RANKING (by accuracy drop at severity=0.8):")
    drops = {
        name: results[name].get(0.8, {}).get("acc_drop", 0)
        for name in results
    }
    for rank, (name, drop) in enumerate(
        sorted(drops.items(), key=lambda x: x[1], reverse=True), 1
    ):
        flag = " ← CRITICAL" if drop > 0.1 else ""
        print(f"    {rank}. {name}: -{drop:.4f}{flag}")

    return results


def _plot_robustness_results(results, severity_levels, baseline_acc,
                             baseline_auc, save_dir):
    """Generate comprehensive robustness visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    # 1. Accuracy vs Severity
    for (name, data), color in zip(results.items(), colors):
        accs = [data[s]["accuracy"] for s in severity_levels]
        axes[0].plot(severity_levels, accs, "o-", label=name, color=color, linewidth=2)
    axes[0].axhline(y=baseline_acc, color="gray", ls="--", alpha=0.5, label="Baseline")
    axes[0].set_xlabel("Corruption Severity")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy Degradation Under Corruption", fontweight="bold")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].set_ylim(0.4, 1.02)
    axes[0].grid(alpha=0.3)

    # 2. AUC vs Severity
    for (name, data), color in zip(results.items(), colors):
        aucs = [data[s]["auc"] for s in severity_levels]
        axes[1].plot(severity_levels, aucs, "o-", label=name, color=color, linewidth=2)
    axes[1].axhline(y=baseline_auc, color="gray", ls="--", alpha=0.5, label="Baseline")
    axes[1].set_xlabel("Corruption Severity")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("AUC-ROC Degradation Under Corruption", fontweight="bold")
    axes[1].legend(fontsize=8, loc="lower left")
    axes[1].set_ylim(0.4, 1.02)
    axes[1].grid(alpha=0.3)

    # 3. Heatmap: accuracy drop at each severity
    corr_names = list(results.keys())
    drop_matrix = np.array([
        [results[name][s]["acc_drop"] for s in severity_levels]
        for name in corr_names
    ])
    im = axes[2].imshow(drop_matrix, cmap="YlOrRd", aspect="auto")
    axes[2].set_xticks(range(len(severity_levels)))
    axes[2].set_xticklabels([f"{s:.1f}" for s in severity_levels])
    axes[2].set_yticks(range(len(corr_names)))
    axes[2].set_yticklabels(corr_names, fontsize=9)
    axes[2].set_xlabel("Severity")
    axes[2].set_title("Accuracy Drop Heatmap", fontweight="bold")
    plt.colorbar(im, ax=axes[2], label="Accuracy Drop")

    # Annotate cells
    for i in range(len(corr_names)):
        for j in range(len(severity_levels)):
            val = drop_matrix[i, j]
            text_color = "white" if val > 0.1 else "black"
            axes[2].text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=text_color)

    plt.suptitle("Model Robustness Under Distribution Shift",
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "robustness_test.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved robustness_test.png")


def _plot_corruption_samples(X_test, save_dir):
    """Show visual examples of each corruption at different severities."""
    sample_idx = 0
    sample = X_test[sample_idx]
    severities = [0.0, 0.3, 0.6, 1.0]

    fig, axes = plt.subplots(len(CORRUPTIONS), len(severities),
                             figsize=(12, 2.5 * len(CORRUPTIONS)))

    for row, (name, fn) in enumerate(CORRUPTIONS.items()):
        for col, sev in enumerate(severities):
            corrupted = fn(sample[np.newaxis], sev)[0]
            axes[row, col].imshow(corrupted[:, :, 0], cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(name, fontsize=9, fontweight="bold")
            if row == 0:
                axes[row, col].set_title(f"Severity: {sev:.1f}", fontsize=9)

    plt.suptitle("Corruption Examples at Different Severities",
                fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "corruption_samples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved corruption_samples.png")
