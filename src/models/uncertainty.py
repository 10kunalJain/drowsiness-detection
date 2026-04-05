"""
Monte Carlo Dropout Uncertainty Estimation.

Standard neural networks output a point estimate (e.g., P(drowsy)=0.82)
but say nothing about *how sure* they are about that estimate. Two images
can both get P=0.82 — one where the model truly sees a drowsy eye, and one
where it's confused between a blink and drowsiness.

MC Dropout fixes this by running N stochastic forward passes with dropout
ENABLED at inference time. The variance across passes is the model's
epistemic uncertainty — uncertainty due to lack of knowledge.

Decision logic:
  Low uncertainty  + High drowsy prob → Confident DROWSY
  Low uncertainty  + Low drowsy prob  → Confident ALERT
  High uncertainty + Any prob         → UNCERTAIN → trigger caution mode

This is critical for safety: "I don't know" is better than a wrong answer.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


@dataclass
class UncertaintyResult:
    """Result from uncertainty-aware prediction."""
    mean_prob: float          # Average drowsy probability across MC samples
    std_prob: float           # Standard deviation (epistemic uncertainty)
    prediction: int           # Final class (0=DROWSY, 1=NATURAL)
    confidence: float         # 1 - normalized uncertainty
    uncertainty_level: str    # LOW / MEDIUM / HIGH
    is_reliable: bool         # Whether to trust this prediction
    mc_samples: np.ndarray    # Raw MC samples for analysis


class MCDropoutEstimator:
    """
    Monte Carlo Dropout for uncertainty estimation.

    At inference time, we run the model N times with dropout active.
    The spread of predictions tells us how uncertain the model is.
    """

    # Uncertainty thresholds (std of MC samples)
    UNCERTAINTY_THRESHOLDS = {
        "LOW": 0.05,       # std < 0.05 → highly confident
        "MEDIUM": 0.12,    # 0.05 ≤ std < 0.12 → moderate uncertainty
        "HIGH": float("inf"),  # std ≥ 0.12 → high uncertainty
    }

    def __init__(self, model: tf.keras.Model, n_samples: int = 30):
        """
        Args:
            model: Trained Keras model (must have Dropout layers)
            n_samples: Number of stochastic forward passes
        """
        self.model = model
        self.n_samples = n_samples

    def _classify_uncertainty(self, std_prob: float) -> tuple:
        """Classify uncertainty level from std."""
        if std_prob < self.UNCERTAINTY_THRESHOLDS["LOW"]:
            level = "LOW"
        elif std_prob < self.UNCERTAINTY_THRESHOLDS["MEDIUM"]:
            level = "MEDIUM"
        else:
            level = "HIGH"
        is_reliable = level != "HIGH"
        confidence = max(0.0, 1.0 - (std_prob / 0.5))
        return level, is_reliable, confidence

    def predict_with_uncertainty(self, image: np.ndarray) -> UncertaintyResult:
        """
        Run MC Dropout inference on a single image.

        Args:
            image: Preprocessed image (H, W, 3) or (1, H, W, 3)

        Returns:
            UncertaintyResult with mean prediction, uncertainty, and reliability
        """
        if image.ndim == 3:
            image = np.expand_dims(image, 0)

        # Run N stochastic forward passes (training=True keeps dropout active)
        mc_samples = np.array([
            self.model(image, training=True).numpy().flatten()[0]
            for _ in range(self.n_samples)
        ])

        mean_prob = mc_samples.mean()
        std_prob = mc_samples.std()
        level, is_reliable, confidence = self._classify_uncertainty(std_prob)

        return UncertaintyResult(
            mean_prob=float(mean_prob),
            std_prob=float(std_prob),
            prediction=int(mean_prob > 0.5),
            confidence=float(confidence),
            uncertainty_level=level,
            is_reliable=is_reliable,
            mc_samples=mc_samples,
        )

    def predict_batch_with_uncertainty(self, images: np.ndarray,
                                       batch_size: int = None) -> tuple:
        """
        Batched MC Dropout — runs N passes over the ENTIRE dataset at once.

        Instead of 1,483 images x 30 passes = 44,490 individual calls,
        this does 30 batched passes = 30 calls with batch_size=64.
        ~15x faster.

        Args:
            images: All images (N, H, W, 3)
            batch_size: Batch size for each pass

        Returns:
            (means, stds, predictions, levels, reliable) — all arrays of len N
        """
        if batch_size is None:
            batch_size = config.BATCH_SIZE

        n_images = len(images)

        # Collect all MC samples: shape (n_samples, n_images)
        all_samples = np.zeros((self.n_samples, n_images), dtype=np.float32)

        import time
        t_start = time.time()

        for mc_pass in range(self.n_samples):
            # Batched forward pass with dropout active
            preds = []
            for start in range(0, n_images, batch_size):
                batch = images[start:start + batch_size]
                # training=True keeps dropout stochastic
                out = self.model(batch, training=True).numpy().flatten()
                preds.append(out)
            all_samples[mc_pass] = np.concatenate(preds)

            elapsed = time.time() - t_start
            avg_per_pass = elapsed / (mc_pass + 1)
            remaining = avg_per_pass * (self.n_samples - mc_pass - 1)
            print(f"    MC pass {mc_pass + 1}/{self.n_samples} "
                  f"({n_images} images) — "
                  f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining",
                  end="\r")

        total = time.time() - t_start
        print(f"\n    All {self.n_samples} passes done in {total:.1f}s "
              f"({n_images * self.n_samples:,} total forward passes)")

        # Compute statistics across MC passes (axis=0)
        means = all_samples.mean(axis=0)   # (n_images,)
        stds = all_samples.std(axis=0)     # (n_images,)
        predictions = (means > 0.5).astype(int)

        # Classify uncertainty levels
        levels = []
        reliable = []
        confidences = []
        for s in stds:
            level, is_rel, conf = self._classify_uncertainty(float(s))
            levels.append(level)
            reliable.append(is_rel)
            confidences.append(conf)

        return means, stds, predictions, np.array(levels), np.array(reliable)

    def analyze_uncertainty_distribution(self, images: np.ndarray,
                                         labels: np.ndarray,
                                         save_dir: Path = None):
        """
        Analyze uncertainty patterns across the dataset.

        Key insights:
        1. Do misclassified samples have higher uncertainty? (They should)
        2. Which class has more uncertainty? (Indicates harder class)
        3. How many predictions should we NOT trust?
        """
        save_dir = save_dir or config.PLOT_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print("UNCERTAINTY ANALYSIS (Monte Carlo Dropout)")
        print("=" * 60)
        print(f"  Running {self.n_samples} batched stochastic passes "
              f"over {len(images)} samples...")

        means, stds, preds, levels, reliable = self.predict_batch_with_uncertainty(images)

        correct = preds == labels
        incorrect = ~correct

        # ── Stats ──────────────────────────────────────────────────────
        print(f"  Avg uncertainty (correct):   {stds[correct].mean():.4f}")
        if incorrect.any():
            print(f"  Avg uncertainty (incorrect): {stds[incorrect].mean():.4f}")
        else:
            print(f"  No incorrect predictions")
        print(f"  Unreliable predictions: {(~reliable).sum()} / {len(labels)} "
              f"({100*(~reliable).mean():.1f}%)")

        acc_reliable = 0.0
        acc_unreliable = 0.0
        if reliable.any():
            acc_reliable = (preds[reliable] == labels[reliable]).mean()
            print(f"  Accuracy (reliable only):  {acc_reliable:.4f}")
        if (~reliable).any():
            acc_unreliable = (preds[~reliable] == labels[~reliable]).mean()
            print(f"  Accuracy (unreliable):     {acc_unreliable:.4f}")

        # ── Plots ──────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Uncertainty: correct vs incorrect
        axes[0, 0].hist(stds[correct], bins=25, alpha=0.6,
                        label="Correct", color="#2ecc71", density=True)
        if incorrect.any():
            axes[0, 0].hist(stds[incorrect], bins=25, alpha=0.6,
                            label="Incorrect", color="#e74c3c", density=True)
        axes[0, 0].axvline(x=self.UNCERTAINTY_THRESHOLDS["MEDIUM"], color="orange",
                          ls="--", label="High uncertainty threshold")
        axes[0, 0].set_xlabel("Uncertainty (Std)")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].set_title("Uncertainty: Correct vs Incorrect", fontweight="bold")
        axes[0, 0].legend()

        # 2. Uncertainty by class
        for cls_idx, name, color in zip([0, 1], config.CLASS_NAMES, ["#e74c3c", "#2ecc71"]):
            mask = labels == cls_idx
            axes[0, 1].hist(stds[mask], bins=25, alpha=0.5, label=name,
                           color=color, density=True)
        axes[0, 1].set_xlabel("Uncertainty (Std)")
        axes[0, 1].set_title("Uncertainty by Class", fontweight="bold")
        axes[0, 1].legend()

        # 3. Mean prediction vs uncertainty (scatter)
        scatter = axes[1, 0].scatter(means, stds, c=correct.astype(float),
                                     cmap="RdYlGn", alpha=0.5, s=15,
                                     edgecolors="none")
        axes[1, 0].axhline(y=self.UNCERTAINTY_THRESHOLDS["MEDIUM"], color="orange",
                           ls="--", alpha=0.7)
        axes[1, 0].axvline(x=0.5, color="gray", ls=":", alpha=0.5)
        axes[1, 0].set_xlabel("Mean P(Drowsy)")
        axes[1, 0].set_ylabel("Uncertainty (Std)")
        axes[1, 0].set_title("Prediction vs Uncertainty\n(green=correct, red=wrong)",
                            fontweight="bold")
        plt.colorbar(scatter, ax=axes[1, 0], label="Correct")

        # 4. Reliability breakdown
        level_names = ["LOW", "MEDIUM", "HIGH"]
        counts = [np.sum(levels == l) for l in level_names]
        accs = []
        for level in level_names:
            level_mask = levels == level
            if level_mask.any():
                accs.append((preds[level_mask] == labels[level_mask]).mean())
            else:
                accs.append(0)

        x = np.arange(len(level_names))
        bars = axes[1, 1].bar(x, counts, color=["#2ecc71", "#f39c12", "#e74c3c"], alpha=0.7)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(level_names)
        axes[1, 1].set_ylabel("Count", color="#3498db")

        ax2 = axes[1, 1].twinx()
        ax2.plot(x, accs, "ko-", markersize=8, linewidth=2)
        ax2.set_ylabel("Accuracy", color="black")
        ax2.set_ylim(0, 1.05)

        for bar, count, acc in zip(bars, counts, accs):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                           f"n={count}", ha="center", fontsize=9)
            ax2.text(bar.get_x() + bar.get_width()/2, acc + 0.03,
                    f"{acc:.2%}", ha="center", fontsize=9, fontweight="bold")

        axes[1, 1].set_title("Samples & Accuracy by Uncertainty Level", fontweight="bold")

        plt.suptitle("Monte Carlo Dropout Uncertainty Analysis",
                    fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_dir / "uncertainty_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved uncertainty_analysis.png")

        return {
            "mean_uncertainty": float(stds.mean()),
            "unreliable_pct": float((~reliable).mean()),
            "accuracy_reliable": float(acc_reliable),
            "accuracy_unreliable": float(acc_unreliable),
        }
