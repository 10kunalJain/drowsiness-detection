"""
Multi-Model Benchmarking Pipeline — compare architectures on accuracy, AUC, and latency.

Compares:
- CustomCNN: Purpose-built for 64x64 grayscale eye images
- MobileNetV2: Lightweight transfer learning baseline
- ResNet50V2: Deeper transfer learning baseline

Each model is trained for a fixed number of epochs under identical conditions.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.data.dataset import get_augmentation_layer
from src.models.drowsiness_model import build_model, compile_model


def _measure_latency(model: tf.keras.Model, input_shape: tuple,
                     n_runs: int = 50) -> float:
    """Measure average inference latency in milliseconds."""
    dummy = np.random.rand(1, *input_shape).astype(np.float32)
    for _ in range(5):
        model.predict(dummy, verbose=0)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=0)
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times)


def _prepare_data_for_model(data: dict, model_type: str):
    """
    Prepare data for a specific model type.
    Transfer learning models need 3-channel 96x96 input;
    custom CNN uses native 1-channel 64x64.
    """
    if model_type == "custom_cnn":
        return data  # Already in correct format
    else:
        # Transfer learning: need to upscale and convert to 3-channel
        import cv2
        result = {}
        for key in data:
            if key.startswith("X_"):
                imgs = data[key]
                # Resize to 96x96 and convert to 3-channel
                resized = []
                for img in imgs:
                    if img.shape[-1] == 1:
                        gray = img[:, :, 0]
                    else:
                        gray = img[:, :, 0]
                    gray_resized = cv2.resize(gray, (96, 96))
                    rgb = np.stack([gray_resized] * 3, axis=-1)
                    resized.append(rgb)
                result[key] = np.array(resized, dtype=np.float32)
            else:
                result[key] = data[key]
        return result


def run_benchmark(data: dict, class_weights: dict = None) -> dict:
    """Train and evaluate all candidate models under identical conditions."""
    augment = get_augmentation_layer()
    results = {}

    for model_name, model_config in config.BENCHMARK_MODELS.items():
        model_type = model_config["type"]

        print(f"\n{'─'*50}")
        print(f"  Benchmarking: {model_name} ({model_type})")
        print(f"{'─'*50}")

        # Prepare data for this model type
        model_data = _prepare_data_for_model(data, model_type)

        model = build_model(model_type=model_type)
        total_params = model.count_params()
        model = compile_model(model)

        input_shape = model.input_shape[1:]

        # Build datasets
        train_ds = tf.data.Dataset.from_tensor_slices(
            (model_data["X_train"], model_data["y_train"])
        )
        train_ds = train_ds.shuffle(len(model_data["X_train"]), seed=config.RANDOM_SEED)
        train_ds = train_ds.batch(config.BATCH_SIZE)
        train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y),
                                num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices(
            (model_data["X_val"], model_data["y_val"])
        )
        val_ds = val_ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # Train
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.BENCHMARK_EPOCHS,
            class_weight=class_weights,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_auc", patience=5,
                    restore_best_weights=True, mode="max"
                ),
            ],
            verbose=1,
        )

        # Evaluate
        y_prob = model.predict(
            model_data["X_test"], batch_size=config.BATCH_SIZE, verbose=0
        ).flatten()
        y_pred = (y_prob > 0.5).astype(int)

        auc_score = roc_auc_score(model_data["y_test"], y_prob)
        f1 = f1_score(model_data["y_test"], y_pred, average="weighted")
        accuracy = (y_pred == model_data["y_test"]).mean()

        latency = _measure_latency(model, input_shape)

        save_path = config.MODEL_DIR / f"benchmark_{model_name}.keras"
        model.save(save_path)
        model_size_mb = save_path.stat().st_size / (1024 * 1024)

        results[model_name] = {
            "auc": auc_score,
            "f1": f1,
            "accuracy": accuracy,
            "latency_ms": latency,
            "model_size_mb": model_size_mb,
            "total_params": total_params,
            "best_val_auc": max(history.history.get("val_auc", [0])),
        }

        print(f"  AUC: {auc_score:.4f} | F1: {f1:.4f} | "
              f"Latency: {latency:.1f}ms | Size: {model_size_mb:.1f}MB")

    _plot_benchmark_results(results)
    _print_recommendation(results)

    return results


def _plot_benchmark_results(results: dict):
    """Generate comparison visualization."""
    config.PLOT_DIR.mkdir(parents=True, exist_ok=True)

    models = list(results.keys())
    metrics = {
        "AUC-ROC": [results[m]["auc"] for m in models],
        "F1-Score": [results[m]["f1"] for m in models],
        "Accuracy": [results[m]["accuracy"] for m in models],
    }

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    x = np.arange(len(models))
    width = 0.25
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for i, (metric_name, values) in enumerate(metrics.items()):
        bars = axes[0].bar(x + i * width, values, width, label=metric_name,
                          color=colors[i], alpha=0.8)
        for bar, val in zip(bars, values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                        f"{val:.3f}", ha="center", fontsize=7)

    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(models, fontsize=9)
    axes[0].set_title("Classification Metrics", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0.4, 1.05)

    latencies = [results[m]["latency_ms"] for m in models]
    bars = axes[1].bar(models, latencies, color=colors[:len(models)], alpha=0.8)
    for bar, val in zip(bars, latencies):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}ms", ha="center", fontsize=9)
    axes[1].set_title("Inference Latency (lower = better)", fontweight="bold")

    sizes = [results[m]["model_size_mb"] for m in models]
    bars = axes[2].bar(models, sizes, color=colors[:len(models)], alpha=0.8)
    for bar, val in zip(bars, sizes):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}MB", ha="center", fontsize=9)
    axes[2].set_title("Model Size (lower = better)", fontweight="bold")

    for m, color in zip(models, colors):
        axes[3].scatter(results[m]["latency_ms"], results[m]["auc"],
                       s=results[m]["model_size_mb"] * 20, color=color,
                       label=m, alpha=0.8, edgecolors="black")
    axes[3].set_xlabel("Latency (ms)")
    axes[3].set_ylabel("AUC-ROC")
    axes[3].set_title("Efficiency Frontier\n(size = model MB)", fontweight="bold")
    axes[3].legend(fontsize=8)

    plt.suptitle("Model Architecture Benchmark", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(config.PLOT_DIR / "model_benchmark.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved model_benchmark.png")


def _print_recommendation(results: dict):
    """Print model selection recommendation."""
    print("\n" + "=" * 60)
    print("MODEL SELECTION RECOMMENDATION")
    print("=" * 60)

    for name, r in results.items():
        max_latency = max(res["latency_ms"] for res in results.values())
        latency_score = 1 - (r["latency_ms"] / max_latency)
        r["composite_score"] = 0.5 * r["auc"] + 0.2 * r["f1"] + 0.3 * latency_score

    best = max(results, key=lambda m: results[m]["composite_score"])

    print(f"\n  Recommended: {best}")
    print(f"\n  {'Model':<18} {'AUC':>6} {'F1':>6} {'Latency':>8} {'Size':>7} {'Score':>7}")
    print(f"  {'─'*52}")
    for name in sorted(results, key=lambda m: results[m]["composite_score"], reverse=True):
        r = results[name]
        marker = " *" if name == best else ""
        print(f"  {name:<18} {r['auc']:>6.4f} {r['f1']:>6.4f} "
              f"{r['latency_ms']:>6.1f}ms {r['model_size_mb']:>5.1f}MB "
              f"{r['composite_score']:>7.4f}{marker}")
