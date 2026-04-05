"""
Main training script — full drowsiness detection pipeline.

Usage:
    python train.py                     # Full pipeline (all 12 steps)
    python train.py --skip-bench        # Skip model benchmarking (faster)
    python train.py --skip-lstm         # Skip LSTM temporal head training
    python train.py --resume-from 7     # Resume from step 7 (loads saved model)

Pipeline:
     1. Load data & EDA
     2. Data quality (imbalance, bias)
     3. Multi-model benchmark
     4. Train best model
     5. Evaluate on test set
     6. Grad-CAM interpretability
     7. Error analysis + label noise
     8. Uncertainty estimation (MC Dropout)
     9. Robustness testing
    10. Failure → Improvement loop
    11. LSTM temporal sequence head
    12. Failure narrative generation
"""
import os
import json
import argparse
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import config
from src.data.dataset import load_dataset, run_eda
from src.data.data_quality import (
    analyze_class_imbalance, detect_label_noise, profile_dataset_bias,
)
from src.engine.trainer import train, evaluate, plot_training_history
from src.engine.benchmark import run_benchmark
from src.engine.improvement import (
    find_optimal_threshold, retrain_with_improvements,
)
from src.models.uncertainty import MCDropoutEstimator
from src.analysis.gradcam import generate_gradcam_grid, generate_comparative_gradcam
from src.analysis.error_analysis import analyze_errors
from src.analysis.robustness import run_robustness_test
from src.analysis.failure_narrative import generate_failure_narrative
from src.utils.experiment_tracker import ExperimentTracker


# ── Checkpoint helpers ─────────────────────────────────────────────────
CHECKPOINT_PATH = config.OUTPUT_DIR / "pipeline_checkpoint.json"


def save_checkpoint(step: int, state: dict):
    """Save pipeline state so we can resume from any step."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {"completed_step": step, **state}
    # Convert non-serializable types
    serializable = {}
    for k, v in checkpoint.items():
        if isinstance(v, np.ndarray):
            continue  # Skip arrays — they'll be recomputed or loaded from model
        elif isinstance(v, (dict, list, str, int, float, bool, type(None))):
            serializable[k] = v
        else:
            serializable[k] = str(v)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(serializable, f, indent=2, default=str)


def load_checkpoint() -> dict:
    """Load last checkpoint."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"completed_step": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bench", action="store_true",
                        help="Skip model benchmarking")
    parser.add_argument("--skip-lstm", action="store_true",
                        help="Skip LSTM temporal head training")
    parser.add_argument("--resume-from", type=int, default=None,
                        help="Resume from step N (loads saved model for steps > 5)")
    args = parser.parse_args()

    tracker = ExperimentTracker()

    # Determine start step
    if args.resume_from is not None:
        start_step = args.resume_from
        print(f"\n  Resuming from step {start_step}")
    else:
        checkpoint = load_checkpoint()
        start_step = 1  # Always start fresh unless --resume-from is given

    print("=" * 60)
    print("  DRIVER DROWSINESS DETECTION SYSTEM")
    print("  Full Training & Analysis Pipeline (12 steps)")
    print("=" * 60)

    # ── Always needed: load data ───────────────────────────────────────
    print("\n[1/12] Loading dataset...")
    data = load_dataset()

    # ── Data quality (needed for class_weights) ────────────────────────
    imbalance = None
    class_weights = None
    bias_report = None

    if start_step <= 2:
        print("\n[2/12] Running exploratory data analysis...")
        run_eda(data)
        save_checkpoint(2, {})

    if start_step <= 3:
        print("\n[3/12] Analyzing data quality...")
        imbalance = analyze_class_imbalance(data["y_train"], data["y_val"], data["y_test"])
        class_weights = imbalance["class_weights"] if imbalance["strategy"] != "none" else None
        bias_report = profile_dataset_bias(data["X_train"], data["y_train"])
        save_checkpoint(3, {
            "class_weights": class_weights,
            "imbalance_ratio": imbalance["imbalance_ratio"],
        })
    else:
        # Recompute lightweight values needed downstream
        imbalance = analyze_class_imbalance(data["y_train"], data["y_val"], data["y_test"])
        class_weights = imbalance["class_weights"] if imbalance["strategy"] != "none" else None
        bias_report = {"brightness_gap": 0}  # Placeholder if skipped

    # ── Benchmarking ───────────────────────────────────────────────────
    benchmark_results = None
    if start_step <= 4 and not args.skip_bench:
        print("\n[4/12] Benchmarking model architectures...")
        with tracker.run("model_benchmark", tags=["benchmark"]) as run:
            benchmark_results = run_benchmark(data, class_weights=class_weights)
            for model_name, metrics in benchmark_results.items():
                run.log_metrics({
                    f"{model_name}_auc": metrics["auc"],
                    f"{model_name}_f1": metrics["f1"],
                    f"{model_name}_latency_ms": metrics["latency_ms"],
                })
        save_checkpoint(4, {})
    elif start_step <= 4:
        print("\n[4/12] Skipping benchmark (--skip-bench)")

    # ── Training ───────────────────────────────────────────────────────
    import tensorflow as tf
    model = None

    if start_step <= 5:
        print("\n[5/12] Training MobileNetV2 (full two-phase pipeline)...")
        with tracker.run("mobilenetv2_full_train", tags=["training", "main"]) as run:
            run.log_params({
                "model": "MobileNetV2",
                "input_size": config.MODEL_INPUT_SIZE,
                "batch_size": config.BATCH_SIZE,
                "epochs": config.EPOCHS,
                "lr_phase1": config.LEARNING_RATE,
                "lr_phase2": config.FINE_TUNE_LR,
                "fine_tune_at": config.FINE_TUNE_AT_EPOCH,
                "class_weights": str(class_weights),
            })
            model, history = train(data, class_weights=class_weights)
            plot_training_history(history)
            run.log_metrics({
                "final_train_auc": history["auc"][-1],
                "final_val_auc": history["val_auc"][-1],
            })
            run.log_artifact(str(config.MODEL_DIR / "drowsiness_detector.keras"))
        save_checkpoint(5, {})
    else:
        # Load saved model for resume
        model_path = config.MODEL_DIR / "drowsiness_detector.keras"
        if not model_path.exists():
            print(f"\n  ERROR: Cannot resume — no saved model at {model_path}")
            print(f"  Run full pipeline first (without --resume-from)")
            return
        print(f"\n[5/12] Loading saved model from {model_path}...")
        model = tf.keras.models.load_model(model_path)

    # ── Evaluation ─────────────────────────────────────────────────────
    eval_results = None
    y_prob = None

    if start_step <= 6:
        print("\n[6/12] Evaluating on test set...")
        eval_results = evaluate(model, data)
        tracker.log_quick("test_evaluation", {}, {
            "test_auc": eval_results["roc_auc"],
            "test_accuracy": eval_results["report"]["accuracy"],
        }, tags=["evaluation"])
        y_prob = eval_results["y_prob"]
        # Save y_prob for resume
        np.save(config.OUTPUT_DIR / "y_prob.npy", y_prob)
        save_checkpoint(6, {"test_auc": eval_results["roc_auc"]})
    else:
        # Load saved predictions
        y_prob_path = config.OUTPUT_DIR / "y_prob.npy"
        if y_prob_path.exists():
            y_prob = np.load(y_prob_path)
            print(f"\n[6/12] Loaded saved predictions ({len(y_prob)} samples)")
        else:
            print("\n[6/12] Re-running evaluation...")
            eval_results = evaluate(model, data)
            y_prob = eval_results["y_prob"]
            np.save(config.OUTPUT_DIR / "y_prob.npy", y_prob)

    # ── Grad-CAM ───────────────────────────────────────────────────────
    if start_step <= 7:
        print("\n[7/12] Generating Grad-CAM interpretability...")
        generate_gradcam_grid(model, data["X_test"], data["y_test"], y_prob)
        generate_comparative_gradcam(model, data["X_test"], data["y_test"], y_prob)
        save_checkpoint(7, {})

    # ── Error Analysis ─────────────────────────────────────────────────
    noise_report = {"total_flagged": 0}
    if start_step <= 8:
        print("\n[8/12] Running error analysis...")
        analyze_errors(data["X_test"], data["y_test"], y_prob)
        noise_report = detect_label_noise(model, data["X_train"], data["y_train"])
        save_checkpoint(8, {"noisy_labels": noise_report["total_flagged"]})

    # ── Uncertainty Estimation ─────────────────────────────────────────
    uncertainty_stats = None
    if start_step <= 9:
        print("\n[9/12] Running uncertainty estimation (Monte Carlo Dropout)...")
        mc_estimator = MCDropoutEstimator(model, n_samples=30)
        uncertainty_stats = mc_estimator.analyze_uncertainty_distribution(
            data["X_test"], data["y_test"]
        )
        tracker.log_quick("uncertainty_analysis", {"mc_samples": 30}, {
            "mean_uncertainty": uncertainty_stats["mean_uncertainty"],
            "unreliable_pct": uncertainty_stats["unreliable_pct"],
            "accuracy_reliable": uncertainty_stats["accuracy_reliable"],
        }, tags=["uncertainty"])
        save_checkpoint(9, uncertainty_stats)
    else:
        uncertainty_stats = {
            "mean_uncertainty": 0, "unreliable_pct": 0,
            "accuracy_reliable": 0, "accuracy_unreliable": 0,
        }

    # ── Robustness Testing ─────────────────────────────────────────────
    robustness_results = None
    if start_step <= 10:
        print("\n[10/12] Running robustness & distribution shift tests...")
        robustness_results = run_robustness_test(model, data["X_test"], data["y_test"])
        for corr_name, severities in robustness_results.items():
            drop = severities.get(0.8, {}).get("acc_drop", 0)
            tracker.log_quick(f"robustness_{corr_name}", {"severity": 0.8}, {
                "accuracy_drop": drop,
            }, tags=["robustness"])
        save_checkpoint(10, {})

    # ── Improvement Loop ───────────────────────────────────────────────
    before_metrics = {"auc": 0, "f1": 0, "recall_drowsy": 0}
    after_metrics = {"auc": 0, "f1": 0, "recall_drowsy": 0}
    optimal_threshold = 0.5

    if start_step <= 11:
        print("\n[11/12] Running improvement loop...")
        optimal_threshold = find_optimal_threshold(data["y_test"], y_prob, metric="f1")

        with tracker.run("improvement_loop", tags=["improvement"]) as run:
            run.log_params({"optimal_threshold": optimal_threshold})
            model, before_metrics, after_metrics = retrain_with_improvements(
                model, data,
                class_weights=class_weights,
                optimal_threshold=optimal_threshold,
            )
            run.log_metrics({
                "before_auc": before_metrics["auc"],
                "after_auc": after_metrics["auc"],
                "before_f1": before_metrics["f1"],
                "after_f1": after_metrics["f1"],
                "auc_improvement": after_metrics["auc"] - before_metrics["auc"],
                "f1_improvement": after_metrics["f1"] - before_metrics["f1"],
            })

        from src.models.drowsiness_model import save_model
        save_model(model, name="drowsiness_detector_improved")
        save_checkpoint(11, {
            "optimal_threshold": optimal_threshold,
            "before_auc": before_metrics["auc"],
            "after_auc": after_metrics["auc"],
        })

    # ── LSTM Temporal Head ─────────────────────────────────────────────
    if start_step <= 12 and not args.skip_lstm:
        print("\n[12a/12] Training LSTM temporal sequence head...")
        from src.models.temporal_lstm import train_temporal_model
        with tracker.run("lstm_temporal", tags=["temporal", "lstm"]) as run:
            run.log_params({"seq_length": 15, "lstm_units": 64})
            lstm_model, lstm_history = train_temporal_model(
                model, data["X_train"], data["y_train"],
                data["X_val"], data["y_val"],
            )
            run.log_metrics({
                "val_accuracy": lstm_history["val_accuracy"][-1],
                "val_auc": lstm_history["val_auc"][-1],
            })
            run.log_artifact(str(config.MODEL_DIR / "temporal_lstm.keras"))
    elif args.skip_lstm:
        print("\n[12a/12] Skipping LSTM (--skip-lstm)")

    # ── Failure Narrative ──────────────────────────────────────────────
    print("\n[12b/12] Generating failure analysis narrative...")
    generate_failure_narrative(
        y_test=data["y_test"],
        y_prob=y_prob,
        X_test=data["X_test"],
        robustness_results=robustness_results,
        improvement_before=before_metrics,
        improvement_after=after_metrics,
        uncertainty_stats=uncertainty_stats,
    )

    # ── Experiment Summary ─────────────────────────────────────────────
    tracker.print_summary()
    save_checkpoint(12, {"status": "complete"})

    # ── Final Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE — ALL 12 STEPS")
    print("=" * 60)
    print(f"\n  Data Quality:")
    print(f"    Class imbalance ratio: {imbalance['imbalance_ratio']:.2f}:1")
    if isinstance(bias_report, dict) and "brightness_gap" in bias_report:
        print(f"    Brightness bias: {bias_report['brightness_gap']:.3f}")
    print(f"    Suspected noisy labels: {noise_report['total_flagged']}")
    if uncertainty_stats:
        print(f"\n  Uncertainty:")
        print(f"    Unreliable predictions: {uncertainty_stats['unreliable_pct']:.1%}")
        print(f"    Accuracy (reliable only): {uncertainty_stats['accuracy_reliable']:.4f}")
    print(f"\n  Model Performance:")
    print(f"    Before improvement — AUC: {before_metrics['auc']:.4f}, F1: {before_metrics['f1']:.4f}")
    print(f"    After improvement  — AUC: {after_metrics['auc']:.4f}, F1: {after_metrics['f1']:.4f}")
    print(f"    Optimal threshold: {optimal_threshold:.2f} (vs default 0.50)")
    print(f"\n  Outputs:")
    print(f"    Models:        {config.MODEL_DIR}")
    print(f"    Plots:         {config.PLOT_DIR}")
    print(f"    Grad-CAM:      {config.GRADCAM_DIR}")
    print(f"    Error analysis:{config.ERROR_DIR}")
    print(f"    Experiments:   {config.OUTPUT_DIR / 'experiments.json'}")
    print(f"    Narrative:     {config.OUTPUT_DIR / 'failure_narrative.md'}")
    print(f"\n  Run 'python detect.py' for real-time inference")
    print(f"  Run 'python detect.py --multimodal' for multi-signal mode")


if __name__ == "__main__":
    main()
