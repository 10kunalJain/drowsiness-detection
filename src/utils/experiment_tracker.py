"""
Lightweight Experiment Tracking System.

No MLflow/W&B dependency — just structured JSON logging.
Every training run, benchmark, or improvement iteration gets logged with:
- Timestamp, model name, hyperparameters
- All metrics (train, val, test)
- Artifacts (model paths, plot paths)
- Environment info

Usage:
    tracker = ExperimentTracker()

    with tracker.run("mobilenetv2_phase1") as run:
        run.log_params({"lr": 1e-3, "batch_size": 64, "epochs": 30})
        # ... training ...
        run.log_metrics({"test_auc": 0.96, "test_f1": 0.94})
        run.log_artifact("outputs/models/best.keras")

    # Later: view all experiments
    tracker.print_summary()
"""
import json
import time
import platform
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


class ExperimentRun:
    """A single experiment run."""

    def __init__(self, name: str, run_id: str):
        self.name = name
        self.run_id = run_id
        self.start_time = time.time()
        self.end_time = None
        self.params = {}
        self.metrics = {}
        self.artifacts = []
        self.tags = []
        self.notes = ""

    def log_params(self, params: dict):
        """Log hyperparameters."""
        self.params.update(params)

    def log_metrics(self, metrics: dict):
        """Log evaluation metrics."""
        self.metrics.update(metrics)

    def log_metric(self, key: str, value: float):
        """Log a single metric."""
        self.metrics[key] = value

    def log_artifact(self, path: str):
        """Log path to a saved artifact (model, plot, etc.)."""
        self.artifacts.append(str(path))

    def add_tag(self, tag: str):
        """Add a tag for filtering."""
        self.tags.append(tag)

    def set_notes(self, notes: str):
        """Add free-text notes about the run."""
        self.notes = notes

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "run_id": self.run_id,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "duration_seconds": round(self.end_time - self.start_time, 1) if self.end_time else None,
            "params": self.params,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "tags": self.tags,
            "notes": self.notes,
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
        }


class ExperimentTracker:
    """
    Lightweight JSON-based experiment tracker.

    All runs are saved to a single JSON file for easy inspection.
    """

    def __init__(self, log_path: Path = None):
        self.log_path = log_path or (config.OUTPUT_DIR / "experiments.json")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._runs = self._load_existing()

    def _load_existing(self) -> list:
        """Load existing experiment log."""
        if self.log_path.exists():
            with open(self.log_path) as f:
                return json.load(f)
        return []

    def _save(self):
        """Persist experiment log to disk."""
        with open(self.log_path, "w") as f:
            json.dump(self._runs, f, indent=2, default=str)

    @contextmanager
    def run(self, name: str, tags: list = None):
        """
        Context manager for an experiment run.

        Usage:
            with tracker.run("my_experiment") as run:
                run.log_params({...})
                run.log_metrics({...})
        """
        run_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment = ExperimentRun(name, run_id)
        if tags:
            experiment.tags = tags

        try:
            yield experiment
        finally:
            experiment.end_time = time.time()
            self._runs.append(experiment.to_dict())
            self._save()
            print(f"  Experiment logged: {run_id} "
                  f"({experiment.to_dict()['duration_seconds']}s)")

    def log_quick(self, name: str, params: dict, metrics: dict,
                  tags: list = None, notes: str = ""):
        """Quick one-line experiment logging without context manager."""
        run_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_data = {
            "name": name,
            "run_id": run_id,
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": 0,
            "params": params,
            "metrics": metrics,
            "artifacts": [],
            "tags": tags or [],
            "notes": notes,
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
        }
        self._runs.append(run_data)
        self._save()

    def print_summary(self):
        """Print a formatted summary of all experiments."""
        if not self._runs:
            print("  No experiments logged yet.")
            return

        print("\n" + "=" * 80)
        print("  EXPERIMENT LOG")
        print("=" * 80)
        print(f"\n  Total runs: {len(self._runs)}\n")

        # Table header
        print(f"  {'Run':<35} {'Duration':>8} ", end="")
        # Collect all metric keys
        all_metrics = set()
        for run in self._runs:
            all_metrics.update(run.get("metrics", {}).keys())
        metric_keys = sorted(all_metrics)[:6]  # Show top 6 metrics
        for key in metric_keys:
            print(f" {key:>10}", end="")
        print()
        print(f"  {'─'*75}")

        for run in self._runs:
            name = run["run_id"][:35]
            duration = run.get("duration_seconds", "?")
            if isinstance(duration, (int, float)):
                dur_str = f"{duration:.0f}s"
            else:
                dur_str = "?"
            print(f"  {name:<35} {dur_str:>8} ", end="")
            for key in metric_keys:
                val = run.get("metrics", {}).get(key)
                if val is not None and isinstance(val, (int, float)):
                    print(f" {val:>10.4f}", end="")
                else:
                    print(f" {'—':>10}", end="")
            print()

        # Best run by each metric
        print(f"\n  {'─'*75}")
        print("  Best runs:")
        for key in metric_keys:
            best_run = None
            best_val = -float("inf")
            for run in self._runs:
                val = run.get("metrics", {}).get(key)
                if val is not None and isinstance(val, (int, float)) and val > best_val:
                    best_val = val
                    best_run = run["run_id"]
            if best_run:
                print(f"    {key}: {best_val:.4f} ({best_run})")

    def get_best_run(self, metric: str) -> dict:
        """Get the run with the highest value of a given metric."""
        best = None
        best_val = -float("inf")
        for run in self._runs:
            val = run.get("metrics", {}).get(metric, None)
            if val is not None and val > best_val:
                best_val = val
                best = run
        return best
