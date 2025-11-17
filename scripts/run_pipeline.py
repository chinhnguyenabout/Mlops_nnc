"""
Utility script to orchestrate the full MLOps demo workflow.

Steps:
1. Generate (or refresh) the synthetic dataset under `data/`.
2. Run the MLflow training + model registry pipeline.
3. Optionally start the Flask web app.
4. Optionally build / run the Docker image.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "classification_data.csv"
PYTHON_BIN = sys.executable


def _run_command(cmd: Sequence[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    """Execute a command and stream output."""
    display_cmd = " ".join(cmd)
    print(f"[RUN] {display_cmd}")
    subprocess.run(cmd, cwd=cwd or ROOT_DIR, check=True, env=env)


def ensure_dataset(force: bool = False) -> None:
    """Generate the dataset if missing or when force=True."""
    if DATA_PATH.exists() and not force:
        print(f"[DATA] Existing dataset found at {DATA_PATH}")
        return

    sys.path.insert(0, str(ROOT_DIR))
    from mlflow_project.data_generator import generate_classification_data

    df, saved = generate_classification_data(save_path=DATA_PATH)
    print(f"[DATA] Generated dataset with shape {df.shape} -> {saved}")


def run_training() -> None:
    """Trigger the MLflow training pipeline."""
    _run_command([PYTHON_BIN, "-m", "mlflow_project.train"])


def start_flask() -> None:
    """Start the Flask application (blocking)."""
    print("[FLASK] Starting Flask app. Press Ctrl+C to stop.")
    _run_command([PYTHON_BIN, "flask_app/app.py"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Project MLOps workflow.")
    parser.add_argument(
        "--force-data",
        action="store_true",
        help="Regenerate the synthetic dataset even if it already exists.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip the training step (useful when only starting Flask or Docker).",
    )
    parser.add_argument(
        "--start-flask",
        action="store_true",
        help="Start the Flask web application after training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_dataset(force=args.force_data)

    if not args.skip_train:
        run_training()

    if args.start_flask:
        start_flask()

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as err:
        print(f"[ERROR] Command failed with exit code {err.returncode}: {err.cmd}")
        sys.exit(err.returncode)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Pipeline interrupted by user.")
        sys.exit(130)


