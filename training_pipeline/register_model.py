"""
register_model.py

Step 3 of the training pipeline: orchestrates load -> train -> evaluate ->
pick the best model -> register it in the Hopsworks Model Registry.

Usage:
    export HOPSWORKS_API_KEY="..."
    python register_model.py
"""

import os
import json
import logging
import shutil
import argparse
from pathlib import Path

import joblib
import hopsworks

from load_features import get_train_test_split
from train_models import train_all_models, predict
from evaluate import evaluate_predictions, print_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.environ.get("HOPSWORKS_PROJECT")

REGISTRY_MODEL_NAME = "aqi_karachi_forecaster"


def evaluate_all(models: dict, X_test, y_test) -> dict:
    """Returns {model_name: metrics_dict} for every trained candidate."""
    results = {}
    for name, (model, scaler) in models.items():
        preds = predict(model, X_test, scaler)
        metrics = evaluate_predictions(y_test, preds)
        print_report(name, metrics)
        results[name] = metrics
    return results


def pick_best(results: dict) -> str:
    """Best = lowest mean RMSE across the three horizons."""
    best_name = min(results, key=lambda name: results[name]["mean_rmse"])
    log.info("Best model: %s (mean RMSE=%.3f)", best_name, results[best_name]["mean_rmse"])
    return best_name


def save_model_locally(name: str, model, scaler) -> Path:
    """Saves the winning model (+ its scaler, if any) to disk for the web app
    and for upload to the Hopsworks Model Registry.
    """
    model_dir = MODEL_DIR / name
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)

    if name == "dense_nn":
        model.save(model_dir / "model.keras")
    else:
        joblib.dump(model, model_dir / "model.joblib")

    if scaler is not None:
        joblib.dump(scaler, model_dir / "scaler.joblib")

    log.info("Saved %s to %s", name, model_dir)
    return model_dir


def register_in_hopsworks(name: str, model_dir: Path, metrics: dict):
    if not HOPSWORKS_API_KEY:
        log.warning("HOPSWORKS_API_KEY not set - skipping registry upload, model saved locally only")
        return None

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
    mr = project.get_model_registry()

    flat_metrics = {"mean_rmse": metrics["mean_rmse"], "mean_mae": metrics["mean_mae"], "mean_r2": metrics["mean_r2"]}

    registry_api = mr.tensorflow if name == "dense_nn" else mr.sklearn
    hw_model = registry_api.create_model(
        name=REGISTRY_MODEL_NAME,
        metrics=flat_metrics,
        description=f"Best AQI forecaster ({name}) - 24h/48h/72h horizons for Karachi",
    )
    hw_model.save(str(model_dir))
    log.info("Registered '%s' v%d in Hopsworks Model Registry", hw_model.name, hw_model.version)
    return hw_model


def main():
    parser = argparse.ArgumentParser(description="Train and register the best AQI forecaster")
    parser.add_argument(
        "--max-history-days", type=int, default=None,
        help="Restrict training to the most recent N days (try this if models "
             "underperform a naive persistence baseline - see check_baseline.py)",
    )
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = get_train_test_split(max_history_days=args.max_history_days)

    models = train_all_models(X_train, y_train)
    results = evaluate_all(models, X_test, y_test)

    best_name = pick_best(results)
    best_model, best_scaler = models[best_name]
    best_metrics = results[best_name]

    model_dir = save_model_locally(best_name, best_model, best_scaler)

    # Keep a plain-text record of what won and why, alongside the model files
    with open(model_dir / "metrics.json", "w") as f:
        json.dump({"model_name": best_name, **best_metrics}, f, indent=2)

    register_in_hopsworks(best_name, model_dir, best_metrics)


if __name__ == "__main__":
    main()