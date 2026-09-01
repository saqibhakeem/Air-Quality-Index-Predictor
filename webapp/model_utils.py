"""
model_utils.py

Loads the best registered model from Hopsworks (whichever type won
training - Ridge, RandomForest, or the dense NN) and provides a uniform
predict() interface regardless of which one it turns out to be.
"""
import shutil
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REGISTRY_MODEL_NAME = "aqi_karachi_forecaster"
TARGET_COLUMNS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]


def fetch_best_model_dir(project, local_dir: str = "downloaded_model") -> Path:
    """Downloads the best (lowest mean_rmse) registered model and returns its local path."""
    mr = project.get_model_registry()
    hw_model = mr.get_best_model(name=REGISTRY_MODEL_NAME, metric="mean_rmse", direction="min")

    if hw_model is None:
        raise RuntimeError(
            f"No model named '{REGISTRY_MODEL_NAME}' found in the registry - "
            "run training_pipeline/register_model.py at least once first."
        )

    local_path_obj = Path(local_dir)

    if local_path_obj.exists():
        log.info(f'Local dire   ctory "{local_dir}" already exists - deleting it to download the latest model.')
        shutil.rmtree(local_path_obj)

    local_path = hw_model.download(local_path=local_dir)
    log.info("Downloaded model '%s' v%d to %s", hw_model.name, hw_model.version, local_path)
    return Path(local_path)


def load_model_from_dir(model_dir: Path):
    """Loads model + scaler (if any) + metadata from a downloaded model directory.

    Returns (model, scaler_or_none, model_name).
    """
    metrics_path = model_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            model_name = json.load(f)["model_name"]
    elif (model_dir / "model.keras").exists():
        model_name = "dense_nn"
    else:
        model_name = "sklearn_model"  # fallback label, doesn't affect loading logic below

    scaler = None
    scaler_path = model_dir / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)

    if model_name == "dense_nn" or (model_dir / "model.keras").exists():
        from tensorflow import keras
        model = keras.models.load_model(model_dir / "model.keras")
    else:
        model = joblib.load(model_dir / "model.joblib")

    log.info("Loaded model '%s' (scaler=%s)", model_name, scaler is not None)
    return model, scaler, model_name


def predict_horizons(model, model_name: str, X_row: pd.DataFrame, scaler=None) -> np.ndarray:
    """Predicts [aqi_24h, aqi_48h, aqi_72h] for a single feature row.

    X_row must be a DataFrame with exactly one row and the same feature
    columns (order doesn't matter as long as names match) used at training time.
    """
    X_input = scaler.transform(X_row) if scaler is not None else X_row
    preds = model.predict(X_input, verbose=0) if model_name == "dense_nn" else model.predict(X_input)
    return np.asarray(preds).reshape(-1)  # flatten (1, 3) -> (3,)
