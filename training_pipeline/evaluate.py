"""
evaluate.py

Shared metrics for the 3-horizon (24h/48h/72h) AQI forecasting models.
Works for any model whose .predict() (or predict-like output) returns an
array shaped (n_samples, 3) matching TARGET_COLUMNS order.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TARGET_COLUMNS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]


def evaluate_predictions(y_true: pd.DataFrame, y_pred: np.ndarray) -> dict:
    """Compute RMSE/MAE/R² per horizon, plus a mean-RMSE summary metric
    used to rank models against each other.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    per_horizon = {}
    for i, col in enumerate(TARGET_COLUMNS):
        rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        mae = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        r2 = float(r2_score(y_true[:, i], y_pred[:, i]))
        per_horizon[col] = {"rmse": round(rmse, 3), "mae": round(mae, 3), "r2": round(r2, 3)}

    mean_rmse = float(np.mean([m["rmse"] for m in per_horizon.values()]))
    mean_mae = float(np.mean([m["mae"] for m in per_horizon.values()]))
    mean_r2 = float(np.mean([m["r2"] for m in per_horizon.values()]))

    return {
        "per_horizon": per_horizon,
        "mean_rmse": round(mean_rmse, 3),
        "mean_mae": round(mean_mae, 3),
        "mean_r2": round(mean_r2, 3),
    }


def print_report(model_name: str, metrics: dict) -> None:
    print(f"\n--- {model_name} ---")
    for horizon, m in metrics["per_horizon"].items():
        print(f"  {horizon:20s} RMSE={m['rmse']:7.3f}  MAE={m['mae']:7.3f}  R2={m['r2']:6.3f}")
    print(f"  {'MEAN':20s} RMSE={metrics['mean_rmse']:7.3f}  MAE={metrics['mean_mae']:7.3f}  R2={metrics['mean_r2']:6.3f}")
