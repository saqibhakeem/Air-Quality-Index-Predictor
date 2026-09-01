"""
explain.py

SHAP feature-importance explanation for the current 24h-ahead prediction.
Only the 24h horizon is explained (not all three) - it's the most
actionable one for a "why is today's forecast what it is" dashboard panel,
and keeps this fast enough to run on every page load.
"""

import logging

import numpy as np
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HORIZON_INDEX = 0  # index of target_aqi_24h within TARGET_COLUMNS


def _make_predict_fn(model, model_name: str, scaler):
    """Returns a plain (n, features) -> (n,) function for the 24h horizon,
    handling scaling internally so the explainer never has to know about it.
    """
    def predict_fn(X: np.ndarray) -> np.ndarray:
        X_input = scaler.transform(X) if scaler is not None else X
        if model_name == "dense_nn":
            preds = model.predict(X_input, verbose=0)
        else:
            preds = model.predict(X_input)
        return np.asarray(preds)[:, HORIZON_INDEX]
    return predict_fn


def compute_top_features(
    model, model_name: str, scaler,
    background_df: pd.DataFrame, instance_df: pd.DataFrame,
    top_n: int = 5,
) -> list:
    """Returns [(feature_name, shap_value), ...] for the single instance row,
    sorted by absolute contribution, largest first.

    background_df: a sample of recent rows used as the reference distribution
    instance_df: the single row being predicted right now (same columns)
    """
    feature_names = list(instance_df.columns)
    background = background_df.to_numpy()
    instance = instance_df.to_numpy()

    try:
        if model_name == "ridge":
            scaled_bg = scaler.transform(background) if scaler is not None else background
            explainer = shap.LinearExplainer(model, scaled_bg)
            scaled_instance = scaler.transform(instance) if scaler is not None else instance
            raw_values = explainer.shap_values(scaled_instance)
            sv = np.asarray(raw_values)[..., HORIZON_INDEX] if np.asarray(raw_values).ndim == 3 else np.asarray(raw_values)

        elif model_name == "random_forest":
            explainer = shap.TreeExplainer(model)
            raw_values = explainer.shap_values(instance)
            sv = np.asarray(raw_values)[..., HORIZON_INDEX] if np.asarray(raw_values).ndim == 3 else np.asarray(raw_values)

        else:  # dense_nn or anything else - model-agnostic fallback
            predict_fn = _make_predict_fn(model, model_name, scaler)
            bg_sample = background[: min(50, len(background))]  # keep KernelExplainer fast
            explainer = shap.KernelExplainer(predict_fn, bg_sample)
            sv = explainer.shap_values(instance, silent=True)
            sv = np.asarray(sv)

        sv = np.asarray(sv).reshape(-1)  # single instance -> (n_features,)
        pairs = sorted(zip(feature_names, sv), key=lambda p: abs(p[1]), reverse=True)
        return pairs[:top_n]

    except Exception:
        log.exception("SHAP explanation failed - dashboard will hide this panel")
        return []
