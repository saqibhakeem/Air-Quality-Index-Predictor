"""
train_models.py

Step 2 of the training pipeline: train and hyperparameter-tune a few
candidate models for the 3-horizon (24h/48h/72h) AQI forecast.

Models:
- Ridge Regression (linear baseline)
- Random Forest (non-linear, handles feature interactions well on tabular data)
- A small Keras dense network (the "deep learning" candidate)

All three natively support multi-output regression (3 targets at once),
so no need for a separate model per horizon or a MultiOutputRegressor wrapper.

Hyperparameter search uses TimeSeriesSplit (not plain KFold) - shuffled CV
would leak future rows into validation folds via the lag/rolling features.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RANDOM_STATE = 42
N_CV_SPLITS = 4


def make_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def train_ridge(X_train: pd.DataFrame, y_train: pd.DataFrame, scaler: StandardScaler) -> Ridge:
    log.info("Training Ridge (grid search over alpha)...")
    X_scaled = scaler.transform(X_train)

    param_grid = {"alpha": [0.1, 1.0, 10.0, 50.0, 100.0]}
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    search = GridSearchCV(
        Ridge(random_state=RANDOM_STATE), param_grid, cv=tscv,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    search.fit(X_scaled, y_train)
    log.info("Best Ridge alpha: %s", search.best_params_)
    return search.best_estimator_


def train_random_forest(X_train: pd.DataFrame, y_train: pd.DataFrame) -> RandomForestRegressor:
    log.info("Training RandomForest (grid search over depth/estimators)...")

    # Removed unconstrained max_depth=None and raised the min_samples_leaf
    # floor - RandomForest's RMSE blew up badly (5.4 -> 12.6+) once trained
    # on a longer, seasonally-heterogeneous history, a classic sign of a tree
    # ensemble overfitting to training-period specifics rather than learning
    # anything that generalizes across a regime shift.
    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [4, 8, 16],
        "min_samples_leaf": [3, 5, 10],
    }
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    search = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    search.fit(X_train, y_train)
    log.info("Best RandomForest params: %s", search.best_params_)
    return search.best_estimator_


def build_dense_nn(n_features: int, n_outputs: int = 3):
    """Small feed-forward network for multi-output regression.

    Note: a true sequence model (LSTM) would need raw windowed time-steps
    as input. Since our features already encode temporal structure via the
    lag/rolling columns from build_features.py, a dense network on top of
    those engineered features is the more direct "deep learning" fit here
    and avoids reshaping into (samples, timesteps, features) tensors for
    what is still fundamentally a tabular problem.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        keras.Input(shape=(n_features,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_outputs, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_dense_nn(X_train: pd.DataFrame, y_train: pd.DataFrame, scaler: StandardScaler,
                    epochs: int = 100, batch_size: int = 32, validation_split: float = 0.15):
    from tensorflow import keras

    log.info("Training dense NN...")
    X_scaled = scaler.transform(X_train)

    model = build_dense_nn(n_features=X_scaled.shape[1], n_outputs=y_train.shape[1])

    # Don't shuffle - validation_split takes the LAST fraction of the (already
    # chronological) array, which keeps the validation set as the most recent
    # slice rather than a random one.
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    model.fit(
        X_scaled, y_train.to_numpy(),
        epochs=epochs, batch_size=batch_size, validation_split=validation_split,
        shuffle=False, callbacks=[early_stop], verbose=0,
    )
    return model


def train_all_models(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """Returns dict of {model_name: (model, scaler_or_none)}."""
    scaler = make_scaler(X_train)

    models = {
        "ridge": (train_ridge(X_train, y_train, scaler), scaler),
        "random_forest": (train_random_forest(X_train, y_train), None),
        "dense_nn": (train_dense_nn(X_train, y_train, scaler), scaler),
    }
    return models


def predict(model, X: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Uniform prediction interface regardless of whether the model needs scaling."""
    X_input = scaler.transform(X) if scaler is not None else X
    preds = model.predict(X_input)
    return np.asarray(preds)