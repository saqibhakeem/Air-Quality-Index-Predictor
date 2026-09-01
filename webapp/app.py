"""
app.py

Karachi AQI Forecast dashboard.

Usage:
    export HOPSWORKS_API_KEY="..."
    export HOPSWORKS_PROJECT="..."   # optional
    streamlit run app.py
"""

import os
from datetime import timedelta

import hopsworks
import pandas as pd
import streamlit as st

from model_utils import fetch_best_model_dir, load_model_from_dir, predict_horizons
from aqi_utils import get_aqi_category, is_hazardous
from explain import compute_top_features

FEATURE_GROUP_NAME = "aqi_karachi_features"
FEATURE_GROUP_VERSION = 3  # kept in sync with feature_pipeline/push_to_feature_store.py
NON_FEATURE_COLUMNS = ["timestamp_utc", "city", "lat", "lon", "weather_main", "aqi_aqicn"]
TARGET_COLUMNS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
HORIZON_LABELS = {"target_aqi_24h": "In 24h", "target_aqi_48h": "In 48h", "target_aqi_72h": "In 72h"}

st.set_page_config(page_title="Karachi AQI Forecast", page_icon="\U0001F32B\uFE0F", layout="wide")


@st.cache_resource(show_spinner="Connecting to Hopsworks...")
def get_project():
    api_key = os.environ.get("HOPSWORKS_API_KEY") or st.secrets.get("HOPSWORKS_API_KEY")
    project_name = os.environ.get("HOPSWORKS_PROJECT") or st.secrets.get("HOPSWORKS_PROJECT")
    return hopsworks.login(api_key_value=api_key, project=project_name)


@st.cache_resource(show_spinner="Loading the latest trained model...")
def get_model():
    project = get_project()
    model_dir = fetch_best_model_dir(project)
    return load_model_from_dir(model_dir)


@st.cache_data(ttl=600, show_spinner="Fetching latest AQI data...")
def get_recent_data(n_hours: int = 168) -> pd.DataFrame:
    """Last `n_hours` of feature rows, for the trend chart and SHAP background."""
    project = get_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    if fg is None:
        raise RuntimeError(
            f"Feature group '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION} doesn't exist in "
            "Hopsworks. Check that FEATURE_GROUP_VERSION here matches the version currently "
            "used in feature_pipeline/push_to_feature_store.py, and that push has run at least once."
        )
    df = fg.select_all().read()
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df.tail(n_hours)


def main():
    st.title("\U0001F32B\uFE0F Karachi AQI Forecast")
    st.caption("3-day Air Quality Index forecast, served from a serverless MLOps pipeline.")

    try:
        recent_df = get_recent_data()
    except Exception as e:
        st.error(f"Couldn't reach the Hopsworks feature store: {e}")
        st.stop()

    if recent_df.empty:
        st.warning("No feature data available yet - the pipeline may still be backfilling.")
        st.stop()

    try:
        model, scaler, model_name = get_model()
    except Exception as e:
        st.error(f"Couldn't load a model from the registry: {e}")
        st.stop()

    feature_cols = [c for c in recent_df.columns if c not in NON_FEATURE_COLUMNS + TARGET_COLUMNS]
    latest_row = recent_df.iloc[[-1]]
    latest_timestamp = latest_row["timestamp_utc"].iloc[0]
    current_aqi = latest_row["aqi_us_epa"].iloc[0]

    preds = predict_horizons(model, model_name, latest_row[feature_cols], scaler)

    # --- Current reading + forecast cards ---------------------------------
    st.subheader("Current & Forecasted AQI")
    cols = st.columns(4)

    label, color, guidance = get_aqi_category(current_aqi)
    with cols[0]:
        st.metric("Right now", f"{current_aqi:.0f}")
        st.markdown(f"<span style='color:{color}'>\u25CF</span> **{label}**", unsafe_allow_html=True)
        st.caption(f"as of {latest_timestamp:%Y-%m-%d %H:%M} UTC")

    any_hazardous = is_hazardous(current_aqi)
    for i, target_col in enumerate(TARGET_COLUMNS):
        pred_aqi = float(preds[i])
        label, color, guidance = get_aqi_category(pred_aqi)
        any_hazardous = any_hazardous or is_hazardous(pred_aqi)
        with cols[i + 1]:
            st.metric(HORIZON_LABELS[target_col], f"{pred_aqi:.0f}")
            st.markdown(f"<span style='color:{color}'>\u25CF</span> **{label}**", unsafe_allow_html=True)

    if any_hazardous:
        st.error(
            "\u26A0\uFE0F **Hazard alert:** AQI is forecast to reach Unhealthy-for-Sensitive-Groups "
            "levels or worse in the next 3 days. Sensitive groups should limit prolonged outdoor exertion."
        )

    st.divider()

    # --- Trend chart: recent actual AQI + forecast points ------------------
    st.subheader("Recent trend + forecast")
    trend_df = recent_df[["timestamp_utc", "aqi_us_epa"]].rename(
        columns={"aqi_us_epa": "AQI", "timestamp_utc": "time"}
    ).set_index("time")

    forecast_points = pd.DataFrame(
        {"AQI": preds},
        index=[latest_timestamp + timedelta(hours=24), latest_timestamp + timedelta(hours=48), latest_timestamp + timedelta(hours=72)],
    )
    st.line_chart(pd.concat([trend_df.tail(72), forecast_points]))

    st.divider()

    # --- SHAP explanation for the 24h prediction ----------------------------
    st.subheader("Why this 24h forecast?")
    background_df = recent_df[feature_cols].tail(72)
    top_features = compute_top_features(
        model, model_name, scaler, background_df, latest_row[feature_cols], top_n=5
    )

    if top_features:
        shap_df = pd.DataFrame(top_features, columns=["Feature", "Impact on 24h AQI"]).set_index("Feature")
        st.bar_chart(shap_df)
        st.caption("Positive values push the 24h forecast up; negative values pull it down.")
    else:
        st.info("Feature explanation isn't available for this run.")

    st.caption(f"Model in use: **{model_name}** \u00b7 Data source: OpenWeather + computed EPA AQI")


if __name__ == "__main__":
    main()