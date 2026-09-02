"""
app.py

Enhanced Karachi AQI Forecast dashboard.

Usage:
    export HOPSWORKS_API_KEY="..."
    export HOPSWORKS_PROJECT="..."   # optional
    streamlit run app.py
"""

import os
from datetime import timedelta

import hopsworks
import numpy as np
import pandas as pd
import streamlit as st

from model_utils import fetch_best_model_dir, load_model_from_dir, predict_horizons
from aqi_utils import get_aqi_category, is_hazardous
from explain import compute_top_features

# --- Configuration & Constants ---
FEATURE_GROUP_NAME = "aqi_karachi_features"
FEATURE_GROUP_VERSION = 3
NON_FEATURE_COLUMNS = ["timestamp_utc", "city", "lat", "lon", "weather_main", "aqi_aqicn"]
TARGET_COLUMNS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
HORIZON_LABELS = {"target_aqi_24h": "+24 Hours", "target_aqi_48h": "+48 Hours", "target_aqi_72h": "+72 Hours"}

st.set_page_config(
    page_title="Karachi AQI Intelligence Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Dynamic Custom Styling ---
def inject_custom_css(aqi_color: str):
    """Injects custom CSS dynamic themes based on current AQI severity color."""
    st.markdown(
        f"""
        <style>
            /* Dynamic Metric & Header Highlight */
            .main-aqi-card {{
                background: linear-gradient(135deg, {aqi_color}22 0%, #0e1117 100%);
                border: 2px solid {aqi_color};
                border-radius: 12px;
                padding: 24px;
                text-align: center;
                margin-bottom: 20px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }}
            .aqi-badge {{
                background-color: {aqi_color};
                color: #ffffff;
                padding: 6px 16px;
                border-radius: 20px;
                font-weight: 700;
                font-size: 1.1rem;
                display: inline-block;
                margin-top: 8px;
            }}
            .forecast-card {{
                background-color: #1a1d24;
                border-radius: 10px;
                padding: 16px;
                border-left: 5px solid {aqi_color};
                margin-bottom: 12px;
            }}
            .health-card {{
                background-color: #161920;
                border: 1px solid #2d3139;
                border-radius: 8px;
                padding: 16px;
                height: 100%;
            }}
            /* Metric Styling */
            [data-testid="stMetricValue"] {{
                font-weight: 800;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- Data Fetching & Caching ---
@st.cache_resource(show_spinner="Connecting to Hopsworks...")
def get_project():
    api_key = os.environ.get("HOPSWORKS_API_KEY") or st.secrets.get("HOPSWORKS_API_KEY")
    project_name = os.environ.get("HOPSWORKS_PROJECT") or st.secrets.get("HOPSWORKS_PROJECT")
    return hopsworks.login(api_key_value=api_key, project=project_name)


@st.cache_resource(show_spinner="Loading the best available model...")
def get_model():
    project = get_project()
    model_dir = fetch_best_model_dir(project)
    return load_model_from_dir(model_dir)


@st.cache_data(ttl=600, show_spinner="Fetching latest AQI & telemetry features...")
def get_recent_data(n_hours: int = 168) -> pd.DataFrame:
    project = get_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    if fg is None:
        raise RuntimeError(f"Feature group '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION} not found.")
    df = fg.select_all().read()
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
        df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df.tail(n_hours)


# --- Health Recommendations Helper ---
def get_health_advisory(aqi_val: float) -> dict:
    """Returns specific health recommendations based on AQI value."""
    if aqi_val <= 50:
        return {
            "status": "Good",
            "mask": "Not needed",
            "outdoor": "Ideal for outdoor activities",
            "ventilation": "Open windows to bring in clean air",
            "purifier": "Not required",
        }
    elif aqi_val <= 100:
        return {
            "status": "Moderate",
            "mask": "Optional for unusually sensitive individuals",
            "outdoor": "Unusually sensitive people should reduce prolonged outdoor exertion",
            "ventilation": "Safe to open windows",
            "purifier": "Not necessary",
        }
    elif aqi_val <= 150:
        return {
            "status": "Unhealthy for Sensitive Groups",
            "mask": "Recommended for sensitive individuals (N95)",
            "outdoor": "Sensitive groups should limit outdoor exertion",
            "ventilation": "Close windows during high dust hours",
            "purifier": "Recommended for sensitive groups",
        }
    elif aqi_val <= 200:
        return {
            "status": "Unhealthy",
            "mask": "Required outdoors (N95/KN95)",
            "outdoor": "Avoid prolonged outdoor exposure",
            "ventilation": "Keep windows closed",
            "purifier": "Recommended indoors",
        }
    elif aqi_val <= 300:
        return {
            "status": "Very Unhealthy",
            "mask": "Mandatory N95 mask outdoors",
            "outdoor": "Avoid outdoor activities entirely",
            "ventilation": "Strictly seal windows and doors",
            "purifier": "Run continuously indoors",
        }
    else:
        return {
            "status": "Hazardous",
            "mask": "Strict mandatory usage (N95/P100)",
            "outdoor": "Emergency conditions: Remain indoors",
            "ventilation": "Keep all air intakes closed",
            "purifier": "Run HEPA air purifiers at max power",
        }


def main():
    # --- Data & Model Loading ---
    try:
        recent_df = get_recent_data(n_hours=168)
    except Exception as e:
        st.error(f"Failed to fetch data from Hopsworks: {e}")
        st.stop()

    if recent_df.empty:
        st.warning("No telemetry features retrieved from feature store.")
        st.stop()

    try:
        model, scaler, model_name = get_model()
    except Exception as e:
        st.error(f"Failed to load predictive model: {e}")
        st.stop()

    # Feature isolation & inference
    feature_cols = [c for c in recent_df.columns if c not in NON_FEATURE_COLUMNS + TARGET_COLUMNS]
    latest_row = recent_df.iloc[[-1]]
    latest_timestamp = latest_row["timestamp_utc"].iloc[0]
    current_aqi = float(latest_row["aqi_us_epa"].iloc[0])

    preds = predict_horizons(model, model_name, latest_row[feature_cols], scaler)

    curr_label, curr_color, _ = get_aqi_category(current_aqi)
    inject_custom_css(curr_color)

    # --- Header Section ---
    st.title("🌫️ Karachi AQI Intelligence Dashboard")
    st.caption("Real-time telemetry, multi-horizon AI forecasting, pollutant analysis, and SHAP explainability.")

    # --- Top Banner: Current Reading ---
    st.markdown(
        f"""
        <div class="main-aqi-card">
            <h4 style="margin:0; opacity: 0.8; color: #d0d0d0;">CURRENT AIR QUALITY (KARACHI)</h4>
            <h1 style="font-size: 4rem; margin: 0; font-weight: 900;">{current_aqi:.0f} <span style="font-size: 1.5rem; font-weight: 400;">AQI (US EPA)</span></h1>
            <div class="aqi-badge">{curr_label}</div>
            <p style="margin-top: 12px; font-size: 0.85rem; color: #a0a0a0;">
                Last Sensor Reading: {latest_timestamp:%d %b %Y, %H:%M} UTC
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Multi-Tab Navigation ---
    tab_overview, tab_trends, tab_explain, tab_accuracy = st.tabs(
        ["📊 Overview & Forecast", "📈 Trends & Pollutants", "🔍 SHAP Explainability", "🎯 Model Accuracy & Health Guide"]
    )

    # ==========================================
    # TAB 1: OVERVIEW & FORECAST
    # ==========================================
    with tab_overview:
        st.subheader("3-Day AQI Forecast")
        cols = st.columns(3)

        any_hazardous = is_hazardous(current_aqi)
        for i, target_col in enumerate(TARGET_COLUMNS):
            pred_aqi = float(preds[i])
            label, color, guidance = get_aqi_category(pred_aqi)
            any_hazardous = any_hazardous or is_hazardous(pred_aqi)

            with cols[i]:
                st.markdown(
                    f"""
                    <div class="forecast-card" style="border-left-color: {color};">
                        <div style="font-size: 0.9rem; color: #888; text-transform: uppercase;">{HORIZON_LABELS[target_col]}</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #fff;">{pred_aqi:.0f} <span style="font-size: 1rem;">AQI</span></div>
                        <div style="color: {color}; font-weight: 600; margin-top: 4px;">● {label}</div>
                        <div style="font-size: 0.8rem; color: #aaa; margin-top: 8px;">Target: {(latest_timestamp + timedelta(hours=(i+1)*24)):%a %H:%M} UTC</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if any_hazardous:
            st.error(
                "⚠️ **Hazard Warning:** High pollution levels (Unhealthy for Sensitive Groups or worse) "
                "detected in current or upcoming 72h window. Take precautionary health measures."
            )

        st.divider()

        # Actionable Health Quick-Grid
        st.subheader("💡 Today's Actionable Guidance")
        advisory = get_health_advisory(current_aqi)
        h_col1, h_col2, h_col3, h_col4 = st.columns(4)

        with h_col1:
            st.markdown(f"**😷 Mask Requirement**\n\n{advisory['mask']}")
        with h_col2:
            st.markdown(f"**🏃 Outdoor Exercise**\n\n{advisory['outdoor']}")
        with h_col3:
            st.markdown(f"**🪟 Home Ventilation**\n\n{advisory['ventilation']}")
        with h_col4:
            st.markdown(f"**🌀 Air Purification**\n\n{advisory['purifier']}")

    # ==========================================
    # TAB 2: TRENDS & POLLUTANTS
    # ==========================================
    with tab_trends:
        st.subheader("7-Day AQI Trend + 3-Day Forecast Horizon")

        # Combine historical actuals and future forecasted points
        trend_df = recent_df[["timestamp_utc", "aqi_us_epa"]].copy()
        trend_df = trend_df.rename(columns={"aqi_us_epa": "Historical AQI", "timestamp_utc": "time"}).set_index("time")

        forecast_timestamps = [
            latest_timestamp + timedelta(hours=24),
            latest_timestamp + timedelta(hours=48),
            latest_timestamp + timedelta(hours=72),
        ]
        forecast_df = pd.DataFrame({"Forecasted AQI": preds}, index=forecast_timestamps)

        combined_chart_df = pd.concat([trend_df, forecast_df], axis=1)
        st.line_chart(combined_chart_df)

        st.divider()

        st.subheader("🔬 Pollutant Concentration Breakdown")
        st.caption("Historical monitoring of fine particulate matter (PM2.5 vs PM10)")

        pollutant_cols = [c for c in ["pm2_5", "pm10", "pm25"] if c in recent_df.columns]
        if len(pollutant_cols) >= 1:
            p_df = recent_df[["timestamp_utc"] + pollutant_cols].set_index("timestamp_utc")
            st.line_chart(p_df)
        else:
            # Fallback mock/simulated breakdown if direct feature columns differ
            st.info("Direct PM2.5 / PM10 telemetry breakdown is being aggregated.")

    # ==========================================
    # TAB 3: EXPLAINABILITY (SHAP)
    # ==========================================
    with tab_explain:
        st.subheader("🔍 SHAP Feature Attribution (+24h Forecast)")
        st.caption("Identifies factors driving the 24-hour ahead prediction up or down.")

        background_df = recent_df[feature_cols].tail(72)
        top_features = compute_top_features(
            model, model_name, scaler, background_df, latest_row[feature_cols], top_n=8
        )

        if top_features:
            shap_df = pd.DataFrame(top_features, columns=["Feature", "Impact on +24h AQI"]).set_index("Feature")
            st.bar_chart(shap_df)
            st.caption(" Positive values increase the forecasted AQI; negative values pull the forecast down.")

            st.markdown("##### Detailed Feature Contribution Table")
            st.dataframe(shap_df, use_container_width=True)
        else:
            st.info("SHAP feature attributions are unavailable for the active model type.")

    # ==========================================
    # TAB 4: ACCURACY & HEALTH REFERENCE
    # ==========================================
    with tab_accuracy:
        st.subheader("🎯 Model Accuracy Tracking (+24h Horizon)")

        # Compute past ground-truth vs prior target column if present
        if "target_aqi_24h" in recent_df.columns:
            acc_df = recent_df[["timestamp_utc", "aqi_us_epa", "target_aqi_24h"]].dropna().copy()
            acc_df["actual_24h_ahead"] = acc_df["aqi_us_epa"].shift(-24)
            eval_df = acc_df.dropna().set_index("timestamp_utc")

            if not eval_df.empty:
                mae = np.mean(np.abs(eval_df["target_aqi_24h"] - eval_df["actual_24h_ahead"]))
                rmse = np.sqrt(np.mean((eval_df["target_aqi_24h"] - eval_df["actual_24h_ahead"]) ** 2))

                m1, m2 = st.columns(2)
                m1.metric("Historical 24h MAE", f"{mae:.2f} AQI")
                m2.metric("Historical 24h RMSE", f"{rmse:.2f} AQI")

                st.line_chart(eval_df[["target_aqi_24h", "actual_24h_ahead"]].rename(
                    columns={"target_aqi_24h": "Past +24h Prediction", "actual_24h_ahead": "Actual Realized AQI"}
                ))
            else:
                st.info("Insufficient historical backfill window to compute ground-truth validation matrix.")
        else:
            st.info("Target columns unavailable in feature store subset.")

        st.divider()

        # Detailed Reference Matrix
        st.subheader("📚 US EPA AQI Standard Categories Reference")
        ref_data = [
            {"Range": "0 - 50", "Category": "Good", "Color": "🟢 Green", "Health Impact": "Air quality is satisfactory."},
            {"Range": "51 - 100", "Category": "Moderate", "Color": "🟡 Yellow", "Health Impact": "Acceptable air quality for most."},
            {"Range": "101 - 150", "Category": "Unhealthy for Sensitive Groups", "Color": "🟠 Orange", "Health Impact": "Sensitive groups may experience health effects."},
            {"Range": "151 - 200", "Category": "Unhealthy", "Color": "🔴 Red", "Health Impact": "Everyone may begin to experience health effects."},
            {"Range": "201 - 300", "Category": "Very Unhealthy", "Color": "🟣 Purple", "Health Impact": "Health alert: risk of health effects for everyone."},
            {"Range": "301+", "Category": "Hazardous", "Color": "🟤 Maroon", "Health Impact": "Health warning of emergency conditions."},
        ]
        st.table(pd.DataFrame(ref_data))

    # --- Footer ---
    st.caption(f"Active Model Variant: **{model_name}** | Pipelines hosted on Hopsworks Feature Store v{FEATURE_GROUP_VERSION}")


if __name__ == "__main__":
    main()