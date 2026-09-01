# Karachi AQI Forecasting System — Project Report

**A serverless MLOps pipeline predicting Air Quality Index 24h, 48h, and 72h ahead**

---

## 1. Overview

This project implements an end-to-end system that forecasts Karachi's Air
Quality Index (AQI) for the next three days, built entirely on free-tier
serverless infrastructure. It covers all four deliverables from the original
brief: an automated feature pipeline, a training pipeline that compares
statistical, classical ML, and deep learning models, a CI/CD layer that keeps
both running unattended, and an interactive dashboard with explainability and
hazard alerting.

| Component | Tool | Role |
|---|---|---|
| Weather + pollutant data | OpenWeather API | Live + historical PM2.5, PM10, NO2, SO2, CO, O3, temperature, humidity, wind, pressure |
| Historical weather backfill | Open-Meteo Archive API (free, no key) | Historical temp/humidity/wind/pressure for backfill |
| AQI ground truth | Computed EPA AQI (from PM2.5/PM10) | Target variable, defined identically for live and historical data |
| Feature store + model registry | Hopsworks (free tier) | Central store for engineered features and trained models |
| Orchestration | GitHub Actions | Hourly feature runs, daily training runs |
| Modeling | scikit-learn (Ridge, RandomForest) + TensorFlow/Keras | Statistical → classical ML → deep learning comparison |
| Explainability | SHAP | Per-feature contribution to the 24h forecast |
| Dashboard | Streamlit | Live predictions, trend chart, hazard alerts |

---

## 2. Architecture

```
                 ┌──────────────────┐
                 │ OpenWeather API   │──────┐
                 │ (live + history)  │      │
                 └──────────────────┘      │
                                            ▼
                 ┌──────────────────┐   ┌─────────────────────┐
                 │ Open-Meteo API    │──▶│  Feature Pipeline    │
                 │ (historical only) │   │  fetch → build →     │
                 └──────────────────┘   │  push                │
                                         └──────────┬───────────┘
                                                     │ features + targets
                                                     ▼
                                         ┌─────────────────────┐
                                         │  Hopsworks            │
                                         │  Feature Store        │
                                         │  + Model Registry     │
                                         └──────────┬───────────┘
                                    ┌────────────────┼────────────────┐
                                    ▼                                 ▼
                        ┌─────────────────────┐         ┌─────────────────────┐
                        │  Training Pipeline    │         │  Streamlit Web App   │
                        │  load → train →       │────────▶│  predict → explain   │
                        │  evaluate → register  │  model  │  → alert             │
                        └─────────────────────┘         └─────────────────────┘

     GitHub Actions: Feature Pipeline runs hourly, Training Pipeline runs daily
```

### Repository structure

```
aqi_predictor/
├── feature_pipeline/
│   ├── fetch_raw_data.py        # live weather + pollutant fetch, EPA AQI computed per row
│   ├── epa_aqi.py                # shared EPA AQI breakpoint calculator
│   ├── backfill_historical.py    # historical backfill (OpenWeather + Open-Meteo)
│   ├── build_features.py         # time features, lags, rolling stats, 3 targets
│   └── push_to_feature_store.py  # writes to Hopsworks Feature Group + Feature View
├── training_pipeline/
│   ├── load_features.py          # reads Feature View, chronological train/test split
│   ├── train_models.py           # Ridge, RandomForest, Keras dense NN
│   ├── evaluate.py                # RMSE / MAE / R² per horizon
│   └── register_model.py         # picks best model, registers in Model Registry
├── webapp/
│   ├── app.py                    # Streamlit dashboard
│   ├── model_utils.py            # loads best model regardless of type
│   ├── aqi_utils.py               # EPA category lookup + hazard threshold
│   └── explain.py                 # SHAP explanation for the 24h forecast
├── notebooks/
│   └── eda.ipynb                  # exploratory data analysis
├── .github/workflows/
│   ├── feature_pipeline.yml       # hourly
│   └── training_pipeline.yml      # daily
└── data/                          # raw_data.csv, features_targets.csv (generated, not committed content-wise beyond raw history)
```

---

## 3. Feature Pipeline

### 3.1 Data sources

- **OpenWeather Air Pollution API** and **Current Weather API** — live PM2.5,
  PM10, NO2, SO2, CO, O3, NH3, plus temperature, humidity, wind speed, and
  pressure for Karachi (24.8607° N, 67.0011° E). Free tier covers this at
  hourly granularity.
- **AQICN** — kept as an optional secondary live reading. It was originally
  planned as the primary AQI ground truth, but AQICN's free API has no
  historical endpoint for arbitrary date ranges, which would have made
  backfill impossible. It remains in the schema as `aqi_aqicn`, nullable.
- **Open-Meteo Historical Weather API** — free, no API key required, used
  only during backfill to reconstruct past temperature/humidity/wind/pressure,
  since OpenWeather's historical weather sits behind a paid subscription tier.

### 3.2 AQI ground truth: computed, not fetched

Rather than depending on a third-party AQI number, `epa_aqi.py` computes the
official US EPA Air Quality Index directly from PM2.5 and PM10 concentrations,
using the current (May 2024 revision) breakpoint tables, taking the maximum of
the two pollutant sub-indices per the EPA's dominant-pollutant convention.
This was a deliberate design change from the original plan (see §7) and has
two benefits: it's defined identically for live and historical data, and it
doesn't depend on any single AQI provider's uptime or rate limits.

The calculator was verified against every official breakpoint boundary
(50, 100, 150, 200, 300, 500) with exact matches.

### 3.3 Engineered features

`build_features.py` computes, per hourly row:

- **Time features**: hour, day, month, day-of-week, is-weekend
- **Derived features**: AQI change rate (first difference), 1h/3h/24h AQI
  lags, 3h/24h rolling mean, 24h rolling std, PM2.5:PM10 ratio
- **Targets**: AQI 24h, 48h, and 72h ahead, produced by shifting the AQI
  series backward — rows in the most recent ~72h window are dropped since
  their future targets aren't known yet

### 3.4 Historical backfill

`backfill_historical.py` fetches OpenWeather's pollution history (chunked in
30-day windows) and Open-Meteo's weather history, merges them on timestamp,
computes the EPA AQI, and upserts into `raw_data.csv` (deduplicated on
timestamp). This is what makes the training pipeline viable on day one rather
than waiting weeks for the hourly pipeline to accumulate enough history.

---

## 4. Training Pipeline

### 4.1 Why a chronological split

This is a time-series forecasting problem, and the engineered features
include lags and rolling statistics computed from neighboring rows. A random
train/test split (scikit-learn's default) would let future information leak
into training and produce a misleadingly good test score. `load_features.py`
instead sorts by timestamp and holds out the most recent 20% of rows as the
test set; `train_models.py` uses `TimeSeriesSplit` (not K-fold) for
hyperparameter search for the same reason.

### 4.2 Models compared

| Model | Category | Notes |
|---|---|---|
| Ridge Regression | Statistical | Grid-searched over `alpha`; linear baseline |
| Random Forest | Classical ML | Grid-searched over depth, estimators, leaf size |
| Dense neural network (Keras) | Deep learning | 2 hidden layers with dropout; early stopping on validation loss |

All three natively support multi-output regression, so a single model
predicts all three horizons (24h/48h/72h) simultaneously rather than needing
three separate models.

**Design note on the deep learning model**: an LSTM was considered, since the
brief mentions deep learning explicitly, but was deliberately not used. The
engineered features already encode temporal structure through the lag and
rolling-statistic columns, so reshaping the data into raw
`(samples, timesteps, features)` sequences would be redundant for what is
fundamentally a tabular regression problem at this feature-engineering stage.
A dense network on the same engineered features is the more direct fit.

### 4.3 Evaluation

`evaluate.py` reports RMSE, MAE, and R² per horizon, plus a mean-RMSE used to
rank the three candidates. `register_model.py` trains all three, evaluates
them, and registers whichever has the lowest mean RMSE in the Hopsworks Model
Registry, tagged with its metrics for traceability.

### 4.4 On reported performance numbers

This report documents methodology, not final metrics — no run against live,
fully-backfilled Karachi data had completed at the time of writing. Every
script was validated with realistic synthetic data during development (see
§6) to confirm correctness, but actual RMSE/MAE/R² values, the winning model,
and category distributions should be captured from a live run and inserted
here before final submission. The EDA notebook's summary section is
deliberately left as a template for the same reason.

---

## 5. Automation (CI/CD)

Two GitHub Actions workflows:

- **`feature_pipeline.yml`** — runs hourly (`cron: "0 * * * *"`). Fetches new
  data, rebuilds features, pushes to Hopsworks, and commits the updated
  `raw_data.csv` back to the repository. The commit-back step exists because
  GitHub-hosted runners are stateless between runs, while `build_features.py`
  needs accumulated raw history to compute lag and rolling features — the
  repository itself is the persistence layer for raw data, and Hopsworks is
  the persistence layer for engineered features and models.
- **`training_pipeline.yml`** — runs daily at 03:00 UTC, reading directly from
  the Hopsworks Feature View, so it has no repository state dependency.

Both support manual triggering via `workflow_dispatch` for testing before
relying on the schedule.

---

## 6. Web Application

The Streamlit dashboard (`webapp/app.py`):

1. Loads the best-scoring model from the Hopsworks Model Registry via
   `model_utils.py`, which dispatches correctly regardless of whether the
   winner was Ridge, RandomForest, or the Keras network.
2. Loads the most recent feature row from the Feature Store and predicts all
   three horizons.
3. Displays current AQI plus the three forecasts as color-coded EPA category
   badges (Good → Hazardous).
4. Shows a hazard alert banner when any horizon crosses the
   Unhealthy-for-Sensitive-Groups threshold (AQI ≥ 150).
5. Plots a trend chart combining the last 72 hours of actual AQI with the
   three forecast points.
6. Explains the 24h prediction with a SHAP bar chart (`explain.py`), using
   `TreeExplainer`/`LinearExplainer` for RandomForest/Ridge and falling back
   to `KernelExplainer` for the neural network.

---

## 7. Engineering decisions and issues caught during development

Documented here for transparency, since a few choices diverged from the
original plan or were corrected after testing surfaced real bugs:

1. **AQICN dropped as the primary AQI target.** Its free API only returns the
   current reading with no historical endpoint, which would have made
   backfill impossible. Switched to a computed EPA AQI from PM2.5/PM10
   instead (§3.2).
2. **`push_to_feature_store.py`'s null-handling bug.** An early version used
   a blanket `dropna()` before pushing to Hopsworks. Since `aqi_aqicn` is
   null by design on every row, this silently dropped 100% of the data.
   Caught by testing with realistic synthetic data before it reached a real
   Hopsworks account; fixed by scoping the null-check to required columns
   only, leaving optional fields exempt.
3. **Empty-dataframe guard added to the same script**, since the first ~72
   hourly pipeline runs (before backfill or before 72h of accumulated
   history) will have no rows with known targets — the script now logs and
   exits cleanly instead of attempting to push an empty table.
4. Every script in this project was tested against synthetic data with
   realistic shapes (matching the real API/Hopsworks schemas) and, where a
   live connection wasn't reachable from the development sandbox, against
   mocked Hopsworks calls that verified the correct methods, parameters, and
   data shapes were being used — including confirming the model registry
   selects the `sklearn` vs `tensorflow` namespace correctly depending on
   which model won training.

---

## 8. Exploratory Data Analysis

`notebooks/eda.ipynb` covers, once run against real backfilled data:

- Data completeness and missing-value check
- AQI trend over time with a 24h rolling average
- Seasonality: AQI by hour of day, day of week, and month (boxplots)
- Correlation matrix across AQI, pollutants, and weather variables
- AQI category distribution using the same EPA breakpoints the dashboard uses
- Hour-to-hour volatility and variability by forecast horizon, which
  contextualizes why 72h RMSE is expected to exceed 24h RMSE

The notebook was executed end-to-end against synthetic data with deliberately
engineered daily/weekly/monthly patterns to confirm every cell runs cleanly;
all patterns were correctly recovered in the resulting plots. Its final
summary section is a template for findings specific to real Karachi data
(seasonal burning, rush-hour effects, dominant pollutant) rather than
pre-filled conclusions.

---

## 9. Setup and running instructions

### Accounts and keys needed
- OpenWeather API key (free): [openweathermap.org/api](https://openweathermap.org/api)
- AQICN token (free, optional): [aqicn.org/data-platform/token](https://aqicn.org/data-platform/token)
- Hopsworks account + API key (free tier): [hopsworks.ai](https://www.hopsworks.ai)

### Local setup
```bash
pip install -r feature_pipeline/requirements.txt
pip install -r training_pipeline/requirements.txt
pip install -r webapp/requirements.txt

export OPENWEATHER_API_KEY=...
export HOPSWORKS_API_KEY=...
```

### First-time run order
```bash
# 1. Backfill ~90 days of history
python feature_pipeline/backfill_historical.py --days 90

# 2. Build engineered features
python feature_pipeline/build_features.py

# 3. Push to Hopsworks
python feature_pipeline/push_to_feature_store.py

# 4. Train and register the best model
python training_pipeline/register_model.py

# 5. Run the dashboard
streamlit run webapp/app.py
```

### Automation
Push the repository to GitHub, add the required secrets under
**Settings → Secrets and variables → Actions**
(`OPENWEATHER_API_KEY`, `AQICN_API_KEY`, `HOPSWORKS_API_KEY`,
`HOPSWORKS_PROJECT`), enable "Read and write permissions" for Actions under
**Settings → Actions → General**, and trigger both workflows once manually to
confirm they run clean before relying on the schedule.

---

## 10. Limitations and possible extensions

- **Single-city scope.** Everything is hardcoded to Karachi's coordinates;
  extending to multiple cities would mean parameterizing the pipeline by
  location and either multiple feature groups or a location feature.
- **Two pollutants drive the AQI calculation.** `epa_aqi.py` only computes
  sub-indices for PM2.5 and PM10. Extending to NO2/SO2/CO/O3 sub-indices
  would give a more complete "dominant pollutant" picture, particularly
  useful if Karachi's non-particulate pollution ever becomes the binding
  constraint.
- **No true sequence model.** As discussed in §4.2, an LSTM/Transformer over
  raw time windows wasn't used. If the engineered features prove
  insufficient once evaluated against real data, this is the natural next
  experiment.
- **Repository-based raw data persistence** (§5) is simple but not
  infinitely scalable — `raw_data.csv` will grow indefinitely with hourly
  commits. A rolling-window trim (e.g. keep the trailing 6 months) would keep
  this manageable long-term.
