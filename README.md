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

### 4.4 A real finding: seasonal regime shift, and how it was resolved

Training against the full backfilled history (spanning March–August, ~3,800
hourly rows) produced a striking result: **every model — Ridge, RandomForest,
and the dense NN — performed worse than a naive persistence baseline**
("assume AQI 24h/48h/72h from now equals AQI right now"), across every
horizon:

| Model | Mean RMSE (full history) |
|---|---|
| Persistence baseline | 6.580 |
| Ridge | 8.790 |
| RandomForest (initial hyperparameters) | 18.526 |
| Dense NN | 9.761 |

RandomForest's especially large error, and the fact that it roughly *doubled*
once more historical data was added rather than improving, pointed to
overfitting rather than a fundamentally unlearnable problem. Diagnosis
proceeded in two steps:

1. **Checked test-set variance directly.** The test window (the most recent
   20% chronologically) had AQI varying only ~10-11% relative to its mean —
   a real feature of that specific period (a stable, likely monsoon-driven
   stretch), not a bug. This explains why R² looked poor even for
   numerically small errors: R² penalizes error relative to a target's
   *actual* variance, and a low-variance test window is an unforgiving
   denominator.
2. **Tightened RandomForest's hyperparameter search** (removed unconstrained
   `max_depth=None`, raised the `min_samples_leaf` floor) and **restricted
   training to a 45-day rolling window** rather than the full multi-season
   history. The hypothesis: models trained across a spring-to-summer
   transition can't generalize to a single, more homogeneous recent regime
   without more than one year of data to learn true seasonality from.

Re-evaluated on the same (45-day-restricted) test split:

| Model | Mean RMSE (45-day window) | Beats baseline (7.258)? |
|---|---|---|
| Persistence baseline | 7.258 | — |
| **Ridge** | **5.753** | **Yes, at every horizon** |
| RandomForest (tightened) | 6.173 | Yes, at every horizon |
| Dense NN | 12.599 | No — 45 days is too little data for a neural net |

Ridge was selected as the registered model: lowest mean RMSE, the only model
with positive R² at any horizon (0.218 at 24h), and consistently ahead of the
baseline. The dense NN's regression here is itself informative — it
illustrates a real data-volume/model-complexity tradeoff rather than a flaw
in the approach, and is worth revisiting once enough history accumulates to
give it a fair chance.

`register_model.py --max-history-days 45` is now the standard invocation
(including in the daily GitHub Actions workflow) rather than the
unconstrained default. `check_baseline.py` and `check_variance.py` are
included in the repository as the diagnostic tools used to reach this
conclusion, and are worth re-running periodically — the right window length
is a function of how much seasonal variety exists in the data at any given
time, not a constant.



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
5. **`weather_main` schema poisoning (feature group v1 → v2).** Open-Meteo's
   historical backfill provides no weather description, so `weather_main` is
   `None` for every backfilled row. The first real push to Hopsworks
   contained only backfilled data, so Hopsworks inferred the column's type
   as `double` from an all-null column instead of `string` - which then
   broke every subsequent push once live fetches contributed real string
   values. Fixed by `.fillna("unknown")` before every push, guaranteeing a
   consistent type regardless of batch composition; required a fresh feature
   group version since the bad schema was already committed.
6. **Silent int/float schema drift (v2 → v3).** A related issue: several
   columns (`humidity_pct`, `aqi_openweather`, `aqi_us_epa`) happened to
   contain only whole numbers in an early batch, so Hopsworks inferred
   `bigint` instead of `double`. This included a genuine bug in
   `epa_aqi.py`: Python's `round(x)` with no second argument returns an
   `int`, not a `float`, so the computed AQI silently changed type depending
   on the input. Fixed at the source (`round(x, 1)` always returns float)
   and defensively (every numeric column is now forced to `float64`
   immediately before every push, regardless of what a given batch happens
   to contain) - a second fresh feature group version was needed.
7. **Kafka topic authorization on the online feature store.** Hopsworks
   writes to both an offline (Delta) and online (Kafka-backed) store when a
   feature group is `online_enabled=True`. The online write path
   consistently failed with `TOPIC_AUTHORIZATION_FAILED` even with a
   correctly-scoped API key. Since nothing in this project's design actually
   reads from the online store's low-latency API (both the training pipeline
   and the web app read via `fg.select_all().read()`, an offline batch
   read), the fix was to pass `storage="offline"` to every `insert()` call -
   removing the failure entirely with no functional loss.
8. **Native Windows can't write to Hopsworks' offline store.** `hsfs`'s
   Python engine writes Delta tables directly via a Rust-based `hdfs-native`
   client, which requires the `libgssapi_krb5` library for its Kerberos
   handshake - a library with no Windows build (Windows uses SSPI, a
   different authentication stack entirely). Reads work fine natively on
   Windows (they go through Arrow Flight, not this path), but every write
   operation (`push_to_feature_store.py`, `register_model.py`) needs to run
   from a real Linux environment - WSL2, in this project's case.
9. **Row-count-based lag/rolling features silently break on any data gap.**
   `build_features.py` originally computed `aqi_lag_24h`, rolling means, etc.
   by counting rows (`.shift(24)`, `.rolling(24)`), which is only correct if
   every row is exactly one real hour apart. A single missed hourly fetch
   (API hiccup, a workflow that didn't fire) silently misaligns every later
   lookup - tested directly by deleting one row from a synthetic series and
   confirming a downstream lag feature was off by multiple AQI points despite
   no error being raised anywhere. Fixed by reindexing to a strict hourly
   `DatetimeIndex` before any lag/rolling computation, making gaps explicit
   (and short ones interpolable) instead of invisible.
10. **A seasonal regime shift, diagnosed and resolved rather than hidden.**
    See §4.4 for the full account: training on the complete multi-season
    history caused every model to underperform a naive persistence baseline;
    restricting to a 45-day rolling window and tightening RandomForest's
    hyperparameters resolved it. This is arguably the most substantive
    engineering finding in the project and is documented in detail rather
    than only reporting the final, working configuration.
11. **A stale version reference in the web app.** `webapp/app.py` still
    pointed at feature group v1 (the original, broken/empty one from #5-6's
    predecessor) after `push_to_feature_store.py` had already moved on to
    v2 and then v3, causing an unhelpful `'NoneType' object has no attribute
    'select_all'` error. A reminder that version constants duplicated across
    files (rather than centralized in one place) need to be updated in
    lockstep - worth refactoring into a shared config module if this project
    is extended further.

---

## 8. Exploratory Data Analysis

`notebooks/eda.ipynb` was run against the real, live Karachi dataset: 3,996
hourly rows spanning 2026-03-05 to 2026-08-29. Key findings:

- **Overall AQI**: mean 75.4 (EPA "Moderate"), std 26.8, ranging from 16.9 up
  to 500 (the Hazardous ceiling was reached at least once in the observed
  period). 10.4% of all hours were Unhealthy-for-Sensitive-Groups (AQI ≥ 151)
  or worse - a meaningful real-world rate for the hazard-alert feature.
- **Dominant pollutant**: PM2.5 correlates more strongly with AQI (r=0.928)
  than PM10 (r=0.881). Among weather variables, wind speed has the strongest
  relationship (r=-0.215 - more wind, lower AQI, consistent with pollutant
  dispersion); temperature and pressure are negligible (|r| < 0.03).
- **Seasonality - the key finding**: June shows by far the widest spread and
  most extreme high outliers (several days reaching 400-500), while July and
  August are comparatively calm. This is a direct visual confirmation of the
  seasonal regime shift diagnosed independently during model training (§4.4)
  - it's the same underlying phenomenon that caused every model to
  underperform a naive baseline when trained on the full March-August
  history, and why restricting training to a recent 45-day window resolved
  it.
- **Weekday vs weekend**: weekends show a higher mean (78.4 vs 74.2) and much
  higher std (39.2 vs 20.0) than weekdays, but this is driven by a handful of
  extreme Sunday outliers rather than a systematic effect. No clear
  rush-hour cycle appears in the hour-of-day breakdown.
- **Hour-to-hour change** is usually small (tightly centered on 0) with rare
  large jumps - consistent with the outlier events being sudden spikes
  rather than gradual drift.
- **Data quality issue found**: `weather_main` is 100% missing across the
  entire dataset, including live-fetched rows, not just backfilled ones -
  meaning `fetch_raw_data.py`'s live weather-description capture isn't
  working as intended. This doesn't affect the AQI forecast (the field was
  never used as a model feature) but is worth investigating if a textual
  weather condition is ever wanted for the dashboard.

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