"""
Builds notebooks/eda.ipynb programmatically via nbformat.
Run once to (re)generate the notebook file: python build_notebook.py
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

# --- Title ---------------------------------------------------------------
cells.append(md(
"""# Karachi AQI - Exploratory Data Analysis

This notebook explores the engineered feature table (`data/features_targets.csv`)
produced by the feature pipeline, to understand:

1. Overall data quality and coverage
2. How AQI trends over time
3. Daily / weekly / monthly seasonality
4. Which pollutants and weather variables correlate most with AQI
5. The distribution of AQI categories (Good / Moderate / ... / Hazardous)

Run this after `backfill_historical.py` + `build_features.py` have produced
a reasonably large `features_targets.csv` (at least a few weeks of hourly data
gives much more meaningful seasonality plots than a few days)."""
))

# --- Setup -----------------------------------------------------------------
cells.append(code(
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (12, 4)
sns.set_style("whitegrid")

DATA_PATH = "../data/features_targets.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp_utc"])
df = df.sort_values("timestamp_utc").reset_index(drop=True)
print(f"Loaded {len(df)} rows spanning {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")
df.head()"""
))

# --- Data overview -----------------------------------------------------------
cells.append(md("## 1. Data overview & quality check"))

cells.append(code(
"""df.info()"""
))

cells.append(code(
"""print("Missing values per column:")
missing = df.isna().sum()
missing[missing > 0]"""
))

cells.append(code(
"""df[["aqi_us_epa", "pm2_5", "pm10", "no2", "so2", "co", "o3",
    "temp_c", "humidity_pct", "wind_speed_ms"]].describe()"""
))

# --- Time series trend -------------------------------------------------------
cells.append(md(
"""## 2. AQI trend over time

The raw hourly series, plus a 24h rolling average to smooth out hour-to-hour
noise and make the underlying trend easier to read."""
))

cells.append(code(
"""fig, ax = plt.subplots()
ax.plot(df["timestamp_utc"], df["aqi_us_epa"], alpha=0.3, label="Hourly AQI")
ax.plot(df["timestamp_utc"], df["aqi_us_epa"].rolling(24, min_periods=1).mean(),
        color="crimson", label="24h rolling mean")
ax.set_ylabel("AQI (EPA scale)")
ax.set_title("Karachi AQI over time")
ax.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()"""
))

# --- Seasonality --------------------------------------------------------------
cells.append(md(
"""## 3. Seasonality patterns

Does AQI follow a daily rhythm (e.g. worse during traffic hours), a weekly
one (weekday vs weekend), or a monthly one (seasonal burning, weather
patterns)? Boxplots make outliers and spread visible, not just the average."""
))

cells.append(code(
"""fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.boxplot(data=df, x="hour", y="aqi_us_epa", ax=axes[0])
axes[0].set_title("AQI by hour of day")

sns.boxplot(data=df, x="day_of_week", y="aqi_us_epa", ax=axes[1])
axes[1].set_title("AQI by day of week (0=Mon)")

sns.boxplot(data=df, x="month", y="aqi_us_epa", ax=axes[2])
axes[2].set_title("AQI by month")

plt.tight_layout()
plt.show()"""
))

cells.append(code(
"""weekend_comparison = df.groupby("is_weekend")["aqi_us_epa"].agg(["mean", "median", "std"])
weekend_comparison.index = weekend_comparison.index.map({0: "Weekday", 1: "Weekend"})
weekend_comparison"""
))

# --- Correlations ---------------------------------------------------------------
cells.append(md(
"""## 4. What correlates with AQI?

A correlation heatmap across pollutants, weather variables, and AQI itself.
Note that PM2.5 and PM10 feed directly into how `aqi_us_epa` is computed, so
their strong correlation is expected - the more informative signal here is
which *weather* variables (temp, humidity, wind, pressure) move AQI, since
those aren't circular."""
))

cells.append(code(
"""corr_cols = ["aqi_us_epa", "pm2_5", "pm10", "no2", "so2", "co", "o3",
             "temp_c", "humidity_pct", "wind_speed_ms", "pressure_hpa"]
corr = df[corr_cols].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Correlation matrix: AQI, pollutants, weather")
plt.tight_layout()
plt.show()"""
))

cells.append(code(
"""corr["aqi_us_epa"].drop("aqi_us_epa").sort_values(key=abs, ascending=False)"""
))

# --- Distribution of AQI categories -------------------------------------------
cells.append(md(
"""## 5. AQI category distribution

Using the same EPA breakpoints the web app uses, so this matches what
users will actually see on the dashboard."""
))

cells.append(code(
"""def categorize(aqi):
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

category_order = ["Good", "Moderate", "Unhealthy for Sensitive Groups",
                   "Unhealthy", "Very Unhealthy", "Hazardous"]

df["aqi_category"] = df["aqi_us_epa"].apply(categorize)
category_counts = df["aqi_category"].value_counts().reindex(category_order).fillna(0)

ax = category_counts.plot(kind="bar", color="steelblue")
ax.set_ylabel("Hours")
ax.set_title("Hours spent in each AQI category")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

print(f"\\n{(df['aqi_category'].isin(['Unhealthy for Sensitive Groups','Unhealthy','Very Unhealthy','Hazardous'])).mean()*100:.1f}% "
      "of hours were Unhealthy-for-Sensitive-Groups or worse")"""
))

# --- Change rate / volatility --------------------------------------------------
cells.append(md(
"""## 6. How fast does AQI change hour to hour?

This matters for the forecast horizons: if AQI is highly volatile,
72h-ahead predictions are inherently harder than 24h-ahead ones."""
))

cells.append(code(
"""fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(df["aqi_change_rate"].dropna(), bins=40, color="darkorange")
axes[0].set_title("Distribution of hour-to-hour AQI change")
axes[0].set_xlabel("AQI change rate")

horizon_std = {
    "+24h": df["target_aqi_24h"].std() if "target_aqi_24h" in df else np.nan,
    "+48h": df["target_aqi_48h"].std() if "target_aqi_48h" in df else np.nan,
    "+72h": df["target_aqi_72h"].std() if "target_aqi_72h" in df else np.nan,
}
axes[1].bar(horizon_std.keys(), horizon_std.values(), color="teal")
axes[1].set_title("AQI variability by forecast horizon")
axes[1].set_ylabel("Std. deviation")

plt.tight_layout()
plt.show()"""
))

# --- Summary --------------------------------------------------------------------
cells.append(md(
"""## 7. Summary

Fill this in after running against real backfilled data - a few things worth
checking specifically for Karachi:

- **Seasonality**: does AQI spike in specific months (e.g. crop-burning season,
  winter inversion) or hours (rush hour traffic)?
- **Weekday vs weekend**: is there a measurable traffic-driven dip on weekends?
- **Dominant pollutant**: does PM2.5 or PM10 correlate more strongly with AQI -
  this affects which pollutant sources matter most for Karachi specifically.
- **Volatility by horizon**: if 72h variability is much higher than 24h, expect
  (and report) higher RMSE at longer horizons in the training pipeline results -
  that's a property of the data, not a modeling failure."""
))

nb["cells"] = cells

with open("../notebooks/eda.ipynb", "w") as f:
    nbf.write(nb, f)

print("Wrote notebooks/eda.ipynb")
