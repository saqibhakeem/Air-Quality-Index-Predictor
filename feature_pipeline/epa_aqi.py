"""
epa_aqi.py

Computes the US EPA Air Quality Index from pollutant concentrations,
using the official (May 2024 revision) breakpoint tables. Shared by
fetch_raw_data.py (live) and backfill_historical.py (historical), so
the target variable is defined identically in both.

Reference: EPA "Final Updates to the Air Quality Index (AQI) for
Particulate Matter" fact sheet, Feb 2024 / effective May 6, 2024.
"""

from typing import Optional

# (conc_low, conc_high, aqi_low, aqi_high) breakpoints, in µg/m³
PM25_BREAKPOINTS = [
    (0.0, 9.0, 0, 50),
    (9.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 325.4, 301, 500),
]

# PM10 breakpoints unchanged by the 2024 revision (24-hr average, µg/m³)
PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]


def _linear_aqi(conc: float, breakpoints: list) -> Optional[float]:
    """Apply the standard EPA piecewise-linear AQI formula."""
    if conc is None:
        return None

    for c_lo, c_hi, aqi_lo, aqi_hi in breakpoints:
        if c_lo <= conc <= c_hi:
            return round((aqi_hi - aqi_lo) / (c_hi - c_lo) * (conc - c_lo) + aqi_lo, 1)

    # Above the top breakpoint - clamp to the max category (500) rather than
    # erroring out, since real-world spikes (e.g. smog events) can exceed it.
    top_c_hi = breakpoints[-1][1]
    if conc > top_c_hi:
        return 500.0
    return None  # negative/invalid concentration


def calculate_aqi_pm25(pm25_ugm3: float) -> Optional[float]:
    return _linear_aqi(pm25_ugm3, PM25_BREAKPOINTS)


def calculate_aqi_pm10(pm10_ugm3: float) -> Optional[float]:
    return _linear_aqi(pm10_ugm3, PM10_BREAKPOINTS)


def compute_overall_aqi(pm25_ugm3: float, pm10_ugm3: float) -> Optional[float]:
    """Overall AQI = max of the sub-indices (the EPA's 'dominant pollutant' rule).

    Only PM2.5 and PM10 are used here since they are the two pollutants that
    drive AQI in Karachi's context and both are available hourly, historically,
    from OpenWeather. NO2/SO2/CO/O3 sub-indices could be added the same way
    if you want a fuller implementation later.
    """
    candidates = [
        v for v in (calculate_aqi_pm25(pm25_ugm3), calculate_aqi_pm10(pm10_ugm3)) if v is not None
    ]
    return max(candidates) if candidates else None