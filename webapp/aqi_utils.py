"""
aqi_utils.py

EPA AQI category lookup - used for color-coded badges and hazard alerts
in the dashboard. Thresholds match the same EPA scale used to compute
aqi_us_epa in the feature pipeline (epa_aqi.py), so labels line up exactly
with the values being predicted.
"""

# (max_aqi_inclusive, label, hex_color, short_guidance)
AQI_CATEGORIES = [
    (50, "Good", "#00A651", "Air quality is satisfactory."),
    (100, "Moderate", "#FFD700", "Acceptable; sensitive groups should be cautious."),
    (150, "Unhealthy for Sensitive Groups", "#FF8C00", "Sensitive groups should limit prolonged outdoor exertion."),
    (200, "Unhealthy", "#FF0000", "Everyone may experience health effects."),
    (300, "Very Unhealthy", "#8F3F97", "Health alert: everyone may experience serious effects."),
    (500, "Hazardous", "#7E0023", "Health warning of emergency conditions."),
]

HAZARD_THRESHOLD = 150  # "Unhealthy for Sensitive Groups" and above triggers an alert


def get_aqi_category(aqi: float) -> tuple:
    """Returns (label, hex_color, guidance) for a given AQI value."""
    if aqi is None:
        return ("Unknown", "#808080", "No data available.")

    for max_aqi, label, color, guidance in AQI_CATEGORIES:
        if aqi <= max_aqi:
            return (label, color, guidance)
    return AQI_CATEGORIES[-1][1:]  # clamp anything above 500 to Hazardous


def is_hazardous(aqi: float) -> bool:
    return aqi is not None and aqi >= HAZARD_THRESHOLD
