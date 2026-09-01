"""
check_baseline.py

Compares the registered model's test-set performance against the simplest
possible baseline: "AQI 24h/48h/72h from now = AQI right now" (persistence).

R² can look bad in a low-variance test window even when a model is doing
something genuinely useful - this comparison answers the more practical
question: is the model beating "assume nothing changes"?

Usage:
    python check_baseline.py                        # full history
    python check_baseline.py --max-history-days 45   # match a windowed register_model.py run
"""

import argparse
import numpy as np
from load_features import get_train_test_split, TARGET_COLUMNS
from evaluate import evaluate_predictions, print_report

parser = argparse.ArgumentParser()
parser.add_argument(
    "--max-history-days", type=int, default=None,
    help="Must match whatever you passed to register_model.py for a fair, "
         "same-test-set comparison.",
)
args = parser.parse_args()

X_train, X_test, y_train, y_test = get_train_test_split(max_history_days=args.max_history_days)

# Persistence baseline: predict the CURRENT aqi_us_epa for all 3 horizons
current_aqi = X_test["aqi_us_epa"].to_numpy()
baseline_preds = np.column_stack([current_aqi, current_aqi, current_aqi])

baseline_metrics = evaluate_predictions(y_test, baseline_preds)
print_report("Persistence baseline (predict no change)", baseline_metrics)

print("\n" + "=" * 60)
print("Compare the RMSE/MAE above against your trained model's numbers.")
print("If your model's RMSE is LOWER than this baseline's, it's genuinely")
print("adding value even though R2 looks negative for both.")
print("=" * 60)