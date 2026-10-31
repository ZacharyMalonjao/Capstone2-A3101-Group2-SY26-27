import csv
import os
import statistics
import sys
import time
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
 
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from validation import validate_inputs  # noqa: E402
 
MODEL_DIR = ROOT / "models"
THRESHOLD_SECONDS = 3.0   # <-- the numeric limit you put in the NFR
N_RUNS = 30
 
# A few different valid inputs (region, gdp, density, urban%, sanitation%, rainfall, temp)
SAMPLE_INPUTS = [
    ("AFR", 1500.0, 80.0, 45.0, 30.0, 1100.0, 26.0),
    ("EUR", 25000.0, 120.0, 75.0, 98.0, 650.0, 10.0),
    ("SEAR", 3500.0, 350.0, 38.0, 60.0, 1800.0, 27.5),
    ("WPR", 9000.0, 150.0, 60.0, 85.0, 1500.0, 24.0),
    ("AMR", 8000.0, 40.0, 80.0, 75.0, 1300.0, 22.0),
    ("EMR", 4500.0, 60.0, 55.0, 70.0, 250.0, 25.0),
]
 
 
def predict_once(model, q1, q3, row):
    region, gdp, dens, urban, sanit, rain, temp = row
    errors, _warnings = validate_inputs(region, gdp, dens, urban, sanit, rain, temp)
    if errors:
        raise ValueError(errors)
    df = pd.DataFrame([{
        "Region_Code": region,
        "Median_GDP": gdp,
        "Median_Population_Density": dens,
        "Median_Urban_pct": urban,
        "Median_Sanitation_Access_pct": sanit,
        "Median_Rainfall": rain,
        "Median_Temp": temp,
    }])
    pred = float(np.expm1(model.predict(df)[0]))
    tier = "Low" if pred < q1 else ("High" if pred > q3 else "Medium")
    return pred, tier
 
 
def main():
    # Cold cost: loading the model files from disk.
    t0 = time.perf_counter()
    model = joblib.load(os.path.join(MODEL_DIR, "model_RandomForest.pkl"))
    thresholds = joblib.load(os.path.join(MODEL_DIR, "tier_thresholds.pkl"))
    load_s = time.perf_counter() - t0
    q1, q3 = thresholds["q1"], thresholds["q3"]
 
    rows = []
    for i in range(N_RUNS):
        inp = SAMPLE_INPUTS[i % len(SAMPLE_INPUTS)]
        t = time.perf_counter()
        pred, tier = predict_once(model, q1, q3, inp)
        elapsed = time.perf_counter() - t
        rows.append((i + 1, inp[0], round(pred, 4), tier, elapsed))
 
    times = [r[4] for r in rows]
    first, rest = times[0], times[1:]
 
    print(f"Model + thresholds load (cold): {load_s:.3f} s")
    print(f"First prediction:               {first:.3f} s")
    print(f"Later predictions (n={len(rest)}):    mean {statistics.mean(rest):.3f} s, "
          f"min {min(rest):.3f} s, max {max(rest):.3f} s")
    worst = max(load_s + first, max(rest))
    print(f"Threshold: {THRESHOLD_SECONDS:.1f} s -> "
          f"{'PASS' if worst <= THRESHOLD_SECONDS else 'FAIL'} (worst case {worst:.3f} s)")
 
    out = Path(__file__).with_name("response_time_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "region", "predicted_cases", "tier", "seconds"])
        w.writerows(rows)
    print(f"Saved per-run results to {out}")
 
 
if __name__ == "__main__":
    main()
 