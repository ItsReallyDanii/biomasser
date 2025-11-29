import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ks_2samp
import os

INPUT_PATH = "results/flow_metrics/flow_metrics.csv"
REPORT_PATH = "results/physics_validation_report.csv"

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"⚠️ Metrics file not found at {INPUT_PATH}")

print(f"📂 Loading: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)

real = df[df["Type"].str.lower() == "real"]
synthetic = df[df["Type"].str.lower() == "synthetic"]

metrics = ["Mean_K", "Mean_dP/dy", "FlowRate", "Porosity", "Anisotropy"]
report = {}

for metric in metrics:
    t_stat, t_p = ttest_ind(real[metric], synthetic[metric], equal_var=False, nan_policy='omit')
    ks_stat, ks_p = ks_2samp(real[metric], synthetic[metric])
    report[metric] = {
        "real_mean": np.nanmean(real[metric]),
        "synthetic_mean": np.nanmean(synthetic[metric]),
        "t_p_value": round(t_p, 5),
        "ks_p_value": round(ks_p, 5),
    }

report_df = pd.DataFrame(report).T
report_df.to_csv(REPORT_PATH)
print(f"\n🧾 Physics validation report saved → {REPORT_PATH}")

# Numerical Summary
similar_t = sum(v["t_p_value"] > 0.05 for v in report.values())
similar_ks = sum(v["ks_p_value"] > 0.05 for v in report.values())

print(f"\n🌿 Flow Physics Validation Summary")
print("-----------------------------------")
print(f"✅ Metrics analyzed: {len(metrics)}")
print(f"✅ Metrics statistically similar (T-test): {similar_t}/{len(metrics)}")
print(f"✅ Metrics distribution-similar (KS-test): {similar_ks}/{len(metrics)}\n")

print(f"📊 Detailed Metric Comparison:\n")
print(report_df)
