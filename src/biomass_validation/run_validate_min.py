from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.biomass_validation.data_ingest import ingest_hydraulic_curve
from src.biomass_validation.sim_adapter import run_sim_once
from src.biomass_validation.compare import dp_error_percent

def fit_scale_least_squares(dp_meas: np.ndarray, dp_proxy: np.ndarray) -> float:
    """
    Fit s minimizing || dp_meas - s * dp_proxy ||^2
    s = (dp_meas·dp_proxy) / (dp_proxy·dp_proxy)
    """
    num = float((dp_meas * dp_proxy).sum())
    den = float((dp_proxy * dp_proxy).sum())
    if den <= 0:
        raise ValueError("Cannot fit scale: dp_proxy has zero energy.")
    return num / den

def main() -> None:
    meas_path = Path("data/raw/hydraulic_curve.csv")
    df_meas = ingest_hydraulic_curve(meas_path)

    config = {
        "project": "biomasser",
        "seed": 1337,
        "sim": {"fluid_viscosity_pa_s": 0.001, "density_kg_m3": 1000},
        "geometry": {"sample_id": "baseline"},
        "outputs": {"out_dir": "outputs"},
    }

    out_dir = Path("outputs") / "sim_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_rows = []
    for q in df_meas["q_m3_s"].tolist():
        r = run_sim_once(config=config, q_m3_s=float(q), out_dir=out_dir)
        sim_rows.append({"q_m3_s": r.q_m3_s, "dp_proxy": r.dp_pa})

    df_sim = pd.DataFrame(sim_rows)

    # Fit scale factor: Pa per proxy-unit
    merged = df_meas.merge(df_sim, on="q_m3_s", how="inner")
    scale = fit_scale_least_squares(
        dp_meas=merged["dp_pa"].to_numpy(dtype=float),
        dp_proxy=merged["dp_proxy"].to_numpy(dtype=float),
    )

    # Apply calibration
    df_sim_cal = df_sim.copy()
    df_sim_cal["dp_pa"] = df_sim_cal["dp_proxy"] * scale

    print("MEASURED:")
    print(df_meas)
    print("\nSIMULATED (proxy):")
    print(df_sim)
    print(f"\nFitted scale (Pa per proxy-unit): {scale:.6e}")
    print("\nSIMULATED (calibrated to Pa):")
    print(df_sim_cal[["q_m3_s", "dp_pa"]])

    err = dp_error_percent(df_meas, df_sim_cal[["q_m3_s", "dp_pa"]])
    print(f"\nMean abs % error (dp) AFTER calibration: {err:.2f}%")

    summary = {
        "scale_pa_per_proxy_unit": scale,
        "mean_abs_percent_error_dp_calibrated": err,
    }
    (Path("outputs") / "reports").mkdir(parents=True, exist_ok=True)
    Path("outputs/reports/validation_min.json").write_text(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
