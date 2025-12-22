import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from src.biomass_validation.data_ingest import ingest_hydraulic_curve
from src.biomass_validation.sim_adapter import run_sim_once

def fit_scale(dp_meas: np.ndarray, dp_proxy: np.ndarray) -> float:
    num = float((dp_meas * dp_proxy).sum())
    den = float((dp_proxy * dp_proxy).sum())
    if den <= 0:
        raise ValueError("Cannot fit scale: dp_proxy has zero energy.")
    return num / den

def mean_abs_pct_err(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_pred - y_true) / y_true)) * 100.0)

def main():
    cfg = yaml.safe_load(Path("configs/validation_config.yaml").read_text())

    df = ingest_hydraulic_curve("data/raw/hydraulic_curve.csv").copy()
    df = df.sort_values("q_m3_s").reset_index(drop=True)

    if "filename" not in df.columns:
        raise ValueError("hydraulic_curve.csv must include a 'filename' column for per-sample validation.")

    # Holdout = last row
    df_train = df.iloc[:-1].copy()
    df_test = df.iloc[-1:].copy()

    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_rows = []
    for _, row in df.iterrows():
        f = str(row["filename"])
        structure_path = str(Path("data/real_xylem_preprocessed") / f)

        cfg_row = dict(cfg)
        cfg_row["structure_png_override"] = structure_path

        r = run_sim_once(config=cfg_row, q_m3_s=float(row["q_m3_s"]), out_dir=out_dir)
        sim_rows.append({"filename": f, "q_m3_s": r.q_m3_s, "dp_proxy": r.dp_proxy})

    df_sim = pd.DataFrame(sim_rows)

    train = df_train.merge(df_sim, on=["filename", "q_m3_s"], how="inner")
    scale = fit_scale(train["dp_pa"].to_numpy(float), train["dp_proxy"].to_numpy(float))

    df_sim["dp_pa_pred"] = df_sim["dp_proxy"] * scale
    test = df_test.merge(df_sim, on=["filename", "q_m3_s"], how="inner")

    err = mean_abs_pct_err(test["dp_pa"], test["dp_pa_pred"])

    summary = {
        "scale_fit_on_train": scale,
        "test_point": test.to_dict(orient="records")[0],
        "test_mean_abs_pct_err": err,
    }
    (out_dir / "holdout_summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)

if __name__ == "__main__":
    main()
