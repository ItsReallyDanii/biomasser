import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.biomass_validation.data_ingest import ingest_hydraulic_curve
from src.biomass_validation.sim_adapter import run_sim_once

def fit_scale(train_meas: pd.DataFrame, train_sim: pd.DataFrame) -> float:
    # scale minimizes || s*dp_proxy - dp_meas || in least squares
    x = train_sim["dp_proxy"].to_numpy()
    y = train_meas["dp_pa"].to_numpy()
    return float(np.dot(x, y) / np.dot(x, x))

def mean_abs_pct_err(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_pred - y_true) / y_true)) * 100.0)

def main():
    cfg_path = Path("configs/validation_config.yaml")
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text())

    meas_path = Path("data/raw/hydraulic_curve.csv")
    df_meas = ingest_hydraulic_curve(meas_path).sort_values("q_m3_s").reset_index(drop=True)

    # holdout: last point = test
    df_train = df_meas.iloc[:-1].copy()
    df_test = df_meas.iloc[-1:].copy()

    out_dir = Path("outputs") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # run sim to get dp_proxy for all q
    sim_rows = []
    for q in df_meas["q_m3_s"].to_list():
        r = run_sim_once(config=cfg, q_m3_s=float(q), out_dir=out_dir)
        sim_rows.append({"q_m3_s": r.q_m3_s, "dp_proxy": r.dp_proxy})
    df_sim = pd.DataFrame(sim_rows)

    # fit scale on TRAIN only
    df_sim_train = df_sim.merge(df_train[["q_m3_s", "dp_pa"]], on="q_m3_s", how="inner")
    scale = fit_scale(df_sim_train, df_sim_train.rename(columns={"dp_pa":"dp_pa"}))
    cfg["pa_per_proxy_unit"] = scale

    # predict on TEST
    df_sim["dp_pa_pred"] = df_sim["dp_proxy"] * scale
    df_test_pred = df_test.merge(df_sim[["q_m3_s","dp_pa_pred"]], on="q_m3_s", how="inner")

    err_test = mean_abs_pct_err(df_test_pred["dp_pa"], df_test_pred["dp_pa_pred"])

    summary = {
        "scale_pa_per_proxy_unit_fit_on_train": scale,
        "test_point": df_test_pred.to_dict(orient="records")[0],
        "test_mean_abs_pct_err": err_test,
    }
    (out_dir / "holdout_summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)

if __name__ == "__main__":
    main()
