from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

from src.simulate_flow import simulate_pressure_field, compute_conductivity


ROOT = Path("data/external/2D-porous-media-images")
LABELS = Path("data/raw/external_perm_labels.csv")
OUT_DIR = Path("outputs/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_img(p: Path, size=(128, 128)) -> np.ndarray:
    arr = np.asarray(Image.open(p).convert("L").resize(size), dtype=np.float32) / 255.0
    return arr


def features_from_img(arr: np.ndarray, laplace_iters: int = 400) -> dict:
    # Assume pores are bright. For simulate_flow(), void is defined as < 0.5.
    # So invert so pores become dark (0) -> counted as open void.
    structure = 1.0 - arr

    porosity = float((arr > 0.5).mean())

    gx, gy = np.gradient(arr)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    edge_density = float((grad_mag > 0.2).mean())
    anisotropy = float(np.mean(np.abs(gx)) / (np.mean(np.abs(gy)) + 1e-8))

    p_field, mask = simulate_pressure_field(structure, inlet=1.0, outlet=0.0, iterations=laplace_iters)
    cond = float(compute_conductivity(p_field, mask))  # higher ~ more permeable (proxy)

    return {
        "porosity": porosity,
        "edge_density": edge_density,
        "anisotropy": anisotropy,
        "laplace_cond": cond,
    }


def main():
    df = pd.read_csv(LABELS)
    if "filename" not in df.columns or "k_m2" not in df.columns:
        raise SystemExit(f"Expected columns filename,k_m2. Got: {list(df.columns)}")

    # Build absolute paths
    df["img_path"] = df["filename"].apply(lambda s: (ROOT / s).as_posix())
    missing = df[~df["img_path"].apply(lambda p: Path(p).exists())]
    if len(missing):
        raise SystemExit(f"Missing {len(missing)} image files. Example: {missing.iloc[0].to_dict()}")

    # Compute features
    feats = []
    for i, row in df.iterrows():
        arr = load_img(Path(row["img_path"]))
        f = features_from_img(arr, laplace_iters=400)
        f["filename"] = row["filename"]
        f["k_m2"] = float(row["k_m2"])
        feats.append(f)

        if (i + 1) % 100 == 0:
            print(f"featurized {i+1}/{len(df)}")

    D = pd.DataFrame(feats)
    # Target in log space (permeability spans orders of magnitude)
    y = np.log10(D["k_m2"].to_numpy())
    X = D[["porosity", "edge_density", "anisotropy", "laplace_cond"]].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, D[["filename", "k_m2"]], test_size=0.2, random_state=42
    )

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    mae_log10 = float(mean_absolute_error(y_test, y_pred))

    # Convert back to linear permeability for a % error sense
    k_true = 10 ** y_test
    k_pred = 10 ** y_pred
    mape = float(np.mean(np.abs((k_pred - k_true) / k_true)) * 100.0)

    summary = {
        "n_total": int(len(D)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": ["porosity", "edge_density", "anisotropy", "laplace_cond"],
        "ridge_alpha": 1.0,
        "r2_log10k": r2,
        "mae_log10k": mae_log10,
        "mape_k_percent": mape,
        "coef": {name: float(c) for name, c in zip(["porosity","edge_density","anisotropy","laplace_cond"], model.coef_)},
        "intercept": float(model.intercept_),
    }

    out_path = OUT_DIR / "external_perm_benchmark.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print("\nRESULT:", json.dumps(summary, indent=2))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
