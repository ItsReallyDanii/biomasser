from __future__ import annotations

import json
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path("data/external/2D-porous-media-images")
LABELS = Path("data/raw/external_perm_labels.csv")
OUT_DIR = Path("outputs/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: scipy improves distance transform; fallback exists
try:
    from scipy.ndimage import distance_transform_edt, label as cc_label
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def load_img(p: Path, size=(128, 128)) -> np.ndarray:
    arr = np.asarray(Image.open(p).convert("L").resize(size), dtype=np.float32) / 255.0
    return arr


def binarize_pores(arr: np.ndarray) -> np.ndarray:
    # Pores assumed bright in this dataset; threshold at 0.5
    return (arr > 0.5)


def bfs_shortest_path_len(pores: np.ndarray) -> int | None:
    """Shortest path length from any left-edge pore to any right-edge pore (4-neighbor)."""
    H, W = pores.shape
    dist = -np.ones((H, W), dtype=np.int32)
    q = deque()

    # seed from left boundary pores
    for y in range(H):
        if pores[y, 0]:
            dist[y, 0] = 0
            q.append((y, 0))

    # BFS
    while q:
        y, x = q.popleft()
        if x == W - 1:
            return int(dist[y, x])
        d = dist[y, x] + 1
        if y > 0 and pores[y - 1, x] and dist[y - 1, x] < 0:
            dist[y - 1, x] = d; q.append((y - 1, x))
        if y < H - 1 and pores[y + 1, x] and dist[y + 1, x] < 0:
            dist[y + 1, x] = d; q.append((y + 1, x))
        if x > 0 and pores[y, x - 1] and dist[y, x - 1] < 0:
            dist[y, x - 1] = d; q.append((y, x - 1))
        if x < W - 1 and pores[y, x + 1] and dist[y, x + 1] < 0:
            dist[y, x + 1] = d; q.append((y, x + 1))
    return None


def connected_component_stats(pores: np.ndarray) -> tuple[float, float, float]:
    """Returns (largest_comp_frac, num_comps_norm, percolates_lr)."""
    H, W = pores.shape
    if HAVE_SCIPY:
        lab, n = cc_label(pores.astype(np.uint8))
        if n == 0:
            return 0.0, 0.0, 0.0
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        largest = float(counts.max()) / float(H * W)
        num_norm = float(n) / float(H * W) * 1e4  # scaled for numeric stability
        left_ids = set(lab[:, 0][lab[:, 0] > 0].tolist())
        right_ids = set(lab[:, -1][lab[:, -1] > 0].tolist())
        percolates = 1.0 if (left_ids & right_ids) else 0.0
        return largest, num_norm, percolates

    # Fallback: rough estimate without full CC labeling
    # Use BFS percolation + treat components count as 0/1 proxy
    percolates = 1.0 if bfs_shortest_path_len(pores) is not None else 0.0
    largest = float(pores.mean())  # crude fallback
    num_norm = 0.0
    return largest, num_norm, percolates


def distance_stats(pores: np.ndarray) -> tuple[float, float]:
    """Returns (mean_dt, p90_dt) of pore-space distance transform."""
    if HAVE_SCIPY:
        # distance in pore space to nearest solid (larger => wider channels)
        dt = distance_transform_edt(pores)
        mean_dt = float(dt[pores].mean()) if pores.any() else 0.0
        p90_dt = float(np.percentile(dt[pores], 90)) if pores.any() else 0.0
        return mean_dt, p90_dt
    return 0.0, 0.0


def features_from_img(arr: np.ndarray) -> dict:
    pores = binarize_pores(arr)
    H, W = pores.shape
    porosity = float(pores.mean())

    # tortuosity via shortest path length
    sp = bfs_shortest_path_len(pores)
    percolates = 1.0 if sp is not None else 0.0
    if sp is None:
        tort = float("inf")
        sp_norm = 0.0
    else:
        sp_norm = float(sp) / float(W)  # normalized path length
        tort = sp_norm  # proxy tortuosity (>=1-ish when path exists)

    largest_frac, num_comps_norm, percolates_lr = connected_component_stats(pores)
    mean_dt, p90_dt = distance_stats(pores)

    # finite tortuosity feature
    tort_f = 10.0 if not np.isfinite(tort) else float(tort)

    return {
        "porosity": porosity,
        "percolates": percolates,
        "largest_comp_frac": largest_frac,
        "num_comps_norm": num_comps_norm,
        "shortest_path_norm": sp_norm,
        "tortuosity_proxy": tort_f,
        "mean_dt": mean_dt,
        "p90_dt": p90_dt,
        "have_scipy": float(HAVE_SCIPY),
        "percolates_lr": percolates_lr,
    }


def mean_abs_pct_err(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_pred - y_true) / y_true)) * 100.0)


def main():
    df = pd.read_csv(LABELS)
    df["img_path"] = df["filename"].apply(lambda s: (ROOT / s).as_posix())
    if not df["img_path"].apply(lambda p: Path(p).exists()).all():
        bad = df[~df["img_path"].apply(lambda p: Path(p).exists())].head(5)
        raise SystemExit(f"Missing image files. Examples: {bad.to_dict(orient='records')}")

    feats = []
    for i, row in df.iterrows():
        arr = load_img(Path(row["img_path"]))
        f = features_from_img(arr)
        f["filename"] = row["filename"]
        f["k_m2"] = float(row["k_m2"])
        feats.append(f)
        if (i + 1) % 200 == 0:
            print(f"featurized {i+1}/{len(df)}")

    D = pd.DataFrame(feats)

    y = np.log10(D["k_m2"].to_numpy())
    feature_cols = [
        "porosity","percolates","largest_comp_frac","num_comps_norm",
        "shortest_path_norm","tortuosity_proxy","mean_dt","p90_dt","percolates_lr"
    ]
    X = D[feature_cols].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=6,
        learning_rate=0.08,
        max_iter=500,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    mae_log10 = float(mean_absolute_error(y_test, y_pred))

    k_true = 10 ** y_test
    k_pred = 10 ** y_pred
    mape = mean_abs_pct_err(k_true, k_pred)

    summary = {
        "n_total": int(len(D)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": feature_cols,
        "have_scipy": bool(HAVE_SCIPY),
        "model": "HistGradientBoostingRegressor",
        "r2_log10k": r2,
        "mae_log10k": mae_log10,
        "mape_k_percent": mape,
    }

    out_path = OUT_DIR / "external_perm_benchmark_v2_features.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print("\nRESULT:", json.dumps(summary, indent=2))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
