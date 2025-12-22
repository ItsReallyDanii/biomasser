from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path("data/external/2D-porous-media-images")
LABELS = Path("data/raw/external_perm_labels.csv")
OUT_DIR = Path("outputs/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PermDataset(Dataset):
    def __init__(self, paths, y_log10k, size=(128, 128)):
        self.paths = paths
        self.y = y_log10k.astype(np.float32)
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = Path(self.paths[idx])
        arr = np.asarray(Image.open(p).convert("L").resize(self.size), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
        y = torch.tensor(self.y[idx])
        return x, y


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.net(x)
        return self.head(x).squeeze(1)


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

    y = np.log10(df["k_m2"].to_numpy(dtype=float)).astype(np.float32)
    paths = df["img_path"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(paths, y, test_size=0.2, random_state=42)

    # normalize targets for easier training
    mu = float(y_train.mean())
    sig = float(y_train.std() + 1e-8)
    y_train_n = (y_train - mu) / sig
    y_test_n = (y_test - mu) / sig

    train_ds = PermDataset(X_train, y_train_n)
    test_ds = PermDataset(X_test, y_test_n)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    model = TinyCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # small, bounded training
    EPOCHS = 5
    for ep in range(1, EPOCHS + 1):
        model.train()
        tot = 0.0
        for xb, yb in train_dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * xb.size(0)
        print(f"epoch {ep}/{EPOCHS} train_mse={tot/len(train_ds):.6f}")

    # eval
    model.eval()
    preds = []
    ys = []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(DEVICE)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            ys.append(yb.numpy())

    y_pred_n = np.concatenate(preds)
    y_true_n = np.concatenate(ys)

    # denormalize to log10k
    y_pred = y_pred_n * sig + mu
    y_true = y_true_n * sig + mu

    r2 = float(r2_score(y_true, y_pred))
    mae_log10 = float(mean_absolute_error(y_true, y_pred))

    k_true = 10 ** y_true
    k_pred = 10 ** y_pred
    mape = mean_abs_pct_err(k_true, k_pred)

    summary = {
        "device": DEVICE,
        "n_total": int(len(df)),
        "n_train": int(len(train_ds)),
        "n_test": int(len(test_ds)),
        "epochs": EPOCHS,
        "r2_log10k": r2,
        "mae_log10k": mae_log10,
        "mape_k_percent": mape,
    }

    out_path = OUT_DIR / "external_perm_cnn.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print("\nRESULT:", json.dumps(summary, indent=2))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
