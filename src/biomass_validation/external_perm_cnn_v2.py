from __future__ import annotations

import argparse
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
CKPT_DIR = Path("outputs/checkpoints")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_gray(path: Path, size=(128, 128)) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L").resize(size), dtype=np.float32) / 255.0
    return arr


class PermDataset(Dataset):
    def __init__(self, paths, y_log10k_n, train: bool, size=(128, 128), seed=0):
        self.paths = np.asarray(paths)
        self.y = np.asarray(y_log10k_n, dtype=np.float32)
        self.train = train
        self.size = size
        self.rng = np.random.default_rng(seed)

    def __len__(self): return len(self.paths)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if self.rng.random() < 0.5:
            x = x[:, ::-1].copy()
        if self.rng.random() < 0.5:
            x = x[::-1, :].copy()
        if self.rng.random() < 0.7:
            a = float(self.rng.uniform(0.8, 1.2))   # contrast
            b = float(self.rng.uniform(-0.05, 0.05)) # brightness
            x = np.clip(a * x + b, 0.0, 1.0)
        if self.rng.random() < 0.5:
            n = self.rng.normal(0.0, 0.02, size=x.shape).astype(np.float32)
            x = np.clip(x + n, 0.0, 1.0)
        return x

    def __getitem__(self, idx):
        p = Path(self.paths[idx])
        x = load_gray(p, self.size)
        if self.train:
            x = self._augment(x)
        x = torch.from_numpy(x).unsqueeze(0)  # [1,H,W]
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


class BetterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.feat = nn.Sequential(
            block(1, 32),
            block(32, 64),
            block(64, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.feat(x)
        x = self.head(x)
        return x.squeeze(1)


def mean_abs_pct_err(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_pred - y_true) / y_true)) * 100.0)


@torch.no_grad()
def eval_loader(model, dl, mu, sig):
    model.eval()
    preds = []
    ys = []
    for xb, yb in dl:
        xb = xb.to(DEVICE)
        pred = model(xb).cpu().numpy()
        preds.append(pred)
        ys.append(yb.numpy())
    y_pred_n = np.concatenate(preds)
    y_true_n = np.concatenate(ys)

    y_pred = y_pred_n * sig + mu
    y_true = y_true_n * sig + mu

    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))

    k_true = 10 ** y_true
    k_pred = 10 ** y_pred
    mape = mean_abs_pct_err(k_true, k_pred)
    return r2, mae, mape


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--epochs", type=int, default=40)
    return ap.parse_args()


def main(seed: int, epochs: int):
    seed_everything(seed)

    df = pd.read_csv(LABELS)
    df["img_path"] = df["filename"].apply(lambda s: (ROOT / s).as_posix())
    if not df["img_path"].apply(lambda p: Path(p).exists()).all():
        bad = df[~df["img_path"].apply(lambda p: Path(p).exists())].head(5)
        raise SystemExit(f"Missing image files. Examples: {bad.to_dict(orient='records')}")

    paths = df["img_path"].to_numpy()
    y = np.log10(df["k_m2"].to_numpy(dtype=float)).astype(np.float32)

    # Split seeded
    X_train, X_test, y_train, y_test = train_test_split(paths, y, test_size=0.2, random_state=seed)
    X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

    mu = float(y_train.mean())
    sig = float(y_train.std() + 1e-8)

    y_train_n = (y_train - mu) / sig
    y_val_n   = (y_val   - mu) / sig
    y_test_n  = (y_test  - mu) / sig

    # Dataset augmentation seeded (train only)
    train_ds = PermDataset(X_train, y_train_n, train=True,  seed=seed)
    val_ds   = PermDataset(X_val,   y_val_n,   train=False, seed=seed)
    test_ds  = PermDataset(X_test,  y_test_n,  train=False, seed=seed)

    # Deterministic-ish shuffle
    g = torch.Generator()
    g.manual_seed(seed)

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0, generator=g)
    val_dl   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=0)

    model = BetterCNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    best_r2 = -1e9
    best_state = None
    patience = 8
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in train_dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()) * xb.size(0)

        train_loss = tot / len(train_ds)
        val_r2, val_mae, val_mape = eval_loader(model, val_dl, mu, sig)
        scheduler.step(val_r2)

        print(f"[seed={seed}] epoch {ep:02d}/{epochs} train_loss={train_loss:.5f}  val_r2={val_r2:.4f}  val_mae={val_mae:.4f}  val_mape={val_mape:.2f}%")

        if val_r2 > best_r2 + 1e-4:
            best_r2 = val_r2
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[seed={seed}] Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    ckpt_path = CKPT_DIR / f"external_perm_cnn_v2_seed{seed}.pt"
    torch.save(model.state_dict(), ckpt_path)

    test_r2, test_mae, test_mape = eval_loader(model, test_dl, mu, sig)

    summary = {
        "seed": int(seed),
        "device": DEVICE,
        "n_total": int(len(df)),
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds)),
        "n_test": int(len(test_ds)),
        "epochs_ran": int(ep),
        "best_val_r2": float(best_r2),
        "test_r2_log10k": float(test_r2),
        "test_mae_log10k": float(test_mae),
        "test_mape_k_percent": float(test_mape),
        "checkpoint": str(ckpt_path),
        "notes": "Predicts log10(k_m2) from grayscale image; augmentation + early stopping; per-seed split+shuffle.",
    }

    out_path = OUT_DIR / f"external_perm_cnn_v2_seed{seed}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print("\nRESULT:", json.dumps(summary, indent=2))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    args = parse_args()
    main(seed=args.seed, epochs=args.epochs)
