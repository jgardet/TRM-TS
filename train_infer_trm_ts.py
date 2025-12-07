#!/usr/bin/env python
"""
Example training and inference script for trm_ts.py (standalone).

- Loads a CSV with OHLCV (optional). If not provided, generates synthetic data.
- Builds sliding windows of length --window for sequence modeling.
- Trains TRM-TS with Gaussian NLL (mean + logvar) for next-step return.
- Runs inference on the test split and prints basic metrics.

Usage:
    # Install deps
    #   pip install -r requirements.txt

    # Train with synthetic data
    #   python train_infer_trm_ts.py --epochs 20 --device cpu

    # Train with CSV (expects at least 'close' column; others optional: open,high,low,volume)
    #   python train_infer_trm_ts.py --csv candles.csv --epochs 20 --device cpu

CSV expectations:
    - Must contain a 'close' column.
    - Features are built from available numeric columns (z-score normalized per column).

Model artifacts:
    - Saves trained weights to trm_ts_example.pt (unless --no-save)
    - Can load with --load and skip training
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Local import
from trm_ts import trm_ts


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WindowedDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, window: int):
        assert x.ndim == 2, "x shape must be (N, F)"
        assert y.ndim == 2, "y shape must be (N, H)"
        self.x = x
        self.y = y
        self.window = window

    def __len__(self) -> int:
        return max(self.x.shape[0] - self.window + 1, 0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        xb = self.x[idx : idx + self.window, :]        # (T, F)
        yb = self.y[idx + self.window - 1, :]          # (H,)
        return xb, yb


def zscore(a: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = np.nanmean(a)
    s = np.nanstd(a)
    s = s if s > eps else eps
    return (a - m) / s


def make_features_from_csv(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if "close" not in df.columns:
        raise ValueError("CSV must contain a 'close' column")
    work = df.copy()

    # Select numeric columns as raw features
    num = work.select_dtypes(include=[np.number]).copy()
    if num.shape[1] == 0:
        num["close"] = work["close"].astype(float)

    # Simple engineered features
    num["ret_1"] = np.log(num["close"]).diff()
    num["ret_5"] = np.log(num["close"]).diff(5)
    num["hl_spread"] = (num.get("high", num["close"]) - num.get("low", num["close"])) / num["close"].replace(0, np.nan)
    num["vol_norm"] = zscore(num.get("volume", pd.Series(np.zeros(len(num)))))

    # Z-score normalize each column independently
    feats = []
    for col in num.columns:
        arr = num[col].astype(float).to_numpy()
        feats.append(zscore(arr))
    X = np.vstack(feats).T  # (N, F)

    # Label: next-step log return of close
    y = np.log(num["close"]).diff().shift(-1).to_numpy()  # (N,)
    y = np.nan_to_num(y, nan=0.0)
    Y = y[:, None]  # horizon = 1

    # Drop initial NaNs introduced by diffs (already zero-filled) - leave as is
    return X.astype(np.float32), Y.astype(np.float32)


def make_synthetic(n: int = 10000, f: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    t = np.arange(n)
    # Price as sum of trends + cycles + noise
    price = (
        0.001 * t
        + 0.5 * np.sin(2 * np.pi * t / 50)
        + 0.25 * np.sin(2 * np.pi * t / 200)
        + 0.1 * np.random.randn(n)
        + 100
    )
    close = price
    openp = close + 0.05 * np.random.randn(n)
    high = np.maximum(openp, close) + 0.1 * np.abs(np.random.randn(n))
    low = np.minimum(openp, close) - 0.1 * np.abs(np.random.randn(n))
    volume = 1000 + 50 * np.sin(2 * np.pi * t / 30) + 100 * np.random.randn(n)

    raw = np.stack([openp, high, low, close, volume], axis=1)

    # Additional random features
    extra = np.random.randn(n, max(0, f - raw.shape[1])) * 0.1
    X = np.concatenate([raw, extra], axis=1)

    # Normalize columns (z-score)
    X = np.apply_along_axis(zscore, 0, X)

    # Label: next-step log return of close
    ret = np.diff(np.log(close), prepend=np.log(close[0]))
    y = np.roll(ret, -1)
    y[-1] = 0.0
    Y = y[:, None].astype(np.float32)

    return X.astype(np.float32), Y


def prepare_data(csv: Optional[str], window: int, test_ratio: float = 0.1):
    if csv:
        df = pd.read_csv(csv)
        X, Y = make_features_from_csv(df)
    else:
        X, Y = make_synthetic(n=12000, f=20)

    n = X.shape[0]
    split = int(n * (1 - test_ratio))

    x_train, x_test = X[:split], X[split:]
    y_train, y_test = Y[:split], Y[split:]

    # Convert to torch
    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_test_t = torch.from_numpy(x_test)
    y_test_t = torch.from_numpy(y_test)

    train_ds = WindowedDataset(x_train_t, y_train_t, window)
    test_ds = WindowedDataset(x_test_t, y_test_t, window)
    return train_ds, test_ds, X.shape[1], Y.shape[1]


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    var = torch.exp(logvar)
    return 0.5 * (logvar + (target - mean) ** 2 / var).mean()


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total = 0.0
    nb = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model({"x": xb})
            loss = gaussian_nll(out["mean"], out["logvar"], yb)
            total += float(loss.item())
            nb += 1
    model.train()
    return total / max(nb, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=None, help="Path to CSV with OHLCV (optional)")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--steps", type=int, default=4, help="recursive steps")
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--save", action="store_true")
    p.add_argument("--no-save", dest="save", action="store_false")
    p.set_defaults(save=True)
    p.add_argument("--load", type=str, default=None, help="Path to load a saved .pt model and skip training")
    args = p.parse_args()

    set_seed(42)

    train_ds, test_ds, n_features, horizon = prepare_data(args.csv, args.window)

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    model = trm_ts(
        n_features=n_features,
        latent_dim=args.latent_dim,
        recursive_steps=args.steps,
        forecast_horizon=horizon,
        multi_scales=(16, 64),
        n_regimes=8,
        ctx_dim=0,
        attn_heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    if args.load and Path(args.load).exists():
        state = torch.load(args.load, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded model from {args.load}")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, drop_last=False)

        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best = math.inf
        patience = 15
        bad = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            ep_loss = 0.0
            nb = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model({"x": xb})
                loss = gaussian_nll(out["mean"], out["logvar"], yb)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optim.step()
                ep_loss += float(loss.item())
                nb += 1

            train_loss = ep_loss / max(nb, 1)
            test_loss = evaluate(model, test_loader, device)
            tag = "★ (best)" if test_loss < best - 0.005 else ""
            print(f"Epoch {epoch:03d} - Train: {train_loss:.4f}, Test: {test_loss:.4f} {tag}")
            if test_loss < best - 0.005:
                best = test_loss
                bad = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    print(f"Early stopping at epoch {epoch} - best test loss: {best:.4f}")
                    model.load_state_dict(best_state)
                    break

        if args.save:
            torch.save(model.state_dict(), "trm_ts_example.pt")
            print("Saved model to trm_ts_example.pt")

    # Inference: sliding evaluation over test set (last batch)
    model.eval()
    with torch.no_grad():
        loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)
        preds = []
        targets = []
        for xb, yb in loader:
            xb = xb.to(device)
            out = model({"x": xb})
            preds.append(out["mean"].cpu())
            targets.append(yb)
        P = torch.cat(preds, dim=0).squeeze(-1)
        T = torch.cat(targets, dim=0).squeeze(-1)
        mae = torch.mean(torch.abs(P - T)).item()
        rmse = torch.sqrt(torch.mean((P - T) ** 2)).item()
        print(f"Inference metrics on test split - MAE: {mae:.6f}, RMSE: {rmse:.6f}")


if __name__ == "__main__":
    main()
