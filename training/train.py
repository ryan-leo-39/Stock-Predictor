"""
Training script — run this on the GPU machine.

Usage:
    python -m training.train

Saves:
    models/model.pt    — trained model weights
    models/scaler.pkl  — fitted feature scaler (needed at inference time)
"""

import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.download import get_sp500_tickers, download_all
from data.dataset import StockSequenceDataset, time_split
from data.features import scale_features, N_FEATURES
from model.lstm import build_model, count_parameters
from training.config import CONFIG
from training.evaluate import evaluate


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("\nFetching S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print(f"Downloading data for {len(tickers)} tickers...")
    raw_data = download_all(tickers, start=CONFIG["data_start"])

    train_data, val_data, test_data = time_split(
        raw_data,
        val_start=CONFIG["val_start"],
        test_start=CONFIG["test_start"],
    )

    # ── 2. Build datasets (no scaler yet — need train data first) ─────────────
    print("\nBuilding training dataset...")
    train_ds = StockSequenceDataset(
        train_data, seq_len=CONFIG["seq_len"], horizon=CONFIG["horizon"]
    )
    print(f"Train samples: {len(train_ds):,}")

    # Fit scaler on training windows
    _, scaled_train, _, _ = scale_features(train_ds.windows)
    # Re-build with scaler applied inline (scaler fitted above)
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    n, s, f = train_ds.windows.shape
    scaler.fit(train_ds.windows.reshape(-1, f))
    train_ds.windows = scaler.transform(
        train_ds.windows.reshape(-1, f)
    ).reshape(n, s, f).astype(np.float32)

    print("Building validation dataset...")
    val_ds = StockSequenceDataset(
        val_data, seq_len=CONFIG["seq_len"], horizon=CONFIG["horizon"], scaler=scaler
    )
    print(f"Val samples: {len(val_ds):,}")

    print("Building test dataset...")
    test_ds = StockSequenceDataset(
        test_data, seq_len=CONFIG["seq_len"], horizon=CONFIG["horizon"], scaler=scaler
    )
    print(f"Test samples: {len(test_ds):,}")

    # ── 3. DataLoaders ────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=2, pin_memory=True
    )

    # ── 4. Model ──────────────────────────────────────────────────────────────
    model = build_model(CONFIG).to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")

    # Class-weighted loss to handle imbalance
    class_weights = train_ds.class_weights()
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Use plain BCE since model outputs sigmoid
    criterion = nn.BCELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )
    amp_scaler = GradScaler(enabled=CONFIG["amp"])

    # ── 5. Training loop ──────────────────────────────────────────────────────
    best_val_acc = 0.0
    epochs_no_improve = 0
    os.makedirs("models", exist_ok=True)

    print("\nStarting training...\n")
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with autocast(enabled=CONFIG["amp"]):
                pred = model(x)
                loss = criterion(pred, y)

            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()

            total_loss += loss.item() * len(y)
            correct += ((pred >= 0.5) == y.bool()).sum().item()
            total += len(y)

        train_loss = total_loss / total
        train_acc = correct / total

        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["accuracy"])

        print(
            f"Epoch {epoch:3d}/{CONFIG['epochs']} | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} acc: {val_metrics['accuracy']:.4f} "
            f"prec: {val_metrics['precision']:.4f} rec: {val_metrics['recall']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"  ✓ Saved best model (val acc: {best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CONFIG["patience"]:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # ── 6. Test evaluation ────────────────────────────────────────────────────
    model.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\nTest results: {test_metrics}")

    # ── 7. Save scaler ────────────────────────────────────────────────────────
    with open(CONFIG["scaler_save_path"], "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nScaler saved to {CONFIG['scaler_save_path']}")
    print("Done. Copy models/ directory to inference machine.")


if __name__ == "__main__":
    train()
