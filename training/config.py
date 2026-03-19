"""
All hyperparameters and training settings in one place.
Edit this file to tune the model without touching training code.
"""

CONFIG = {
    # --- Data ---
    "seq_len": 60,           # input window: 60 trading days (~3 months)
    "horizon": 5,            # predict 5 trading days (1 week) ahead
    "val_start": "2021-01-01",
    "test_start": "2023-01-01",
    "data_start": "2005-01-01",

    # --- Model ---
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": False,  # set True for slightly better accuracy, ~2x slower

    # --- Training ---
    "batch_size": 512,       # large batch is fine for LSTM on GPU
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 7,           # early stopping patience (epochs)
    "amp": True,             # automatic mixed precision (fp16) — saves VRAM

    # --- Paths ---
    "model_save_path": "models/model.pt",
    "scaler_save_path": "models/scaler.pkl",
}
