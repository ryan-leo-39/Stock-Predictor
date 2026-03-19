"""
PyTorch Dataset that builds sliding-window sequences from feature DataFrames.
Supports multi-stock training by concatenating windows from all tickers.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.features import compute_features, compute_labels, N_FEATURES


class StockSequenceDataset(Dataset):
    """
    Sliding-window Dataset over one or more stocks.

    Each sample is:
        X: (seq_len, n_features) float32 tensor
        y: scalar float32 label (1.0 = price up in `horizon` days, 0.0 = down)
    """

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        seq_len: int = 60,
        horizon: int = 5,
        scaler=None,
    ):
        """
        Args:
            data:     dict mapping ticker -> raw OHLCV DataFrame
            seq_len:  number of past trading days in each input window
            horizon:  number of days ahead to predict
            scaler:   fitted RobustScaler (optional; applies per-feature scaling)
        """
        self.seq_len = seq_len
        self.horizon = horizon
        self.scaler = scaler

        self.windows: list[np.ndarray] = []
        self.labels: list[float] = []

        for ticker, ohlcv in data.items():
            self._add_ticker(ticker, ohlcv)

        self.windows = np.array(self.windows, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

        if scaler is not None:
            n, s, f = self.windows.shape
            self.windows = scaler.transform(
                self.windows.reshape(-1, f)
            ).reshape(n, s, f).astype(np.float32)

    def _add_ticker(self, ticker: str, ohlcv: pd.DataFrame) -> None:
        try:
            features = compute_features(ohlcv)
            labels = compute_labels(ohlcv, self.horizon)

            # Align features and labels on the same index
            common_idx = features.index.intersection(labels.index)
            features = features.loc[common_idx]
            labels = labels.loc[common_idx]

            feat_arr = features.values
            label_arr = labels.values

            # Drop trailing rows where label would look beyond available data
            valid_end = len(feat_arr) - self.horizon

            for i in range(self.seq_len, valid_end):
                window = feat_arr[i - self.seq_len : i]
                label = label_arr[i]
                if not np.isnan(window).any() and not np.isnan(label):
                    self.windows.append(window)
                    self.labels.append(float(label))
        except Exception as e:
            print(f"  Skipping {ticker}: {e}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.windows[idx])
        y = torch.tensor(self.labels[idx])
        return x, y

    def class_weights(self) -> torch.Tensor:
        """
        Returns [weight_class0, weight_class1] for use with BCELoss weighting.
        Corrects for class imbalance (markets trend up slightly more than down).
        """
        n_pos = self.labels.sum()
        n_neg = len(self.labels) - n_pos
        total = len(self.labels)
        w0 = total / (2.0 * n_neg) if n_neg > 0 else 1.0
        w1 = total / (2.0 * n_pos) if n_pos > 0 else 1.0
        return torch.tensor([w0, w1], dtype=torch.float32)


def time_split(
    data: dict[str, pd.DataFrame],
    val_start: str = "2021-01-01",
    test_start: str = "2023-01-01",
) -> tuple[dict, dict, dict]:
    """
    Split each ticker's DataFrame by date into train/val/test.
    Uses temporal split (not random) to avoid lookahead bias.
    """
    train_data, val_data, test_data = {}, {}, {}
    for ticker, df in data.items():
        train_data[ticker] = df[df.index < val_start]
        val_data[ticker] = df[(df.index >= val_start) & (df.index < test_start)]
        test_data[ticker] = df[df.index >= test_start]
    return train_data, val_data, test_data
