"""
LSTM model for binary stock direction classification.

Architecture:
  - Stacked LSTM layers with dropout
  - Final linear layer → sigmoid output
  - Output: probability that price will be higher in `horizon` days
"""

import torch
import torch.nn as nn

from data.features import N_FEATURES


class StockLSTM(nn.Module):
    """
    Stacked bidirectional LSTM classifier.

    Input:  (batch, seq_len, n_features)
    Output: (batch,) — probability in [0, 1]
    """

    def __init__(
        self,
        input_size: int = N_FEATURES,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Additional dropout before head
        self.dropout = nn.Dropout(dropout)

        # Classification head
        lstm_out_size = hidden_size * self.num_directions
        self.head = nn.Sequential(
            nn.Linear(lstm_out_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            prob: (batch,) — probability of upward move
        """
        # lstm_out: (batch, seq_len, hidden * directions)
        lstm_out, _ = self.lstm(x)

        # Take the last timestep's output
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)

        prob = self.head(last_out).squeeze(-1)
        return prob


def build_model(config: dict) -> StockLSTM:
    """Build model from a config dict (from training/config.py)."""
    return StockLSTM(
        input_size=N_FEATURES,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        bidirectional=config.get("bidirectional", False),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
