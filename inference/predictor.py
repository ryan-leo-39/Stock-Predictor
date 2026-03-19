"""
Inference engine — loads the trained model and generates predictions
for one or more tickers without any training dependencies.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch

from data.download import download_ticker
from data.features import compute_features, N_FEATURES
from model.lstm import StockLSTM, build_model
from training.config import CONFIG

_MODEL_PATH = CONFIG["model_save_path"]
_SCALER_PATH = CONFIG["scaler_save_path"]


class Predictor:
    """
    Loads model.pt and scaler.pkl once, then generates predictions on demand.
    Designed to be instantiated once and reused across Streamlit reruns.
    """

    def __init__(self, model_path: str = _MODEL_PATH, scaler_path: str = _SCALER_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = CONFIG["seq_len"]

        # Load scaler
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler not found at {scaler_path}. "
                "Copy models/ directory from training machine."
            )
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Copy models/ directory from training machine."
            )
        self.model = build_model(CONFIG).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict_ticker(self, ticker: str) -> dict:
        """
        Download latest data for a ticker and return a prediction dict.

        Returns:
            {
                "ticker": str,
                "probability": float,   # P(price up in 5 days)
                "signal": str,          # "BUY", "AVOID", or "NEUTRAL"
                "confidence": float,    # distance from 0.5, scaled to [0, 1]
                "price_history": pd.DataFrame,  # OHLCV for charting
            }
        """
        ohlcv = download_ticker(ticker, use_cache=True)
        if ohlcv is None or len(ohlcv) < self.seq_len + 60:
            return {"ticker": ticker, "error": "Insufficient data"}

        features = compute_features(ohlcv)
        if len(features) < self.seq_len:
            return {"ticker": ticker, "error": "Insufficient feature data"}

        # Take the most recent window
        window = features.values[-self.seq_len:]  # (seq_len, n_features)

        # Scale
        window_scaled = self.scaler.transform(window).astype(np.float32)

        x = torch.from_numpy(window_scaled).unsqueeze(0).to(self.device)  # (1, seq_len, n_feat)

        with torch.no_grad():
            prob = self.model(x).item()

        signal = self._get_signal(prob)
        confidence = abs(prob - 0.5) * 2  # [0, 1] — how far from neutral

        return {
            "ticker": ticker,
            "probability": prob,
            "signal": signal,
            "confidence": confidence,
            "price_history": ohlcv,
        }

    def scan_tickers(self, tickers: list[str], top_n: int = 10) -> dict:
        """
        Run predictions for a list of tickers and return ranked buy/avoid lists.

        Returns:
            {
                "buy":   list of result dicts, sorted by probability desc
                "avoid": list of result dicts, sorted by probability asc
                "all":   list of all result dicts
            }
        """
        results = []
        for ticker in tickers:
            result = self.predict_ticker(ticker)
            if "error" not in result:
                results.append(result)

        buy_signals = sorted(
            [r for r in results if r["signal"] == "BUY"],
            key=lambda r: r["probability"],
            reverse=True,
        )[:top_n]

        avoid_signals = sorted(
            [r for r in results if r["signal"] == "AVOID"],
            key=lambda r: r["probability"],
        )[:top_n]

        return {"buy": buy_signals, "avoid": avoid_signals, "all": results}

    @staticmethod
    def _get_signal(prob: float) -> str:
        if prob >= 0.60:
            return "BUY"
        elif prob <= 0.40:
            return "AVOID"
        return "NEUTRAL"
