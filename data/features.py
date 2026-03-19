"""
Feature engineering pipeline.
Takes raw OHLCV DataFrame and returns a normalized feature matrix
ready for the PyTorch Dataset.
"""

import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import RobustScaler


# Features the model will use — order matters (must stay consistent train/inference)
FEATURE_COLUMNS = [
    "returns",          # daily log return
    "high_low_pct",     # (high - low) / close  — intraday range
    "close_open_pct",   # (close - open) / open — intraday direction
    "volume_ratio",     # volume / 20-day avg volume
    "rsi",              # RSI(14), normalized to [0, 1]
    "macd",             # MACD line
    "macd_signal",      # MACD signal line
    "macd_hist",        # MACD histogram
    "bb_upper_dist",    # % distance from upper Bollinger Band
    "bb_lower_dist",    # % distance from lower Bollinger Band
    "ema10_dist",       # % distance from EMA(10)
    "ema50_dist",       # % distance from EMA(50)
    "sma20_dist",       # % distance from SMA(20)
    "atr_pct",          # ATR(14) / close — normalized volatility
    "obv_norm",         # On-Balance Volume (normalized)
]

N_FEATURES = len(FEATURE_COLUMNS)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicator features from a raw OHLCV DataFrame.
    Returns a DataFrame of features aligned to the same index.
    NaN rows (due to indicator warmup) are dropped.
    """
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    open_ = df["Open"].squeeze()
    volume = df["Volume"].squeeze()

    feat = pd.DataFrame(index=df.index)

    # Price-based features
    feat["returns"] = np.log(close / close.shift(1))
    feat["high_low_pct"] = (high - low) / close
    feat["close_open_pct"] = (close - open_) / open_.replace(0, np.nan)

    # Volume
    vol_ma20 = volume.rolling(20).mean()
    feat["volume_ratio"] = volume / vol_ma20.replace(0, np.nan)

    # RSI
    feat["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi() / 100.0

    # MACD
    macd_ind = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    feat["macd"]        = macd_ind.macd()
    feat["macd_signal"] = macd_ind.macd_signal()
    feat["macd_hist"]   = macd_ind.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    feat["bb_upper_dist"] = (bb.bollinger_hband() - close) / close
    feat["bb_lower_dist"] = (close - bb.bollinger_lband()) / close

    # Moving average distances
    ema10 = ta.trend.EMAIndicator(close, window=10).ema_indicator()
    ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    sma20 = ta.trend.SMAIndicator(close, window=20).sma_indicator()
    feat["ema10_dist"] = (close - ema10) / close
    feat["ema50_dist"] = (close - ema50) / close
    feat["sma20_dist"] = (close - sma20) / close

    # ATR (normalized volatility)
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    feat["atr_pct"] = atr / close

    # OBV (normalized by rolling std to make stationary)
    obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    obv_std = obv.rolling(50).std().replace(0, np.nan)
    feat["obv_norm"] = obv / obv_std

    # Drop NaN rows (indicator warmup period)
    feat = feat.dropna()

    return feat[FEATURE_COLUMNS]


def compute_labels(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """
    Binary label: 1 if close price is higher `horizon` days in the future, else 0.
    """
    close = df["Close"].squeeze()
    future_return = close.shift(-horizon) / close - 1
    labels = (future_return > 0).astype(int)
    return labels


def scale_features(
    train_features: np.ndarray,
    val_features: np.ndarray | None = None,
    test_features: np.ndarray | None = None,
) -> tuple:
    """
    Fit a RobustScaler on training data and transform all splits.
    RobustScaler is better than StandardScaler for financial data with outliers.
    Returns (scaler, scaled_train, scaled_val, scaled_test).
    """
    n_train, seq_len, n_feat = train_features.shape
    scaler = RobustScaler()

    scaler.fit(train_features.reshape(-1, n_feat))

    def transform(arr):
        n, s, f = arr.shape
        return scaler.transform(arr.reshape(-1, f)).reshape(n, s, f)

    scaled_train = transform(train_features)
    scaled_val   = transform(val_features)  if val_features  is not None else None
    scaled_test  = transform(test_features) if test_features is not None else None

    return scaler, scaled_train, scaled_val, scaled_test
