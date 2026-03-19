"""
Download and cache historical OHLCV data for S&P 500 stocks using yfinance.
Data is cached locally as CSV files to avoid re-downloading on every run.
"""

import os
import time
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# Cache directory relative to project root
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")


def get_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 constituent tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df["Symbol"].tolist()
    # Wikipedia uses dots (BRK.B), yfinance uses dashes (BRK-B)
    tickers = [t.replace(".", "-") for t in tickers]
    return sorted(tickers)


def get_tech_tickers() -> list[str]:
    """Return S&P 500 tickers in the Information Technology sector."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tech = df[df["GICS Sector"] == "Information Technology"]["Symbol"].tolist()
    return [t.replace(".", "-") for t in sorted(tech)]


def download_ticker(
    ticker: str,
    start: str = "2005-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """
    Download OHLCV data for a single ticker.
    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    Returns None if download fails or data is too short.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{ticker}.csv")

    if use_cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df if len(df) >= 100 else None

    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty or len(df) < 100:
            return None
        # Keep only OHLCV columns
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"
        df.to_csv(cache_path)
        return df
    except Exception as e:
        print(f"  Warning: failed to download {ticker}: {e}")
        return None


def download_all(
    tickers: list[str],
    start: str = "2005-01-01",
    delay: float = 0.2,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Download data for a list of tickers with progress bar.
    Returns a dict mapping ticker -> DataFrame.
    """
    data = {}
    for ticker in tqdm(tickers, desc="Downloading"):
        df = download_ticker(ticker, start=start, use_cache=use_cache)
        if df is not None:
            data[ticker] = df
        time.sleep(delay)  # be polite to yfinance servers
    print(f"\nSuccessfully loaded {len(data)}/{len(tickers)} tickers.")
    return data


def refresh_cache(tickers: list[str]) -> None:
    """Force re-download of all tickers, bypassing cache."""
    download_all(tickers, use_cache=False)


if __name__ == "__main__":
    print("Fetching S&P 500 ticker list...")
    tickers = get_sp500_tickers()
    print(f"Found {len(tickers)} tickers. Downloading...")
    download_all(tickers)
