"""
Streamlit UI for the stock prediction application.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from data.download import get_sp500_tickers, get_tech_tickers
from inference.predictor import Predictor

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.signal-buy    { color: #00c853; font-weight: 700; font-size: 1.2rem; }
.signal-avoid  { color: #d50000; font-weight: 700; font-size: 1.2rem; }
.signal-neutral{ color: #ffd600; font-weight: 700; font-size: 1.2rem; }
.metric-card   { background: #1e1e2e; border-radius: 8px; padding: 16px; margin: 4px 0; }
</style>
""", unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_predictor():
    return Predictor()


@st.cache_data(ttl=3600, show_spinner="Fetching ticker list...")
def load_sp500_tickers():
    return get_sp500_tickers()


@st.cache_data(ttl=3600)
def load_tech_tickers():
    return get_tech_tickers()


# ── Helper: price chart ───────────────────────────────────────────────────────
def plot_price_history(ohlcv: pd.DataFrame, ticker: str, signal: str, prob: float):
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=ohlcv.index[-120:],  # last ~6 months
        open=ohlcv["Open"].iloc[-120:],
        high=ohlcv["High"].iloc[-120:],
        low=ohlcv["Low"].iloc[-120:],
        close=ohlcv["Close"].iloc[-120:],
        name=ticker,
        increasing_line_color="#00c853",
        decreasing_line_color="#d50000",
    ))

    # Annotation for prediction
    color = "#00c853" if signal == "BUY" else "#d50000" if signal == "AVOID" else "#ffd600"
    fig.add_annotation(
        x=ohlcv.index[-1],
        y=float(ohlcv["Close"].iloc[-1]),
        text=f"  {signal} ({prob:.0%})",
        showarrow=True,
        arrowhead=2,
        arrowcolor=color,
        font=dict(color=color, size=14),
        bgcolor="#1e1e2e",
        bordercolor=color,
    )

    fig.update_layout(
        template="plotly_dark",
        title=f"{ticker} — Last 6 Months",
        xaxis_rangeslider_visible=False,
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def signal_badge(signal: str, prob: float) -> str:
    cls = {"BUY": "signal-buy", "AVOID": "signal-avoid", "NEUTRAL": "signal-neutral"}[signal]
    arrow = {"BUY": "▲", "AVOID": "▼", "NEUTRAL": "●"}[signal]
    return f'<span class="{cls}">{arrow} {signal} &nbsp; {prob:.0%}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Stock Predictor")
    st.caption("5-day directional prediction using LSTM")

    universe = st.selectbox(
        "Stock universe",
        ["S&P 500", "Tech (S&P 500 IT sector)", "Custom ticker"],
    )

    st.divider()
    st.markdown("**Signal thresholds**")
    buy_threshold = st.slider("Buy signal above", 0.50, 0.95, 0.60, 0.05)
    avoid_threshold = st.slider("Avoid signal below", 0.05, 0.50, 0.40, 0.05)

    st.divider()
    top_n = st.slider("Top N signals to show", 5, 20, 10)

    model_ok = os.path.exists("models/model.pt") and os.path.exists("models/scaler.pkl")
    if not model_ok:
        st.warning(
            "⚠️ Model files not found.\n\n"
            "Run `python -m training.train` on the GPU machine "
            "and copy `models/` here."
        )


# ── Main content ──────────────────────────────────────────────────────────────
st.title("Stock Price Direction Predictor")
st.caption(
    "Model predicts the probability that a stock's closing price will be **higher in 5 trading days**. "
    "Not financial advice."
)

if not model_ok:
    st.error("Model not loaded. See sidebar for instructions.")
    st.stop()

predictor = load_predictor()
# Apply user threshold overrides
predictor._get_signal = staticmethod(
    lambda prob: "BUY" if prob >= buy_threshold else "AVOID" if prob <= avoid_threshold else "NEUTRAL"
)

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_search, tab_scan = st.tabs(["🔍 Analyze a Stock", "📊 Market Scan"])


# ─── Tab 1: Single stock analysis ────────────────────────────────────────────
with tab_search:
    col1, col2 = st.columns([3, 1])
    with col1:
        if universe == "Custom ticker":
            ticker_input = st.text_input("Enter ticker symbol", value="AAPL").upper().strip()
        else:
            tickers_list = load_sp500_tickers() if universe == "S&P 500" else load_tech_tickers()
            ticker_input = st.selectbox("Select a stock", tickers_list)
    with col2:
        st.write("")
        st.write("")
        analyze_btn = st.button("Analyze →", use_container_width=True)

    if analyze_btn and ticker_input:
        with st.spinner(f"Running prediction for {ticker_input}..."):
            result = predictor.predict_ticker(ticker_input)

        if "error" in result:
            st.error(f"Could not analyze {ticker_input}: {result['error']}")
        else:
            # Signal header
            st.markdown(
                f"### {ticker_input} &nbsp; {signal_badge(result['signal'], result['probability'])}",
                unsafe_allow_html=True,
            )

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("5-Day Up Probability", f"{result['probability']:.1%}")
            m2.metric("Signal", result["signal"])
            m3.metric("Confidence", f"{result['confidence']:.1%}")
            last_close = result["price_history"]["Close"].iloc[-1]
            m4.metric("Last Close", f"${last_close:.2f}")

            # Chart
            fig = plot_price_history(
                result["price_history"],
                ticker_input,
                result["signal"],
                result["probability"],
            )
            st.plotly_chart(fig, use_container_width=True)


# ─── Tab 2: Market scan ───────────────────────────────────────────────────────
with tab_scan:
    if universe == "Custom ticker":
        st.info("Select S&P 500 or Tech universe in the sidebar to run a market scan.")
    else:
        scan_btn = st.button(
            f"Run scan on {'S&P 500' if universe == 'S&P 500' else 'Tech stocks'} →",
            use_container_width=False,
        )

        if scan_btn:
            tickers_list = load_sp500_tickers() if universe == "S&P 500" else load_tech_tickers()
            progress = st.progress(0, text="Scanning stocks...")
            results_container = st.empty()

            buy_list, avoid_list = [], []
            for i, ticker in enumerate(tickers_list):
                result = predictor.predict_ticker(ticker)
                if "error" not in result:
                    if result["signal"] == "BUY":
                        buy_list.append(result)
                    elif result["signal"] == "AVOID":
                        avoid_list.append(result)
                progress.progress((i + 1) / len(tickers_list), text=f"Scanning {ticker}...")

            progress.empty()

            buy_list.sort(key=lambda r: r["probability"], reverse=True)
            avoid_list.sort(key=lambda r: r["probability"])

            col_buy, col_avoid = st.columns(2)

            with col_buy:
                st.subheader(f"▲ Top Buy Signals ({len(buy_list)} found)")
                if buy_list:
                    rows = []
                    for r in buy_list[:top_n]:
                        rows.append({
                            "Ticker": r["ticker"],
                            "Up Prob": f"{r['probability']:.1%}",
                            "Confidence": f"{r['confidence']:.1%}",
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.write("No buy signals found with current thresholds.")

            with col_avoid:
                st.subheader(f"▼ Top Avoid Signals ({len(avoid_list)} found)")
                if avoid_list:
                    rows = []
                    for r in avoid_list[:top_n]:
                        rows.append({
                            "Ticker": r["ticker"],
                            "Up Prob": f"{r['probability']:.1%}",
                            "Confidence": f"{r['confidence']:.1%}",
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.write("No avoid signals found with current thresholds.")
