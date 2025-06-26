import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add src/ to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import (
    load_latest_model_outputs,
    simulate_strategy_returns,
    calculate_cumulative_returns,
    calculate_drawdowns,
    calculate_sharpe_ratio
)

# Streamlit page setup
st.set_page_config(page_title="Strategy Backtest", layout="wide")
st.title("📈 Strategy Backtest & Signal Evaluation")

st.markdown("""
Backtest your ML-generated trading signals and compare the performance of your strategy against the asset.
""")

# Load model outputs
try:
    model, y_test, y_pred, X_test = load_latest_model_outputs()
    st.success("✅ Loaded model outputs.")
except Exception as e:
    st.error(f"❌ Could not load model outputs. Please train a model first.\n\n{e}")
    st.stop()

# Prepare DataFrame
df = X_test.copy()

# Flatten predictions to 1D
y_pred_flat = y_pred.ravel() if hasattr(y_pred, "ravel") else y_pred

# Add Signal column (decode classes back to signals)
df["Signal"] = pd.Series(y_pred_flat, index=df.index).map({0: -1, 1: 0, 2: 1})

# --- 🔍 Identify appropriate price column ---
price_col = None
for col in ["close", "Close", "Adj Close"]:
    if col in df.columns:
        price_col = col
        break

if price_col is None:
    st.error("❌ Price column not found. Please check your input data format.")
    st.stop()

# Compute future returns
df["future_return"] = df[price_col].pct_change().shift(-1)

# --- ⚙️ RISK & COST CONTROLS ---
st.sidebar.header("🛠️ Simulation Settings")

slippage = st.sidebar.slider("💸 Slippage (%)", 0.0, 1.0, 0.1, step=0.05) / 100
txn_cost = st.sidebar.slider("📉 Transaction Cost (%)", 0.0, 1.0, 0.1, step=0.05) / 100
stop_loss = st.sidebar.slider("🛑 Stop Loss (%)", 0.0, 10.0, 5.0, step=0.5) / 100
take_profit = st.sidebar.slider("💰 Take Profit (%)", 0.0, 20.0, 10.0, step=0.5) / 100
max_hold = st.sidebar.slider("⏳ Max Holding Days", 1, 10, 5)

# Simulate strategy with full risk controls
df["strategy_return"] = simulate_strategy_returns(
    data=df,
    transaction_cost=txn_cost,
    slippage=slippage,
    stop_loss=stop_loss,
    take_profit=take_profit,
    max_holding_days=max_hold
)

# Compute cumulative returns
df["strategy_cumret"] = calculate_cumulative_returns(df["strategy_return"])
df["asset_cumret"] = calculate_cumulative_returns(df["future_return"].fillna(0))

# --- 📊 Cumulative Returns Plot ---
st.subheader("📊 Cumulative Returns")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["strategy_cumret"], label="🤖 Strategy (Net)")
ax.plot(df["asset_cumret"], label="📈 Asset")
ax.set_title("Cumulative Returns (After Costs, SL/TP)")
ax.legend()
st.pyplot(fig)

# --- 📋 Performance Metrics ---
st.subheader("📋 Performance Summary")
final_return = df["strategy_cumret"].iloc[-1] - 1
volatility = df["strategy_return"].std() * np.sqrt(252)
sharpe = calculate_sharpe_ratio(df["strategy_return"])

metrics = {
    "Total Strategy Return": f"{final_return:.2%}",
    "Volatility (Annualized)": f"{volatility:.2%}",
    "Sharpe Ratio": f"{sharpe:.2f}"
}
st.table(metrics)

# --- 📉 Drawdown Chart ---
st.subheader("📉 Strategy Drawdown")
df["drawdown"] = calculate_drawdowns(df["strategy_cumret"])
fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.fill_between(df.index, df["drawdown"], color="red")
ax2.set_title("Strategy Drawdown")
st.pyplot(fig2)
