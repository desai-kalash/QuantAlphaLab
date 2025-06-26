import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src/ to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import (
    load_latest_model_outputs,
    simulate_strategy_returns,
    simulate_portfolio_equity_curve,
    calculate_drawdowns,
)

st.set_page_config(page_title="Portfolio Analysis", layout="wide")
st.title("📂 Portfolio-Level Analysis")

st.markdown("""
Understand how your ML strategy would behave if it were deployed in the market. Simulate starting capital, leverage, and more to visualize the equity curve and portfolio metrics.
""")

# Load model outputs
try:
    model, y_test, y_pred, X_test = load_latest_model_outputs()
    st.success("✅ Model outputs loaded.")
except Exception as e:
    st.error(f"❌ Failed to load model outputs: {e}")
    st.stop()

# Prepare DataFrame
df = X_test.copy()
y_pred_flat = y_pred.ravel() if hasattr(y_pred, "ravel") else y_pred
df["Signal"] = pd.Series(y_pred_flat, index=df.index).map({0: -1, 1: 0, 2: 1})

# Identify price column
price_col = next((col for col in ["close", "Close", "Adj Close"] if col in df.columns), None)
if price_col is None:
    st.error("❌ Price column not found.")
    st.stop()

# Compute returns
df["future_return"] = df[price_col].pct_change().shift(-1)
df["strategy_return"] = simulate_strategy_returns(df)
st.write("Signal Value Counts:", df["Signal"].value_counts())
st.write("Strategy Return Stats:", df["strategy_return"].describe())



# --- Sidebar Portfolio Settings ---
st.sidebar.header("📊 Portfolio Settings")
initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1000, value=100000, step=1000)
leverage = st.sidebar.slider("Leverage", 1.0, 5.0, 1.0, step=0.1)
capital_per_trade_pct = st.sidebar.slider("Capital per Trade (%)", 0.01, 1.0, 1.0, step=0.01)

# Simulate Portfolio Equity
equity_curve = simulate_portfolio_equity_curve(
    df["strategy_return"],
    initial_capital=initial_capital,
    leverage=leverage,
    capital_per_trade_pct=capital_per_trade_pct * 100
)

# Drawdowns
drawdowns = calculate_drawdowns(equity_curve)

# --- Portfolio Equity Curve ---
st.subheader("📈 Portfolio Equity Curve")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(equity_curve, label="💼 Portfolio Value")
ax1.set_title("Portfolio Growth Over Time")
ax1.set_ylabel("Portfolio ($)")
ax1.legend()
st.pyplot(fig1)

# --- Portfolio Metrics ---
st.subheader("📊 Portfolio Metrics")
total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
volatility = df["strategy_return"].std() * np.sqrt(252)
sharpe = df["strategy_return"].mean() / df["strategy_return"].std() * np.sqrt(252)
max_drawdown = drawdowns.min()

metrics = {
    "Final Portfolio Value": f"${equity_curve.iloc[-1]:,.2f}",
    "Total Return": f"{total_return:.2%}",
    "Annualized Volatility": f"{volatility:.2%}",
    "Sharpe Ratio": f"{sharpe:.2f}",
    "Max Drawdown": f"{max_drawdown:.2%}"
}
metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["value"])
st.table(metrics_df)

# --- Drawdown Chart ---
st.subheader("📉 Portfolio Drawdown")
fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.fill_between(drawdowns.index, drawdowns, color="red")
ax2.set_title("Portfolio Drawdown Over Time")
st.pyplot(fig2)
