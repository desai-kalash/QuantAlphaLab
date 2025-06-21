import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.backtester import run_full_backtest
from src.data_loader import download_stock_data
from src.feature_engineering import add_technical_indicators
from src.labeling import add_forward_return_labels, clean_labeled_data
from src.model import train_and_evaluate_pipeline

# 1. Load and prepare data
df = download_stock_data("AAPL", "2020-01-01", "2024-12-31")
df = add_technical_indicators(df)
df = add_forward_return_labels(df)
df = clean_labeled_data(df)

# 2. Train model and run backtest
model = train_and_evaluate_pipeline(df)
backtest_data, metrics = run_full_backtest(model, df)

# 3. Build results dict
backtest_results = {
    "dates": backtest_data.index.tolist(),
    "strategy_equity": backtest_data["equity_curve"].tolist(),
    "buy_hold_equity": backtest_data["buy_and_hold"].tolist(),
    "metrics": metrics
}

# 4. Display equity curve
st.title("📈 AI Alpha Strategy vs Buy & Hold")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(backtest_results["dates"], backtest_results["strategy_equity"], label="AI Strategy", color="blue")
ax.plot(backtest_results["dates"], backtest_results["buy_hold_equity"], label="Buy & Hold", linestyle="--", color="orange")
ax.set_ylabel("Portfolio Value ($)")
ax.set_title("Equity Curve Comparison")
ax.legend()
st.pyplot(fig)

# 5. Show performance metrics
st.subheader("📊 Strategy Performance Summary")
cols = st.columns(4)
for i, (key, val) in enumerate(backtest_results["metrics"].items()):
    cols[i % 4].metric(label=key, value=str(val))
