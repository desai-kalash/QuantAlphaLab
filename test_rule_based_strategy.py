# test_rule_based_strategy.py

from src.data_loader import download_stock_data
from src.feature_engineering import add_technical_indicators
from src.backtester import simulate_trading, calculate_performance_metrics, print_performance_report, plot_equity_curves
import pandas as pd
import numpy as np

def generate_rule_based_signals(data):
    """
    Simpler rule-based signals:
    Buy when RSI < 30
    Sell when RSI > 70
    """
    df = data.copy()

    df['predicted_signal'] = np.where(df['RSI'] < 30, 1,
                               np.where(df['RSI'] > 70, -1, 0))

    # Show rule distribution
    signal_counts = df['predicted_signal'].value_counts().sort_index()
    signal_names = {-1: "Sell", 0: "Hold", 1: "Buy"}
    print("📊 Rule-based signal distribution:")
    for signal, count in signal_counts.items():
        print(f"   {signal_names[signal]} ({signal}): {count}")

    return df


# ------------------ MAIN SCRIPT ------------------

if __name__ == "__main__":
    # Load and prepare data
    df = download_stock_data("AAPL", "2020-01-01", "2024-12-31")
    df = add_technical_indicators(df)

    # Generate signals using rule-based logic
    df_with_signals = generate_rule_based_signals(df)

    # Run backtest
    result_df = simulate_trading(df_with_signals)
    metrics = calculate_performance_metrics(result_df)
    print_performance_report(metrics)
    plot_equity_curves(result_df)
