import pandas as pd
import numpy as np
from src.data_loader import download_stock_data
from src.feature_engineering import add_technical_indicators
from src.labeling import add_forward_return_labels, clean_labeled_data
from src.model import train_and_evaluate_pipeline
from src.backtester import run_full_backtest
import matplotlib.pyplot as plt

# -----------------------------------------------
# 🧨 Inject adversarial candles
# -----------------------------------------------
def inject_adversarial_noise(df, num_candles=10, magnitude=0.15, seed=42):
    np.random.seed(seed)
    df = df.copy()
    indices = np.random.choice(len(df), size=num_candles, replace=False)

    for i in indices:
        direction = np.random.choice([-1, 1])  # Spike up or down
        df.iloc[i]['high'] += direction * magnitude * df.iloc[i]['close']
        df.iloc[i]['low'] -= direction * magnitude * df.iloc[i]['close']
        df.iloc[i]['open'] = df.iloc[i]['close'] * (1 + direction * magnitude * 0.5)

    return df

# -----------------------------------------------
# 🚀 Run the test
# -----------------------------------------------
if __name__ == "__main__":
    print("📈 Downloading clean data...")
    df = download_stock_data("AAPL", "2020-01-01", "2024-12-31")
    df = add_technical_indicators(df)
    df = add_forward_return_labels(df)
    df = clean_labeled_data(df)

    print("✅ Training base model...")
    base_model = train_and_evaluate_pipeline(df)

    print("\n🧨 Injecting synthetic adversarial candles...")
    tampered_df = inject_adversarial_noise(df, num_candles=20, magnitude=0.2)

    print("🚀 Running backtest on tampered data...")
    backtest_data, metrics = run_full_backtest(base_model, tampered_df)

    print("\n✅ Completed adversarial candle robustness test.")
