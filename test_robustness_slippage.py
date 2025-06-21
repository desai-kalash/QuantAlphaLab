import pandas as pd
import warnings
from src.data_loader import download_stock_data
from src.feature_engineering import add_technical_indicators
from src.labeling import add_forward_return_labels, clean_labeled_data
from src.model import train_and_evaluate_pipeline
from src.backtester import run_full_backtest

warnings.filterwarnings("ignore")

# 1. Download data
print("📈 Downloading AAPL data from 2020–2024...")
df = download_stock_data("AAPL", "2020-01-01", "2024-12-31")
df = add_technical_indicators(df)
df = add_forward_return_labels(df)
df = clean_labeled_data(df)

# 2. Train base model
print("✅ Training model on clean data...")
model = train_and_evaluate_pipeline(df)

# 3. Run backtests with different slippage levels
slippage_rates = [0.000, 0.001, 0.0025, 0.005]  # 0%, 0.10%, 0.25%, 0.50%

for slip in slippage_rates:
    print(f"\n🔁 Running backtest with {slip*100:.2f}% slippage...")
    _, metrics = run_full_backtest(
        model,
        df,
        transaction_cost=slip,
        cooldown_days=1,
        plot_results=True
    )

    print(f"📊 Slippage {slip*100:.2f}% → Final Capital: ${metrics['final_capital']:,.2f} | Sharpe: {metrics['sharpe_ratio']:.2f} | Drawdown: {metrics['max_drawdown_pct']:.2f}%")
