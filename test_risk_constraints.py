from src.data_loader import download_stock_data
from src.feature_engineering import add_technical_indicators
from src.labeling import add_forward_return_labels, clean_labeled_data
from src.model import train_and_evaluate_pipeline
from src.backtester import run_full_backtest

print("📈 Running risk-constrained backtest on AAPL")

# Load and process data
df = download_stock_data("AAPL", "2020-01-01", "2024-12-31")
df = add_technical_indicators(df)
df = add_forward_return_labels(df)
df = clean_labeled_data(df)

# Train model
model = train_and_evaluate_pipeline(df)

# Backtest with risk limits
_ = run_full_backtest(
    model, df,
    initial_capital=1000,
    transaction_cost=0.001,
    cooldown_days=1,
    plot_results=True,
    feature_cols=['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower'],
    stop_loss_threshold=-0.05,       # stop trading if 1-day return < -5%
    max_drawdown_limit=-0.25         # stop strategy if portfolio drops 25%
)
