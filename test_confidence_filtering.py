from src.data_loader import download_stock_data
from src.feature_engineering import add_technical_indicators
from src.labeling import add_forward_return_labels, clean_labeled_data
from src.model import train_and_evaluate_pipeline, prepare_features_and_labels
from src.backtester import run_full_backtest

# 1. Load and process data
df = download_stock_data("AAPL", "2020-01-01", "2024-12-31")
df = add_technical_indicators(df)
df = add_forward_return_labels(df)
df = clean_labeled_data(df)

# 2. Train model
model = train_and_evaluate_pipeline(df)

# 3. Run backtest with confidence filtering
print("📊 Running backtest with confidence threshold = 0.6")
run_full_backtest(model, df, feature_cols=[
    'SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower'
], plot_results=True)
