import pandas as pd
from src.data_loader import download_stock_data
from src.feature_engineering import add_technical_indicators
from src.labeling import add_forward_return_labels, clean_labeled_data
from src.model import train_and_evaluate_pipeline
from src.backtester import run_full_backtest

# -----------------------------------
# Define feature variants
# -----------------------------------
feature_sets = {
    'All Features': ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower'],
    'No RSI': ['SMA_20', 'SMA_50', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower'],
    'No MACD': ['SMA_20', 'SMA_50', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower'],
    'No SMA_50': ['SMA_20', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower'],
    'No Bollinger Bands': ['SMA_20', 'SMA_50', 'RSI', 'MACD'],
}

# -----------------------------------
# Load and prepare data
# -----------------------------------
print("📈 Downloading data...")
df = download_stock_data("AAPL", "2020-01-01", "2024-12-31")
df = add_technical_indicators(df)
df = add_forward_return_labels(df)
df = clean_labeled_data(df)

# -----------------------------------
# Loop over each feature set
# -----------------------------------
for label, features in feature_sets.items():
    print(f"\n🔬 Testing Feature Set: {label}")
    
    # Train model
    model = train_and_evaluate_pipeline(df, feature_cols=features)
    
    # Run backtest
    _, metrics = run_full_backtest(model, df, feature_cols=features)

    print(f"📊 {label} → Final Capital: ${metrics['final_capital']:.2f} | Sharpe: {metrics['sharpe_ratio']:.2f} | Drawdown: {metrics['max_drawdown_pct']:.2f}%")
