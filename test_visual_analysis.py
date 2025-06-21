"""
Visual EDA Script: Rolling Volatility, Return, Correlation
Run this script to generate EDA plots for any stock.
"""

from src.data_loader import download_stock_data
from src.feature_engineering import add_technical_indicators
from src.visual_analysis import (
    plot_rolling_volatility,
    plot_rolling_return,
    plot_feature_correlation
)

# Step 1: Download historical data
ticker = "AAPL"
start = "2020-01-01"
end = "2024-12-31"

print(f"📥 Downloading data for {ticker} from {start} to {end}...")
df = download_stock_data(ticker, start, end)

# Step 2: Add technical indicators
print("🧠 Adding indicators...")
df = add_technical_indicators(df)

# Step 3: Plot rolling volatility
print("📊 Plotting rolling volatility...")
plot_rolling_volatility(df, window=20)

# Step 4: Plot rolling return
print("📈 Plotting rolling return...")
plot_rolling_return(df, window=10)

# Step 5: Plot feature correlation
print("🔗 Plotting feature correlation matrix...")
plot_feature_correlation(df)

print("✅ EDA analysis complete.")
