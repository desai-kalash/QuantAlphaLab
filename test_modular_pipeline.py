"""
Test script for the modularized Alpha Detection Engine
Now supports command-line input for ticker and date range
"""

# --- IMPORTS ---
import sys
import argparse
from datetime import datetime
from src.data_loader import download_stock_data
from src.feature_engineering import add_technical_indicators
from src.labeling import add_forward_return_labels, clean_labeled_data
from src.model import train_and_evaluate_pipeline
from src.backtester import run_full_backtest

# --- CLI Argument Parsing ---
parser = argparse.ArgumentParser(description="Run Alpha Detection Engine for any stock.")
parser.add_argument("--ticker", default="AAPL", help="Stock ticker symbol (e.g., AAPL, TSLA, BTC-USD)")
parser.add_argument("--start", default="2020-01-01", help="Start date in YYYY-MM-DD")
parser.add_argument("--end", default="2024-12-31", help="End date in YYYY-MM-DD")

args = parser.parse_args()
TICKER = args.ticker.upper()
START_DATE = args.start
END_DATE = args.end

# --- PIPELINE START ---
print(f"📥 Downloading stock data for {TICKER} from {START_DATE} to {END_DATE}...")
df = download_stock_data(ticker=TICKER, start_date=START_DATE, end_date=END_DATE)

if df.empty:
    print(f"❌ No valid data found for {TICKER}. Exiting...")
    sys.exit()

print("🧠 Adding technical indicators...")
df = add_technical_indicators(df)

print("🎯 Labeling data...")
df = add_forward_return_labels(df)
df = clean_labeled_data(df)

print("🤖 Training model...")
model = train_and_evaluate_pipeline(df)

print("📊 Running backtest...")
metrics = run_full_backtest(model, df)

print("✅ Pipeline completed successfully!")
