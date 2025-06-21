"""
Runs the Alpha Detection Pipeline for multiple tickers
and saves individual strategy reports for each.
"""

from src.data_loader import download_stock_data
from src.feature_engineering import add_technical_indicators
from src.labeling import add_forward_return_labels, clean_labeled_data
from src.model import train_and_evaluate_pipeline
from src.backtester import run_full_backtest
import os
import pandas as pd

# 🔁 List of tickers to run
tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"]
start_date = "2020-01-01"
end_date = "2024-12-31"

# 💾 Output folder
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# 🧾 Log metrics from each ticker
summary = []

for ticker in tickers:
    print(f"\n{'='*60}\n📈 Processing Ticker: {ticker}\n{'='*60}")
    
    try:
        # Load and process
        df = download_stock_data(ticker, start_date, end_date)
        if df.empty:
            print(f"⚠️ Skipping {ticker}: No data")
            continue

        df = add_technical_indicators(df)
        df = add_forward_return_labels(df)
        df = clean_labeled_data(df)

        # Train model
        model = train_and_evaluate_pipeline(df)

        # Run backtest
        backtest_data, metrics = run_full_backtest(model, df, plot_results=False)

        # Save summary
        metrics['ticker'] = ticker
        summary.append(metrics)

        # Save per-ticker CSV
        save_path = os.path.join(output_dir, f"{ticker}_backtest.csv")
        backtest_data.to_csv(save_path)
        print(f"✅ Backtest data saved to {save_path}")

    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")

# Save overall metrics summary
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(output_dir, "multi_ticker_summary.csv"), index=False)
print("\n📊 All ticker results saved to results/multi_ticker_summary.csv")
