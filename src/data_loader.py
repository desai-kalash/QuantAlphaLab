"""
Data Loader Module
Downloads and cleans stock data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd


def download_stock_data(ticker="AAPL", start_date="2020-01-01", end_date="2024-12-31"):
    import yfinance as yf
    import pandas as pd

    print(f"📈 Downloading {ticker} data from {start_date} to {end_date}...")

    try:
        data = yf.download(ticker, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

        # ✅ Flatten columns: handle MultiIndex (e.g., ('AAPL', 'Close'))
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[1].lower() for col in data.columns]
        else:
            data.columns = [col.lower() for col in data.columns]

        print("📊 Cleaned columns:", data.columns.tolist())

        # ✅ Handle missing or alternative close columns
        if 'close' in data.columns:
            data['daily_return'] = data['close'].pct_change()
        elif 'adj_close' in data.columns:
            print("⚠️ Using 'adj_close' instead of 'close'")
            data['close'] = data['adj_close']
            data['daily_return'] = data['close'].pct_change()
        else:
            raise ValueError(f"❌ 'close' or 'adj_close' column not found in {ticker} data.")

        if data.empty or len(data) < 100:
            raise ValueError(f"❌ Not enough data for {ticker} — only {len(data)} rows.")

        print(f"✅ Downloaded {len(data)} rows of data")
        return data

    except Exception as e:
        print(f"🚨 Error fetching {ticker}: {e}")
        return pd.DataFrame()




def save_data(data: pd.DataFrame, filepath: str):
    """Save DataFrame to CSV file."""
    data.to_csv(filepath)
    print(f"💾 Data saved to {filepath}")


def load_data(filepath: str) -> pd.DataFrame:
    """Load DataFrame from CSV file."""
    try:
        data = pd.read_csv(filepath, index_col=0)
        print(f"📂 Data loaded from {filepath}")
        return data
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return pd.DataFrame()
