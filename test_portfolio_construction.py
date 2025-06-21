import os
import pandas as pd
import matplotlib.pyplot as plt

# 📁 Folder where individual backtest files are saved
RESULTS_FOLDER = "results"
TICKERS = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"]
capital = 1000  # assumed initial capital per ticker

# 📊 Load equity curves from each ticker's backtest
portfolio_df = pd.DataFrame()

for ticker in TICKERS:
    filepath = os.path.join(RESULTS_FOLDER, f"{ticker}_backtest.csv")
    if not os.path.exists(filepath):
        print(f"❌ Missing file for {ticker}. Skipping.")
        continue

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df = df[["equity_curve"]].rename(columns={"equity_curve": ticker})
    portfolio_df = pd.concat([portfolio_df, df], axis=1)

# 🧹 Drop rows with any missing data (due to differing trading days)
portfolio_df.dropna(inplace=True)
print(f"✅ Loaded equity curves for: {portfolio_df.columns.tolist()}")

# 📈 Equal-weighted portfolio (simple average)
portfolio_df["Equal_Weighted"] = portfolio_df.mean(axis=1)

# 🔍 Normalize all curves for plotting (starting at 1000)
normalized = portfolio_df.div(portfolio_df.iloc[0]) * capital

# 📊 Plotting
plt.figure(figsize=(12, 6))
for col in normalized.columns[:-1]:
    plt.plot(normalized.index, normalized[col], linestyle="--", alpha=0.5, label=col)

plt.plot(normalized.index, normalized["Equal_Weighted"], label="📊 Equal Weighted Portfolio", color="black", linewidth=2)

plt.title("Portfolio Growth: Equal-Weighted Strategy")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
