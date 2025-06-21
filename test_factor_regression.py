import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from src.data_loader import download_stock_data
from src.feature_engineering import add_technical_indicators
from src.labeling import add_forward_return_labels, clean_labeled_data
from src.model import train_and_evaluate_pipeline
from src.backtester import run_full_backtest

# 1. Get backtest results
df = download_stock_data("AAPL", "2020-01-01", "2024-12-31")
df = add_technical_indicators(df)
df = add_forward_return_labels(df)
df = clean_labeled_data(df)

model = train_and_evaluate_pipeline(df)
backtest_data, _ = run_full_backtest(model, df)

# 2. Get strategy daily returns
backtest_data['strategy_return'] = backtest_data['equity_curve'].pct_change()
returns = backtest_data[['strategy_return']].dropna()
returns.index = pd.to_datetime(returns.index)

# 3. Load Fama-French factors
# Load Fama-French 3-Factor Daily data from local CSV
ff = pd.read_csv("data/F-F_Research_Data_Factors_daily.csv", skiprows=3)

# Clean formatting
ff = ff.rename(columns={'Unnamed: 0': 'Date'})
ff = ff[ff['Date'].str.len() == 8]  # Filter out footer rows
ff['Date'] = pd.to_datetime(ff['Date'], format='%Y%m%d')
ff.set_index('Date', inplace=True)

# Convert to numeric and scale to decimal (from percent)
ff = ff.apply(pd.to_numeric, errors='coerce') / 100
ff = ff.dropna()


# Keep only Mkt-RF, SMB, HML, MOM, RF
factors = ff[['Mkt-RF', 'SMB', 'HML', 'RF']].copy()
factors = factors / 100  # convert % to decimal

# 4. Resample to daily (forward fill monthly values)
factors = factors.resample('D').ffill()

# 5. Merge
merged = returns.join(factors, how='inner')
merged['excess_return'] = merged['strategy_return'] - merged['RF']

# 6. Regression
X = merged[['Mkt-RF', 'SMB', 'HML']]
y = merged['excess_return']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# 7. Results
print("\n📊 FACTOR REGRESSION RESULTS:")
print(model.summary())

# Optional: plot cumulative excess return
merged['cumulative_alpha'] = merged['excess_return'].cumsum()
merged['cumulative_alpha'].plot(title="Cumulative Excess Return (Strategy Alpha)", figsize=(10, 4))
plt.grid(True)
plt.show()
