import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_rolling_volatility(data, window=20):
    """
    Plot rolling volatility over time.
    """
    rolling_vol = data['daily_return'].rolling(window).std()

    plt.figure(figsize=(12, 4))
    plt.plot(rolling_vol, label=f"{window}-Day Rolling Volatility", color='orange')
    plt.title(f"{window}-Day Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rolling_return(data, window=10):
    """
    Plot rolling return over time.
    """
    rolling_ret = data['close'].pct_change(periods=window)

    plt.figure(figsize=(12, 4))
    plt.plot(rolling_ret, label=f"{window}-Day Rolling Return", color='green')
    plt.title(f"{window}-Day Rolling Return")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_correlation(data, features=None):
    """
    Plot correlation matrix between selected features.
    """
    if features is None:
        features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower']

    corr_data = data[features].dropna()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("📊 Feature Correlation Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
