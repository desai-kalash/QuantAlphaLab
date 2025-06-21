"""
Backtesting Module
Simulates trading strategies and calculates performance metrics
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple


def generate_signals(model, data, feature_cols=None, threshold=0.6):
    """
    Generate trading signals using model prediction probabilities.

    Args:
        model: Trained ML model with predict_proba
        data (pd.DataFrame): Feature dataframe
        feature_cols (list): List of feature column names
        threshold (float): Confidence threshold (0.0 to 1.0)

    Returns:
        pd.DataFrame: Data with predicted_signal column
    """
    print("🔮 Generating trading signals with confidence filtering...")

    if feature_cols is None:
        feature_cols = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower']

    available_features = [col for col in feature_cols if col in data.columns]
    X = data[available_features].dropna()

    # Get class probabilities
    probas = model.predict_proba(X)
    max_probs = probas.max(axis=1)

    # Handle model output shape (especially for CatBoost)
    preds = model.predict(X)
    import numpy as np
    if isinstance(preds[0], np.ndarray):
        preds = [int(p[0]) for p in preds]
    else:
        preds = preds.tolist()

    from src.labeling import reverse_label_mapping
    raw_signals = reverse_label_mapping(preds, model_type="xgboost")  # xgboost-style label mapping

    # Apply confidence threshold
    confident_signals = [
        signal if prob >= threshold else 0
        for signal, prob in zip(raw_signals, max_probs)
    ]

    df = data.copy()
    df['predicted_signal'] = np.nan
    df.loc[X.index, 'predicted_signal'] = confident_signals

    # Print signal distribution
    signal_counts = pd.Series(confident_signals).value_counts().sort_index()
    print("📊 Predicted signal distribution (after filtering):")
    signal_names = {-1: "Sell", 0: "Hold", 1: "Buy"}
    for signal, count in signal_counts.items():
        print(f"   {signal_names[signal]} ({signal}): {count}")

    return df




def simulate_trading(
    data,
    initial_capital=1000,
    transaction_cost=0.001,
    cooldown_days=1,
    stop_loss_threshold=None,        # e.g., -0.05 for -5%
    max_drawdown_limit=None          # e.g., -0.20 for -20%
):
    print(f"💰 Simulating trading with ${initial_capital:,.2f} initial capital...")
    df = data.copy()
    df['predicted_signal'] = df['predicted_signal'].shift(1)
    df['next_close'] = df['close'].shift(-1)
    df['price_change'] = df['next_close'] / df['close'] - 1

    capital = initial_capital
    equity_curve = []
    last_trade_day = -np.inf
    trades_executed = 0
    peak_capital = capital
    index_list = df.index.tolist()

    for i, row in df.iterrows():
        signal = row['predicted_signal']
        day_index = index_list.index(i)

        # Handle missing data or cooldown
        if pd.isna(signal) or pd.isna(row['price_change']):
            equity_curve.append(capital)
            continue
        if (day_index - last_trade_day) <= cooldown_days:
            equity_curve.append(capital)
            continue

        # Stop if portfolio hits drawdown limit
        current_drawdown = (capital - peak_capital) / peak_capital
        if max_drawdown_limit is not None and current_drawdown <= max_drawdown_limit:
            print(f"🛑 Max drawdown limit hit ({current_drawdown:.2%}). Halting trades.")
            equity_curve.append(capital)
            continue

        # Buy logic
        if signal == 1 and (stop_loss_threshold is None or row['price_change'] > stop_loss_threshold):
            capital *= (1 + row['price_change'] - transaction_cost)
            last_trade_day = day_index
            trades_executed += 1

        # Sell logic
        elif signal == -1:
            capital *= (1 - transaction_cost)
            last_trade_day = day_index
            trades_executed += 1

        # Update peak
        peak_capital = max(peak_capital, capital)
        equity_curve.append(capital)

    df = df.iloc[:len(equity_curve)].copy()
    df['equity_curve'] = equity_curve
    df['buy_and_hold'] = (df['close'] / df['close'].iloc[0]) * initial_capital

    print(f"📊 Executed {trades_executed} trades")
    print(f"💰 Final capital: ${capital:,.2f}")
    return df



def calculate_performance_metrics(data, equity_col='equity_curve') -> Dict:
    """
    Calculate comprehensive performance metrics
    
    Args:
        data (pd.DataFrame): Data with equity curve
        equity_col (str): Name of equity curve column
        
    Returns:
        dict: Dictionary of performance metrics
    """
    print("📈 Calculating performance metrics...")
    
    equity = data[equity_col].dropna()
    
    # Basic metrics
    initial_capital = equity.iloc[0]
    final_capital = equity.iloc[-1]
    total_return = (final_capital / initial_capital) - 1
    
    # Calculate returns
    returns = equity.pct_change().dropna()
    
    # Sharpe ratio (annualized, assuming daily data)
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    
    # Maximum drawdown
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_drawdown = drawdown.min()
    
    # Trading metrics
    signals = data['predicted_signal'].dropna()
    buy_signals = (signals == 1).sum()
    sell_signals = (signals == -1).sum()
    total_trades = buy_signals + sell_signals
    
    # Win rate calculation
    trade_returns = []
    for signal, price_change in zip(data['predicted_signal'], data['price_change']):
        if signal == 1 and not pd.isna(price_change):  # Buy trades
            trade_returns.append(price_change > 0)
    
    win_rate = np.mean(trade_returns) if trade_returns else 0
    
    # Compile metrics
    metrics = {
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown * 100,
        'win_rate': win_rate,
        'win_rate_pct': win_rate * 100,
        'total_trades': total_trades,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'final_capital': final_capital,
        'initial_capital': initial_capital
    }
    
    return metrics


def print_performance_report(metrics: Dict):
    """
    Print formatted performance report
    
    Args:
        metrics (dict): Performance metrics dictionary
    """
    print("\n" + "="*50)
    print("📊 STRATEGY PERFORMANCE REPORT")
    print("="*50)
    print(f"💰 Total Return:        {metrics['total_return_pct']:.2f}%")
    print(f"📈 Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
    print(f"📉 Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
    print(f"⚖️ Win Rate:             {metrics['win_rate_pct']:.2f}%")
    print(f"📊 Total Trades:        {metrics['total_trades']}")
    print(f"🎯 Buy Signals:         {metrics['buy_signals']}")
    print(f"🎯 Sell Signals:        {metrics['sell_signals']}")
    print(f"💵 Final Capital:       ${metrics['final_capital']:,.2f}")
    print("="*50)


def plot_equity_curves(data, save_path=None):
    """
    Plot strategy equity curve vs buy and hold
    
    Args:
        data (pd.DataFrame): Data with equity curves
        save_path (str): Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot strategy vs buy and hold
    plt.plot(data.index, data['equity_curve'], 
             label='📈 AI Strategy', color='blue', linewidth=2)
    plt.plot(data.index, data['buy_and_hold'], 
             label='📉 Buy & Hold', color='orange', linestyle='--', linewidth=2)
    
    plt.title("Strategy Performance vs Buy & Hold", fontsize=14, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")
    
    plt.show()


def run_full_backtest(model, data, initial_capital=1000, transaction_cost=0.001, 
                      cooldown_days=1, feature_cols=None, plot_results=True,
                      stop_loss_threshold=None, max_drawdown_limit=None):

    """
    Run complete backtesting pipeline
    
    Args:
        model: Trained ML model
        data (pd.DataFrame): Stock data with features
        initial_capital (float): Starting capital
        transaction_cost (float): Transaction cost per trade
        cooldown_days (int): Minimum days between trades
        feature_cols (list): List of feature column names
        plot_results (bool): Whether to plot equity curves
        
    Returns:
        tuple: (backtested_data, performance_metrics)
    """
    print("🚀 Running full backtest pipeline...")
    
    # Generate signals
    signal_data = generate_signals(model, data, feature_cols)
    
    # Simulate trading
    backtest_data = simulate_trading(
    signal_data,
    initial_capital=initial_capital,
    transaction_cost=transaction_cost,
    cooldown_days=cooldown_days,
    stop_loss_threshold=stop_loss_threshold,
    max_drawdown_limit=max_drawdown_limit
)
    
    # Calculate metrics
    metrics = calculate_performance_metrics(backtest_data)
    
    # Print report
    print_performance_report(metrics)
    
    # Plot results
    if plot_results:
        plot_equity_curves(backtest_data)
    
    return backtest_data, metrics


# Example usage (for testing)
if __name__ == "__main__":
    print("Backtesting module loaded successfully!")