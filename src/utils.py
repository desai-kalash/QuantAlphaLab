import numpy as np
import pandas as pd

# === 📈 Performance Metrics Utilities ===

def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Calculate cumulative returns."""
    return (1 + returns).cumprod()

def calculate_drawdowns(cumulative_returns: pd.Series) -> pd.Series:
    """Calculate drawdowns from cumulative returns."""
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate annualized Sharpe ratio."""
    excess_returns = returns - risk_free_rate
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

# === 🤖 Strategy Return Simulation ===

def simulate_strategy_returns(
    data: pd.DataFrame,
    transaction_cost: float = 0.001,
    slippage: float = 0.001,
    stop_loss: float = None,
    take_profit: float = None,
    max_holding_days: int = None
) -> pd.Series:
    """
    Simulate strategy returns based on ML signals.
    
    Args:
        data: DataFrame with 'Signal' and 'future_return' columns
        transaction_cost: % cost per trade
        slippage: % slippage per trade
        stop_loss: stop loss threshold (e.g., 0.05 for 5%)
        take_profit: take profit threshold (e.g., 0.10 for 10%)
        max_holding_days: max days to hold a position

    Returns:
        pd.Series of net daily returns after applying trading logic
    """
    signal = data["Signal"].shift(1).fillna(0)
    future_returns = data["future_return"].fillna(0)
    net_returns = pd.Series(index=data.index, dtype=float)

    in_trade = False
    trade_return = 0
    holding_days = 0
    trade_direction = 0

    for i in range(len(data)):
        if not in_trade and signal.iloc[i] != 0:
            in_trade = True
            trade_return = 0
            holding_days = 0
            trade_direction = signal.iloc[i]

        if in_trade:
            holding_days += 1
            ret = future_returns.iloc[i] * trade_direction
            trade_return += ret

            # Exit conditions
            exit = False
            if stop_loss and trade_return <= -stop_loss:
                exit = True
            if take_profit and trade_return >= take_profit:
                exit = True
            if max_holding_days and holding_days >= max_holding_days:
                exit = True

            if exit:
                in_trade = False
                trade_return -= transaction_cost + slippage
                net_returns.iloc[i] = trade_return
            else:
                net_returns.iloc[i] = 0  # Still holding
        else:
            net_returns.iloc[i] = 0  # No position

    return net_returns.fillna(0)

# === 📊 Portfolio Simulation ===

def simulate_portfolio_equity_curve(
    strategy_returns: pd.Series,
    initial_capital: float = 100000,
    leverage: float = 1.0,
    capital_per_trade_pct: float = 1.0
) -> pd.Series:
    """
    Simulate portfolio equity curve from strategy returns.

    Args:
        strategy_returns: Series of returns
        initial_capital: Total portfolio capital
        leverage: leverage multiplier
        capital_per_trade_pct: fraction of capital per trade (0-1)

    Returns:
        pd.Series of portfolio value over time
    """
    capital_per_trade = initial_capital * (capital_per_trade_pct / 100.0)
    leveraged_returns = strategy_returns * leverage
    pnl = capital_per_trade * leveraged_returns
    equity_curve = pnl.cumsum() + initial_capital
    return equity_curve

# === 🧠 Model Output Caching for Streamlit ===

_model_outputs_cache = {}

def save_model_outputs(model, y_test, y_pred, X_test):
    """Cache model and outputs in memory for Streamlit session use."""
    global _model_outputs_cache
    _model_outputs_cache["model"] = model
    _model_outputs_cache["y_test"] = y_test
    _model_outputs_cache["y_pred"] = y_pred
    _model_outputs_cache["X_test"] = X_test

def load_latest_model_outputs():
    """Retrieve last saved model and outputs."""
    global _model_outputs_cache
    return (
        _model_outputs_cache["model"],
        _model_outputs_cache["y_test"],
        _model_outputs_cache["y_pred"],
        _model_outputs_cache["X_test"]
    )
