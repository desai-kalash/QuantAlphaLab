import pandas as pd
from src.backtester import simulate_trading

def test_simulate_trading_returns_dataframe():
    # Minimal dummy data
    df = pd.DataFrame({
        'close': [100, 102, 101, 103],
        'predicted_signal': [None, 1, 0, None]
    })

    # Run simulation
    df_result = simulate_trading(df.copy())

    # Check that result contains equity curve
    assert 'equity_curve' in df_result.columns

