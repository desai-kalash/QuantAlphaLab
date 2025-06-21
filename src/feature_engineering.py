"""
Feature Engineering Module
Adds technical indicators to stock data
"""
import pandas as pd
import pandas_ta as ta


def add_technical_indicators(data):
    """
    Add technical indicators to stock price data
    
    Args:
        data (pd.DataFrame): Stock data with OHLCV columns
        
    Returns:
        pd.DataFrame: Data with technical indicators added
    """
    print("🧠 Adding technical indicators...")
    
    # Make a copy to avoid modifying original data
    df = data.copy()
    
    try:
        # Simple Moving Averages
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['SMA_50'] = ta.sma(df['close'], length=50)
        print("✅ Added SMA indicators")
        
        # RSI (Relative Strength Index)
        df['RSI'] = ta.rsi(df['close'], length=14)
        print("✅ Added RSI indicator")
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.macd(df['close'])
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['Signal_Line'] = macd['MACDs_12_26_9']
            print("✅ Added MACD indicators")
        else:
            print("⚠️ MACD calculation failed")
        
        # Bollinger Bands
        bb = ta.bbands(df['close'])
        if bb is not None and bb.shape[1] >= 3:
            df['BB_upper'] = bb.iloc[:, 0]
            df['BB_middle'] = bb.iloc[:, 1]
            df['BB_lower'] = bb.iloc[:, 2]
            print("✅ Added Bollinger Bands")
        else:
            print("⚠️ Bollinger Bands calculation failed")
        
        print(f"📊 Total features added: {len(df.columns) - len(data.columns)}")
        return df
        
    except Exception as e:
        print(f"❌ Error adding technical indicators: {str(e)}")
        raise


def get_feature_columns():
    """
    Get list of feature columns for model training
    
    Returns:
        list: List of feature column names
    """
    return ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower']


def validate_features(data, required_features=None):
    """
    Validate that required features exist in the data
    
    Args:
        data (pd.DataFrame): Stock data
        required_features (list): List of required feature names
        
    Returns:
        list: Available features from the required list
    """
    if required_features is None:
        required_features = get_feature_columns()
    
    available_features = [col for col in required_features if col in data.columns]
    missing_features = [col for col in required_features if col not in data.columns]
    
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
    
    print(f"✅ Available features: {available_features}")
    return available_features


# Example usage (for testing)
if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    # Download sample data
    data = yf.download("AAPL", start="2023-01-01", end="2023-12-31")
    data.columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Add technical indicators
    enhanced_data = add_technical_indicators(data)
    
    # Validate features
    features = validate_features(enhanced_data)
    print(f"📊 Final shape: {enhanced_data.shape}")