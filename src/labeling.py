"""
Labeling Module
Creates trading signals based on forward returns
"""
import pandas as pd
import numpy as np


def classify_signal(future_return, buy_threshold=0.02, sell_threshold=-0.02):
    """
    Classify trading signal based on future return
    
    Args:
        future_return (float): Forward return value
        buy_threshold (float): Minimum return for Buy signal (default: 2%)
        sell_threshold (float): Maximum return for Sell signal (default: -2%)
        
    Returns:
        int: Trading signal (1=Buy, 0=Hold, -1=Sell)
    """
    if pd.isna(future_return):
        return 0  # Hold if no data
    
    if future_return > buy_threshold:
        return 1   # Buy
    elif future_return < sell_threshold:
        return -1  # Sell
    else:
        return 0   # Hold


def add_forward_return_labels(data, lookahead_days=3, buy_threshold=0.02, sell_threshold=-0.02):
    """
    Add forward-looking return labels to stock data
    
    Args:
        data (pd.DataFrame): Stock data with 'close' price column
        lookahead_days (int): Number of days to look ahead (default: 3)
        buy_threshold (float): Minimum return for Buy signal (default: 2%)
        sell_threshold (float): Maximum return for Sell signal (default: -2%)
        
    Returns:
        pd.DataFrame: Data with future_return and Signal columns added
    """
    print(f"✨ Creating labels with {lookahead_days}-day lookahead...")
    
    # Make a copy to avoid modifying original data
    df = data.copy()
    
    try:
        # Calculate future returns
        df['future_return'] = df['close'].shift(-lookahead_days) / df['close'] - 1
        
        # Apply classification function
        df['Signal'] = df['future_return'].apply(
            lambda x: classify_signal(x, buy_threshold, sell_threshold)
        )
        
        # Convert to integer type
        df['Signal'] = df['Signal'].astype(int)
        
        # Print label distribution
        label_counts = df['Signal'].value_counts().sort_index()
        print("📊 Label distribution:")
        signal_names = {-1: "Sell", 0: "Hold", 1: "Buy"}
        for signal, count in label_counts.items():
            print(f"   {signal_names[signal]} ({signal}): {count}")
        
        print(f"✅ Labels created successfully")
        return df
        
    except Exception as e:
        print(f"❌ Error creating labels: {str(e)}")
        raise


def clean_labeled_data(data, required_features=None):
    """
    Clean data by removing rows with missing features or labels
    
    Args:
        data (pd.DataFrame): Labeled stock data
        required_features (list): List of required feature columns
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    print("🧹 Cleaning labeled data...")
    
    original_length = len(data)
    
    # Default required features
    if required_features is None:
        required_features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower']
    
    # Only check features that exist in the data
    available_features = [col for col in required_features if col in data.columns]
    
    # Add Signal to required columns
    required_columns = available_features + ['Signal']
    
    # Remove rows with missing values
    cleaned_data = data.dropna(subset=required_columns)
    
    removed_rows = original_length - len(cleaned_data)
    print(f"📊 Removed {removed_rows} rows with missing data")
    print(f"📊 Final clean dataset: {cleaned_data.shape}")
    
    return cleaned_data


def map_labels_for_model(labels, model_type="xgboost"):
    """
    Map labels for specific model requirements
    
    Args:
        labels (pd.Series): Original labels (-1, 0, 1)
        model_type (str): Type of model ("xgboost" or "sklearn")
        
    Returns:
        pd.Series: Mapped labels
    """
    if model_type.lower() == "xgboost":
        # XGBoost requires non-negative labels: -1→0, 0→1, 1→2
        label_map = {-1: 0, 0: 1, 1: 2}
        mapped_labels = labels.map(label_map)
        print("🔁 Mapped labels for XGBoost: Sell→0, Hold→1, Buy→2")
        return mapped_labels
    else:
        return labels  # Keep original labels for other models


def reverse_label_mapping(predictions, model_type="xgboost"):
    """
    Reverse label mapping from model predictions to trading signals
    
    Args:
        predictions (array-like): Model predictions
        model_type (str): Type of model used
        
    Returns:
        array-like: Trading signals (-1, 0, 1)
    """
    if model_type.lower() == "xgboost":
        # Reverse XGBoost mapping: 0→-1, 1→0, 2→1
        reverse_map = {0: -1, 1: 0, 2: 1}
        return [reverse_map[pred] for pred in predictions]
    else:
        return predictions


# Example usage (for testing)
if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    # Download sample data
    data = yf.download("AAPL", start="2023-01-01", end="2023-12-31")
    data.columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Add labels
    labeled_data = add_forward_return_labels(data)
    print(f"📊 Labeled data shape: {labeled_data.shape}")