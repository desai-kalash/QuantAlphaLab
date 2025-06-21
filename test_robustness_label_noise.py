import pandas as pd
import numpy as np
from src.data_loader import download_stock_data
from src.feature_engineering import add_technical_indicators
from src.labeling import add_forward_return_labels, clean_labeled_data, map_labels_for_model
from src.model import train_xgboost_model, evaluate_model
from src.backtester import run_full_backtest

def add_label_noise(y, noise_level=0.1, random_state=42):
    """
    Introduce noise into the label array by randomly changing a percentage of labels.
    """
    np.random.seed(random_state)
    y_noisy = y.copy()
    n_noisy = int(len(y) * noise_level)
    indices = np.random.choice(len(y), n_noisy, replace=False)

    unique_labels = sorted(list(set(y)))
    for i in indices:
        current = y[i]
        options = [label for label in unique_labels if label != current]
        y_noisy[i] = np.random.choice(options)
    
    return y_noisy

# === Load and prepare data ===
df = download_stock_data("AAPL", "2020-01-01", "2024-12-31")
df = add_technical_indicators(df)
df = add_forward_return_labels(df)
df = clean_labeled_data(df)

# Feature matrix
from src.model import prepare_features_and_labels
X, y = prepare_features_and_labels(df)
y = map_labels_for_model(y, "xgboost")

# Time split for robustness
split_index = int(len(X) * 0.7)
X_train, y_train = X.iloc[:split_index], y[:split_index]
X_test, y_test = X.iloc[split_index:], y[split_index:]

# === Run tests for different noise levels ===
for noise_pct in [0.0, 0.1, 0.2, 0.3]:
    print(f"\n🎯 Running with {int(noise_pct*100)}% label noise...")
    y_train_noisy = add_label_noise(y_train, noise_level=noise_pct)

    # Train model
    model = train_xgboost_model(X_train, y_train_noisy)

    # Evaluate
    print(f"\n📊 Evaluation on Clean Test Data:")
    evaluate_model(model, X_test, y_test)

    # Run backtest on clean test data
    test_df = df.iloc[split_index:].copy()
    _ = run_full_backtest(model, test_df)
