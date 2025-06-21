import pandas as pd
import numpy as np
from src.data_loader import download_stock_data
from src.feature_engineering import add_technical_indicators
from src.labeling import add_forward_return_labels, clean_labeled_data, map_labels_for_model, reverse_label_mapping
from src.model import train_xgboost_model, train_lightgbm_model, train_catboost_model, prepare_features_and_labels
from src.backtester import simulate_trading, calculate_performance_metrics, print_performance_report, plot_equity_curves

# Load and prepare data
df = download_stock_data("AAPL", "2020-01-01", "2024-12-31")
df = add_technical_indicators(df)
df = add_forward_return_labels(df)
df = clean_labeled_data(df)

# Prepare features and labels
X, y = prepare_features_and_labels(df)
y_mapped = map_labels_for_model(y, "xgboost")  # consistent label mapping

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.2, shuffle=False)

# Train models
xgb_model = train_xgboost_model(X_train, y_train)
lgb_model = train_lightgbm_model(X_train, y_train)
cat_model = train_catboost_model(X_train, y_train)

# Ensemble: Average predict_proba
xgb_probs = xgb_model.predict_proba(X)
lgb_probs = lgb_model.predict_proba(X)
cat_probs = cat_model.predict_proba(X)

# Average probabilities
ensemble_probs = (xgb_probs + lgb_probs + cat_probs) / 3.0
max_probs = ensemble_probs.max(axis=1)
pred_classes = np.argmax(ensemble_probs, axis=1)

# Map class to signals
raw_signals = reverse_label_mapping(pred_classes, "xgboost")

# Apply confidence threshold (like Phase 3.2)
confidence_threshold = 0.6
confident_signals = [
    signal if prob >= confidence_threshold else 0
    for signal, prob in zip(raw_signals, max_probs)
]

# Prepare backtest data
df['predicted_signal'] = np.nan
df.loc[X.index, 'predicted_signal'] = confident_signals

# Backtest
df_bt = simulate_trading(df)
metrics = calculate_performance_metrics(df_bt)
print_performance_report(metrics)
plot_equity_curves(df_bt)
