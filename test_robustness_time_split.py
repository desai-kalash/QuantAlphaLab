import pandas as pd
from src.data_loader import download_stock_data
from src.feature_engineering import add_technical_indicators
from src.labeling import add_forward_return_labels, clean_labeled_data
from src.model import prepare_features_and_labels, train_xgboost_model, evaluate_model
from src.backtester import run_full_backtest
from src.labeling import map_labels_for_model

# 1. Download full data (2020–2024)
print("📈 Downloading AAPL data from 2020–2024...")
df = download_stock_data("AAPL", "2020-01-01", "2024-12-31")
df = add_technical_indicators(df)
df = add_forward_return_labels(df)
df = clean_labeled_data(df)

# 2. Split into time-based train/test sets
train_df = df[df.index < "2023-01-01"]
test_df = df[df.index >= "2023-01-01"]

print(f"🧠 Training size: {train_df.shape}, 📊 Testing size: {test_df.shape}")

# 3. Prepare features & labels
X_train, y_train = prepare_features_and_labels(train_df)
X_test, y_test = prepare_features_and_labels(test_df)

# 4. Map labels for XGBoost
y_train_mapped = map_labels_for_model(y_train, "xgboost")
y_test_mapped = map_labels_for_model(y_test, "xgboost")

# 5. Train model
model = train_xgboost_model(X_train, y_train_mapped)

# 6. Evaluate on test period
print("\n📊 Evaluation on 2023–2024 Test Period:")
_ = evaluate_model(model, X_test, y_test_mapped)

# 7. Run backtest only on test set (2023–2024)
print("\n🚀 Running backtest on test period only...")
_ = run_full_backtest(model, test_df)
