"""
Time-Series Cross-Validation with Backtesting
Logs fold-by-fold performance to CSV
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data_loader import download_stock_data
from src.feature_engineering import add_technical_indicators
from src.labeling import (
    add_forward_return_labels,
    clean_labeled_data,
    map_labels_for_model,
    reverse_label_mapping,
)
from src.model import prepare_features_and_labels
from src.backtester import simulate_trading, calculate_performance_metrics

# ✅ 1. Create results directory
os.makedirs("results", exist_ok=True)

# ✅ 2. Download and prepare data
df = download_stock_data("AAPL", "2020-01-01", "2024-12-31")
df = add_technical_indicators(df)
df = add_forward_return_labels(df)
df = clean_labeled_data(df)

# ✅ 3. Prepare features and labels
X, y = prepare_features_and_labels(df)
y_mapped = map_labels_for_model(y, model_type="xgboost")

# ✅ 4. Time-Series Cross Validation
tscv = TimeSeriesSplit(n_splits=5)
fold_results = []

print("\n⏳ Starting TimeSeriesSplit (5 folds)...\n")

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_mapped.iloc[train_idx], y_mapped.iloc[test_idx]

    print(f"\n📂 Fold {fold}: Train = {X_train.shape}, Test = {X_test.shape}")

    # ✅ 5. Train XGBoost model
    model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    model.fit(X_train, y_train)

    # ✅ 6. Classification Metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("🔍 Classification Metrics:")
    print(f"\n📈 Accuracy: {accuracy:.4f}")
    print(f"\n📊 Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"\n📊 Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # ✅ 7. Generate trading signals and simulate trading
    df_test = df.iloc[test_idx].copy()
    df_test["predicted_signal"] = reverse_label_mapping(y_pred, "xgboost")
    df_backtest = simulate_trading(df_test)
    perf = calculate_performance_metrics(df_backtest)

    # ✅ 8. Print Financial Performance
    print("📈 Financial Performance:")
    print(f"   Total Return:  {perf['total_return_pct']:.2f}%")
    print(f"   Sharpe Ratio:  {perf['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown:  {perf['max_drawdown_pct']:.2f}%")

    # ✅ 9. Store metrics for CSV
    fold_results.append({
        "fold": fold,
        "accuracy": accuracy,
        "total_return_pct": perf["total_return_pct"],
        "sharpe_ratio": perf["sharpe_ratio"],
        "max_drawdown_pct": perf["max_drawdown_pct"],
        "final_capital": perf["final_capital"],
        "total_trades": perf["total_trades"],
        "buy_signals": perf["buy_signals"],
        "sell_signals": perf["sell_signals"]
    })

# ✅ 10. Save all fold results to CSV
results_df = pd.DataFrame(fold_results)
results_path = "results/fold_performance.csv"
results_df.to_csv(results_path, index=False)

print(f"\n✅ All fold results saved to {results_path}")
print(results_df)
