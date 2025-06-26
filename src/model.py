"""
Model Training Module
Trains and evaluates machine learning models for trading signals
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

from src.utils import save_model_outputs

from src.utils import save_model_outputs  # ✅ NEW: to cache model outputs


def prepare_features_and_labels(data, feature_cols=None):
    """
    Prepare features and labels for model training
    """
    print("🔧 Preparing features and labels...")

    if feature_cols is None:
        feature_cols = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower']

    available_features = [col for col in feature_cols if col in data.columns]
    missing_features = [col for col in feature_cols if col not in data.columns]

    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")

    print(f"✅ Using features: {available_features}")

    X = data[available_features].dropna()
    y = data['Signal'].loc[X.index]

    print(f"📊 Feature matrix shape: {X.shape}")
    print(f"📊 Label distribution:\n{y.value_counts().sort_index()}")

    return X, y


def train_xgboost_model(X_train, y_train, **kwargs):
    """
    Train XGBoost classifier
    """
    print("🤖 Training XGBoost model...")

    default_params = {
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'random_state': 42
    }
    default_params.update(kwargs)

    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train)

    print("✅ Model training completed")
    return model


def train_lightgbm_model(X_train, y_train, **kwargs):
    """
    Train a LightGBM classifier.
    """
    import lightgbm as lgb
    print("⚡ Training LightGBM model...")

    default_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'random_state': 42,
        'verbose': -1
    }
    default_params.update(kwargs)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_train, y_train)

    print("✅ LightGBM training complete")
    return model


def train_catboost_model(X_train, y_train, **kwargs):
    """
    Train a CatBoost classifier.
    """
    from catboost import CatBoostClassifier
    print("🐱 Training CatBoost model...")

    default_params = {
        'loss_function': 'MultiClass',
        'iterations': 300,
        'learning_rate': 0.05,
        'depth': 6,
        'verbose': False,
        'random_seed': 42,
        'auto_class_weights': 'Balanced'
    }
    default_params.update(kwargs)

    model = CatBoostClassifier(**default_params)
    model.fit(X_train, y_train)

    print("✅ CatBoost training complete")
    return model


def evaluate_model(model, X_test, y_test, show_details=True):
    """
    Evaluate model performance
    """
    print("📊 Evaluating model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if show_details:
        print(f"\n📈 Accuracy: {accuracy:.4f}")
        print(f"\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"\n📊 Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    metrics = {
        'accuracy': accuracy,
        'predictions': y_pred,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return metrics


def save_model(model, filepath="../models/xgb_model.pkl"):
    """
    Save trained model to file
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        print(f"✅ Model saved to {filepath}")
    except Exception as e:
        print(f"❌ Error saving model: {str(e)}")
        raise


def load_model(filepath="../models/xgb_model.pkl"):
    """
    Load trained model from file
    """
    try:
        model = joblib.load(filepath)
        print(f"✅ Model loaded from {filepath}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise


def train_and_evaluate_pipeline(data, model_type="xgboost"):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb

    feature_cols = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower']
    
    # Add price column to retain for backtesting
    X = data[feature_cols].copy()
    X["close"] = data["close"] if "close" in data.columns else data.iloc[:, 0]

    y = data["Signal"].map({-1: 0, 0: 1, 1: 2})  # Encode labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if model_type == "xgboost":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    elif model_type == "lightgbm":
        model = lgb.LGBMClassifier()
    elif model_type == "catboost":
        model = cb.CatBoostClassifier(verbose=0)
    else:
        raise ValueError("Unsupported model type.")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    save_model_outputs(model, y_test, y_pred, X_test)  # ✅ Save for frontend access

    return model, X_test, y_test, y_pred



# For manual testing
if __name__ == "__main__":
    print("Model training module loaded successfully!")
