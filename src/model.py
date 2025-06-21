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


def prepare_features_and_labels(data, feature_cols=None):
    """
    Prepare features and labels for model training
    
    Args:
        data (pd.DataFrame): Stock data with features and Signal column
        feature_cols (list): List of feature column names
        
    Returns:
        tuple: (X, y) features and labels
    """
    print("🔧 Preparing features and labels...")
    
    # Default feature columns
    if feature_cols is None:
        feature_cols = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower']
    
    # Safety check for available features
    available_features = [col for col in feature_cols if col in data.columns]
    missing_features = [col for col in feature_cols if col not in data.columns]
    
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
    
    print(f"✅ Using features: {available_features}")
    
    # Extract features and labels
    X = data[available_features]
    y = data['Signal']
    
    # Remove rows with any missing values
    X = X.dropna()
    y = y.loc[X.index]
    
    print(f"📊 Feature matrix shape: {X.shape}")
    print(f"📊 Label distribution:\n{y.value_counts().sort_index()}")
    
    return X, y


def train_xgboost_model(X_train, y_train, **kwargs):
    """
    Train XGBoost classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        **kwargs: Additional XGBoost parameters
        
    Returns:
        XGBClassifier: Trained model
    """
    print("🤖 Training XGBoost model...")
    
    # Default parameters
    default_params = {
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'random_state': 42
    }
    
    # Update with any provided parameters
    default_params.update(kwargs)
    
    # Create and train model
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
    from catboost import CatBoostClassifier

    print("🐱 Training CatBoost model...")

    default_params = {
        'loss_function': 'MultiClass',
        'iterations': 300,
        'learning_rate': 0.05,
        'depth': 6,
        'verbose': False,
        'random_seed': 42,
        'auto_class_weights': 'Balanced'  # ✅ This is the new line you add
    }

    default_params.update(kwargs)

    model = CatBoostClassifier(**default_params)
    model.fit(X_train, y_train)

    print("✅ CatBoost training complete")
    return model




def evaluate_model(model, X_test, y_test, show_details=True):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        show_details (bool): Whether to print detailed metrics
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    print("📊 Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    if show_details:
        print(f"\n📈 Accuracy: {accuracy:.4f}")
        print(f"\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"\n📊 Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    # Return metrics dictionary
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
    
    Args:
        model: Trained model
        filepath (str): Path to save the model
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        joblib.dump(model, filepath)
        print(f"✅ Model saved to {filepath}")
        
    except Exception as e:
        print(f"❌ Error saving model: {str(e)}")
        raise


def load_model(filepath="../models/xgb_model.pkl"):
    """
    Load trained model from file
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        model: Loaded model
    """
    try:
        model = joblib.load(filepath)
        print(f"✅ Model loaded from {filepath}")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise


def train_and_evaluate_pipeline(data, test_size=0.2, feature_cols=None, model_type="xgboost", **model_params):
    """
    Training pipeline with model selection.
    """
    print("🚀 Starting training pipeline...")

    X, y = prepare_features_and_labels(data, feature_cols)

    from .labeling import map_labels_for_model
    y_mapped = map_labels_for_model(y, "xgboost")  # still 0,1,2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_mapped, test_size=test_size, shuffle=False
    )

    print(f"🧠 Training set: {X_train.shape}")
    print(f"🧪 Testing set: {X_test.shape}")

    if model_type == "xgboost":
        model = train_xgboost_model(X_train, y_train, **model_params)
    elif model_type == "lightgbm":
        model = train_lightgbm_model(X_train, y_train, **model_params)
    elif model_type == "catboost":
        model = train_catboost_model(X_train, y_train, **model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    metrics = evaluate_model(model, X_test, y_test)
    print("🎯 Training pipeline completed!")

    return model




# Example usage (for testing)
if __name__ == "__main__":
    # This would be used to test the functions
    print("Model training module loaded successfully!")