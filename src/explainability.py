"""
Explainability Module using SHAP
Generates global and local model interpretation plots
"""

import shap
import pandas as pd
import matplotlib.pyplot as plt


def plot_global_shap(model, X):
    """
    Plot global SHAP summary (feature importance across all classes)

    Args:
        model: Trained XGBoost model
        X (pd.DataFrame): Feature matrix
    """
    print("📊 Generating global SHAP summary...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X, plot_type="bar")  # global feature importance
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_local_shap(model, X, index=0, class_id=None):
    """
    Plot SHAP explanation for a single prediction (multi-class XGBoost-safe)

    Args:
        model: Trained model
        X (pd.DataFrame): Feature matrix
        index (int): Row index to explain
        class_id (int): Optional - class index to explain (0=Sell, 1=Hold, 2=Buy)
    """
    print(f"🔍 Explaining prediction at row {index}...")

    # Generate SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Select class automatically if not given
    if class_id is None:
        predicted_class = model.predict(X.iloc[[index]])[0]
        class_id = predicted_class
        print(f"📌 Automatically selected predicted class: {class_id}")

    # Create single-class SHAP explanation
    single_explanation = shap.Explanation(
        values=shap_values.values[index, class_id],
        base_values=shap_values.base_values[index, class_id],
        data=shap_values.data[index],
        feature_names=shap_values.feature_names
    )

    # Plot waterfall chart
    shap.plots.waterfall(single_explanation, max_display=7)
    plt.tight_layout()
    plt.show()
    plt.close()
