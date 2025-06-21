from src.data_loader import download_stock_data
from src.feature_engineering import add_technical_indicators
from src.labeling import add_forward_return_labels, clean_labeled_data
from src.model import train_and_evaluate_pipeline, prepare_features_and_labels
from src.explainability import plot_global_shap, plot_local_shap

# Download and prepare data
df = download_stock_data("AAPL", "2020-01-01", "2024-12-31")
df = add_technical_indicators(df)
df = add_forward_return_labels(df)
df = clean_labeled_data(df)

# Train model
model = train_and_evaluate_pipeline(df)

# Get features (X)
X, _ = prepare_features_and_labels(df)

# Global explanation
plot_global_shap(model, X)

# Local explanation (for predicted class)
plot_local_shap(model, X, index=0)
