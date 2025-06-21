import pandas as pd
from src.model import train_and_evaluate_pipeline
from src.labeling import clean_labeled_data, add_forward_return_labels
from src.feature_engineering import add_technical_indicators
from src.data_loader import download_stock_data

def test_train_pipeline_runs():
    df = download_stock_data("AAPL", "2020-01-01", "2021-01-01")
    df = add_technical_indicators(df)
    df = add_forward_return_labels(df)
    df = clean_labeled_data(df)
    
    model = train_and_evaluate_pipeline(df)
    
    assert model is not None
