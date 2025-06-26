import streamlit as st
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Add src/ to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data_loader import download_stock_data
from src.feature_engineering import add_technical_indicators
from src.labeling import add_forward_return_labels, clean_labeled_data
from src.model import train_and_evaluate_pipeline

# Page config
st.set_page_config(page_title="Model Training", layout="wide")
st.title("📊 Model Training")

st.markdown("""
This module allows you to:

- 📈 Download or upload stock data  
- 🧠 Add technical indicators  
- 🔖 Label forward returns  
- 🤖 Train and evaluate ML models
""")

# Session state init
if 'data' not in st.session_state:
    st.session_state.data = None

# 1️⃣ Data Source
st.subheader("📂 Step 1: Choose Your Data Source")
data_source = st.radio("Select source:", ["🗕 Upload CSV", "🌐 Download from Yahoo Finance"], horizontal=True)

if data_source == "🗕 Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        try:
            st.session_state.data = pd.read_csv(uploaded_file, index_col=0)
            st.success("✅ CSV data loaded successfully.")
            st.dataframe(st.session_state.data.head())
        except Exception as e:
            st.error(f"❌ Failed to load CSV: {e}")

elif data_source == "🌐 Download from Yahoo Finance":
    ticker = st.text_input("Enter ticker (e.g., AAPL):", value="AAPL")
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start date:", pd.to_datetime("2022-01-01"))
    with col2:
        end = st.date_input("End date:", pd.to_datetime("2023-12-31"))

    if st.button("Download Data"):
        with st.spinner("🗕 Downloading stock data..."):
            st.session_state.data = download_stock_data(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if st.session_state.data is not None and not st.session_state.data.empty:
            st.success(f"✅ {len(st.session_state.data)} rows downloaded for {ticker}")
            st.dataframe(st.session_state.data.head())
        else:
            st.error("❌ Failed to download data. Check ticker or date range.")

# If data exists
if st.session_state.data is not None:
    st.subheader("📄 Preview Existing Data")
    st.dataframe(st.session_state.data.head())

    # 2️⃣ Feature Engineering
    st.subheader("🧠 Step 2: Add Technical Indicators")
    with st.spinner("Calculating indicators..."):
        data = add_technical_indicators(st.session_state.data.copy())
    st.session_state.data = data
    st.success("✅ Technical indicators added.")
    st.dataframe(data.tail(3))

    # 3️⃣ Labeling
    st.subheader("🔖 Step 3: Add Labels for Future Returns")
    col1, col2, col3 = st.columns(3)
    with col1:
        lookahead = st.slider("Lookahead Days", 1, 10, 3, key="lookahead")
    with col2:
        buy_threshold = st.slider("Buy Threshold", 0.0, 0.1, 0.02, key="buy_threshold")
    with col3:
        sell_threshold = st.slider("Sell Threshold", -0.1, 0.0, -0.02, key="sell_threshold")

    with st.spinner("Generating labels..."):
        data = add_forward_return_labels(data.copy(), lookahead, buy_threshold, sell_threshold)
    st.session_state.data = data
    st.success("✅ Labels generated.")
    st.dataframe(data.tail(3))

    st.caption("🧪 Debug: Future Return & Signal Preview")
    st.write(data[["future_return", "Signal"]].tail(10))

    # 4️⃣ Clean Data
    st.subheader("🧹 Step 4: Clean & Prepare Data")
    clean_data = clean_labeled_data(data.copy())
    st.write(f"📿 Final dataset shape: {clean_data.shape}")
    st.dataframe(clean_data.tail(3))

    # 5️⃣ Model Training
    st.subheader("🤖 Step 5: Train Your Model")
    model_choice = st.selectbox("Choose model type:", ["xgboost", "lightgbm", "catboost"])

    if st.button("🚀 Train Model"):
        with st.spinner(f"Training {model_choice} model..."):
            model, y_test, y_pred, X_test = train_and_evaluate_pipeline(clean_data, model_type=model_choice)

        st.success("🎯 Model training and evaluation complete.")

        # Convert to 1D arrays if needed
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.iloc[:, 0]
        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.iloc[:, 0]

        y_test = pd.Series(y_test).astype(int).to_numpy()
        y_pred = pd.Series(y_pred).astype(int).to_numpy()

        # 6️⃣ Classification Report
        st.subheader("📈 Classification Report")
        try:
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose())
        except Exception as e:
            st.error(f"❌ Error generating classification report: {e}")

        # 7️⃣ Confusion Matrix
        st.subheader("🧹 Confusion Matrix")
        try:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error generating confusion matrix: {e}")

        # 8️⃣ Feature Importance
        st.subheader("🔍 Feature Importance")
        try:
            importances = model.feature_importances_
            feat_names = model.feature_name_ if hasattr(model, 'feature_name_') else X_test.columns
            feat_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
            st.bar_chart(feat_df.set_index("Feature"))
        except Exception:
            st.warning("⚠️ Feature importance not available for this model.")

        # 9️⃣ Download Trained Model (Better Format)
        st.subheader("📂 Save Model")
        try:
            if model_choice == "lightgbm":
                model.booster_.save_model("trained_model.txt")
                with open("trained_model.txt", "rb") as f:
                    st.download_button("📅 Download LightGBM Model (.txt)", f, file_name="trained_model.txt")

            elif model_choice == "xgboost":
                model.save_model("trained_model.json")
                with open("trained_model.json", "rb") as f:
                    st.download_button("📅 Download XGBoost Model (.json)", f, file_name="trained_model.json")

            elif model_choice == "catboost":
                model.save_model("trained_model.cbm")
                with open("trained_model.cbm", "rb") as f:
                    st.download_button("📅 Download CatBoost Model (.cbm)", f, file_name="trained_model.cbm")
        except Exception as e:
            st.error(f"❌ Error saving model: {e}")
