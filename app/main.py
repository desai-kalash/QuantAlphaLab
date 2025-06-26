# app/main.py

import streamlit as st
from PIL import Image
import os

st.set_page_config(
    page_title="QuantAlphaLab",
    page_icon="📈",
    layout="wide"
)

# Optional logo
logo_path = "app/assets/logo.png"
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, width=150)

# App Title
st.title("📊 QuantAlphaLab – Alpha Research Platform")

# Introduction
st.markdown("""
Welcome to **QuantAlphaLab**, an interactive platform designed for quantitative analysts and ML researchers to:
- Train and evaluate ML-based alpha signals
- Perform strategy backtesting and robustness testing
- Construct portfolios with risk constraints
- Export research-ready reports

Use the sidebar or navigate to a module below to begin your research.
""")

# Navigation Links
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🚀 Start Model Training"):
        st.switch_page("pages/1_📊_Model_Training.py")
with col2:
    if st.button("📈 View Signals & Predictions"):
        st.switch_page("pages/2_🧠_Signal_Generation.py")
with col3:
    if st.button("📊 Backtest Strategies"):
        st.switch_page("pages/3_📈_Backtest_Evaluation.py")

# Optional Footer
st.markdown("""---""")
st.markdown("💡 Tip: All modules are available in the sidebar →")

