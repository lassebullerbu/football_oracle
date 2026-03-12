import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# setup Streamlit page configuration
st.set_page_config(page_title="Football Oracle: Bundesliga Predictor", layout="wide")

# --- 1. load Assets (Model & Pipeline) ---
@st.cache_resource
def load_assets():
    # use Path to load model and pipeline
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "models", "bundesliga_stack_reg_model.pkl")
    pipeline_path = os.path.join(base_path, "models", "bundesliga_pipeline.pkl")

    if not os.path.exists(model_path) or not os.path.exists(pipeline_path):
        st.error("❌ Not found model or Pipeline please run main.py to train the model first!")
        return None, None

    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    return model, pipeline

model, pipeline = load_assets()
