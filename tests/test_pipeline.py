import os
import joblib
import pandas as pd
import pytest

def test_model_files_exist():
    """检查训练后的模型文件是否存在 exam whether trained model exists"""
    model_path = "models/bundesliga_stack_reg_model.pkl"
    assert os.path.exists(model_path), f"model file  {model_path} not found！"

def test_processed_data_exists():
    """check Step 1 whether preprocessed data exists"""
    data_path = "raw_data/selected_data.csv"
    assert os.path.exists(data_path), f"processed_data {data_path} missing！"
