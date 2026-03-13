import os
import joblib
import pandas as pd
import numpy as np

# Import Models
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

# Import data loader from src
from src.load_data import load_transformed_dataset

def build_model():
    """
    build_model(Architecture)

    """
    print("--- Building Model Architecture ---")

    # Base Regressors
    base_regressors = [
        ('rf', RandomForestRegressor(n_estimators=400, max_depth=7, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=400, learning_rate=0.01, max_depth=7, random_state=42)),
        ('lgbm', LGBMRegressor(n_estimators=400, learning_rate=0.01, num_leaves=31, random_state=42, verbosity=-1))
    ]

    # create Stacking Regressor MultiOutputRegressor
    stack_reg = StackingRegressor(
        estimators=base_regressors,
        final_estimator=Ridge(alpha=1.0),
        cv=3,
        n_jobs=-1,
        passthrough=True
    )

    model = MultiOutputRegressor(stack_reg)
    return model

def train_model(model):
    """
    Run model and save to models/
    """
    print("--- 🚀 Starting Training Process ---")

    # 1. load src/load_data.py
    (X_train, X_test,
     y_train_res, y_train_sco,
     y_test_res, y_test_sco,
     pipeline) = load_transformed_dataset()

    # 2. Train model
    print("--- 🧠 Training Stacking Regressor (Goals Prediction) ---")
    model.fit(X_train, y_train_sco)
    print("✅ Training Complete!")

    # 3. evaluate model
    train_score = model.score(X_train, y_train_sco)
    test_score = model.score(X_test, y_test_sco)
    print(f"\n📊 Evaluation Results:")
    print(f"   - Train R² Score: {train_score:.4f}")
    print(f"   - Test R² Score: {test_score:.4f}")

    # 4. save model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "bundesliga_stack_reg_model.pkl")

    joblib.dump(model, model_path)
    print(f"--- 💾 Model saved successfully at: {model_path} ---")

    return model
