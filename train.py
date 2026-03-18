import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

# Import Models
from sklearn.ensemble import StackingRegressor, VotingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

# Import data loader from src
from src.load_data import load_transformed_dataset

# GCP & MLflow Configuration
GCS_BUCKET = "gs://football-oracle-mlflow-artifacts/mlflow-data"
EXPERIMENT_NAME = "Football_Oracle_Final_v6"

def setup_mlflow():
    mlflow.set_tracking_uri("http://localhost:5001")
    experiments = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiments is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=GCS_BUCKET)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow configured with experiment: {EXPERIMENT_NAME} and GCS bucket: {GCS_BUCKET}")


def build_model():
    print("---Building Voting Ensemble Architecture ---")
    base_regressors = [
        ('rf', RandomForestRegressor(n_estimators=500, max_depth=8, random_state=42)),
        ('et', ExtraTreesRegressor(n_estimators=500, max_depth=8, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)),
        ('lgbm', LGBMRegressor(n_estimators=500, learning_rate=0.01, num_leaves=31, subsample=0.8, random_state=42, verbosity=-1))
    ]

    voter = VotingRegressor(estimators=base_regressors, n_jobs=-1)

    return MultiOutputRegressor(voter)

def get_res_label(h, a, m=0.215):
    """return result label (2: Home Win, 1: Draw, 0: Away Win)"""
    if (h - a) > m: return 2
    if (a - h) > m: return 0
    return 1


def train_model():
    print("--- 🚀 Starting Training Process (Separated Artifacts) ---")

    # 1. Load Data & Pipeline
    (X_train, X_test, y_train_res, y_train_sco, y_test_res, y_test_sco, pipeline) = load_transformed_dataset()

    # 2. Build Model
    model = build_model()

    with mlflow.start_run(run_name="Voting_V4_Separated"):
        # 3. Train Model
        print("--- Training Model ---")
        model.fit(X_train, y_train_sco)

        # 4. Evaluation
        y_pred_sco = model.predict(X_test)
        test_r2 = r2_score(y_test_sco, y_pred_sco)
        y_pred_res = [get_res_label(h, a) for h, a in y_pred_sco]
        acc = accuracy_score(y_test_res, y_pred_res)

        mlflow.log_metrics({"R2_Score": test_r2, "Accuracy": acc})
        print(f"R2: {test_r2:.4f}, Accuracy: {acc:.4f}")

        # 5. Save Artifacts to Local
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "football_stack_reg_model.pkl")
        pipeline_path = os.path.join(model_dir, "football_pipeline.pkl")

        joblib.dump(model, model_path)
        # ( pipeline  dump in load_transformed_dataset but we can also dump here to ensure it's the fitted version )
        joblib.dump(pipeline, pipeline_path)

        #  push it to GCS Bucket to sync with MLflow
        # send model Artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="football_oracle_model")

        # send pipeline Artifact
        mlflow.log_artifact(pipeline_path, artifact_path="pipeline")

        print(f"--- 💾 All Artifacts (Model & Pipeline) synced with GCS ---")

if __name__ == "__main__":
    setup_mlflow()
    train_model()
