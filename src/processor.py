import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer

# 1. Feature Engineering (Diffs & Logs)
def internal_feature_engineering(X):
    X_out = X.copy()
    #  (Diffs)
    X_out['pos_diff'] = X_out['opponent_position'] - X_out['own_position']
    X_out['streak2_diff'] = X_out['own_streak_2'] - X_out['opponent_streak_2']
    X_out['streak5_diff'] = X_out['own_streak_5'] - X_out['opponent_streak_5']
    X_out['rest_diff'] = X_out['own_restday'] - X_out['opponent_restday']

    #  Log Transformation for Market Value
    X_out['log_own_mv'] = np.log1p(X_out['own_market_value'].astype(float))
    X_out['log_opp_mv'] = np.log1p(X_out['opponent_market_value'].astype(float))
    X_out['log_mv_ratio'] = X_out['log_own_mv'] - X_out['log_opp_mv']

    return X_out

def create_datasets(processed_data):
    """
    Create feature matrix X and target vectors y_result (match outcome)
    and y_score (goals scored) from the processed dataset."""
# 1. sort by date
    processed_data = processed_data.sort_values('date').reset_index(drop=True)

    # 2. features
    base_features = [
        'is_home', 'own_restday', 'opponent_restday',
        'own_market_value', 'opponent_market_value',
        'own_position', 'opponent_position',
        'own_streak_2', 'opponent_streak_2',
        'own_streak_5', 'opponent_streak_5'
    ]

    # 3. seperate X and Targets
    X = processed_data[base_features]
    y_result = processed_data['target_result']
    y_score = processed_data[['own_goals', 'opponent_goals']]
    return X, y_result, y_score

def split_data(X, y_result, y_score):
    """
    Split the dataset into training and testing sets
    based on chronological order (80% train, 20% test)."""
    # 4. cut 80%
    split_idx = int(len(X) * 0.8)

    # 5. seperate data
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_result, y_test_result = y_result.iloc[:split_idx], y_result.iloc[split_idx:]
    y_train_score, y_test_score = y_score.iloc[:split_idx], y_score.iloc[split_idx:]

    print(f"✅ Data split complete: Train {len(X_train)}, Test {len(X_test)}")

    return X_train, X_test, y_train_result, y_train_score, y_test_result, y_test_score

def create_preprocessing_pipeline():
    """
    Create a preprocessing pipeline that includes:
    1. Feature engineering (differences and log transformations)
    2. Scaling (StandardScaler for most features, MinMaxScaler for positions)
    3. Combine all steps into a single Pipeline object.
    """

    # 2.  Scaling (StandardScaler & MinMaxScaler)
    preprocessor = ColumnTransformer(
    transformers=[
        # [GROUP 1] StandardScaler:
        ('std_scale', StandardScaler(), [
            'log_own_mv', 'log_opp_mv', 'log_mv_ratio', # Market
            'own_restday', 'opponent_restday', 'rest_diff', # Rest
            'own_streak_2', 'opponent_streak_2', 'streak2_diff', # Streak 2
            'own_streak_5', 'opponent_streak_5', 'streak5_diff'  # Streak 5
        ]),

        # [GROUP 2] MinMaxScaler:
        ('minmax_scale', MinMaxScaler(), [
            'own_position', 'opponent_position', 'pos_diff'
        ]),

        # Binary:
        ('bin', 'passthrough', ['is_home'])
    ]
)

    # 3. combine Pipeline
    pipeline = Pipeline([
        ('feat_eng', FunctionTransformer(internal_feature_engineering)),
        ('scaler', preprocessor)
    ])

    pipeline.set_output(transform="pandas")

    return pipeline

def fit_transform_pipeline(pipeline, X_train, X_test):
    """
    Fit the preprocessing pipeline on the training data and transform both
    training and testing datasets. Returns the transformed datasets as pandas DataFrames."""
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    print("✅ Preprocessing pipeline fit and transformed successfully.")
    return X_train_transformed, X_test_transformed
