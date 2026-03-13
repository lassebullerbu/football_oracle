import pandas as pd
import numpy as np
import os
import joblib
from .processor import (
    create_datasets,
    split_data,
    create_preprocessing_pipeline,
    fit_transform_pipeline
)

def load_data():
    data_dir = "./raw_data/"
    print(f"--- 1. Loading Data from {data_dir} ---")

    # ตรวจสอบว่ามีโฟลเดอร์หรือไม่
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Folder '{data_dir}' not found")

    # โหลดไฟล์ CSV จากโฟลเดอร์ rawdata
    try:
        games = pd.read_csv(os.path.join(data_dir, "games.csv"))
        club_games = pd.read_csv(os.path.join(data_dir, "club_games.csv"))
        appearances = pd.read_csv(os.path.join(data_dir, "appearances.csv"))
        player_valuations = pd.read_csv(os.path.join(data_dir, "player_valuations.csv"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"ไฟล์ไม่ครบใน {data_dir}: {e}")

    games = games[games['competition_id'] == 'L1'].copy()
    games['date'] = pd.to_datetime(games['date'])

    club_games = club_games.merge(games[['game_id', 'date']], on='game_id', how='inner')
    club_games['date'] = pd.to_datetime(club_games['date'])

    appearances['date'] = pd.to_datetime(appearances['date'])
    player_valuations['date_val'] = pd.to_datetime(player_valuations['date'])

    print("--- 2. Calculating Market Value (Latest Only) ---")
    combined = appearances[['game_id', 'player_id', 'player_club_id', 'date']].merge(
        player_valuations[['player_id', 'date_val', 'market_value_in_eur']],
        on='player_id', how='left'
    )
    combined = combined[combined['date'] >= combined['date_val']]
    combined = combined.sort_values(['game_id', 'player_id', 'date_val'], ascending=[True, True, False])
    combined = combined.drop_duplicates(subset=['game_id', 'player_id'])

    club_mv = combined.groupby(['game_id', 'player_club_id'])['market_value_in_eur'].sum().reset_index()
    club_mv.columns = ['game_id', 'club_id', 'agg_market_value']

    print("--- 3. Engineering Club-Level Features ---")
    club_games = club_games.sort_values(['club_id', 'date'])

    # คำนวณ pts จากข้อมูลที่มีในมือ
    club_games['pts'] = 0
    club_games.loc[club_games['own_goals'] > club_games['opponent_goals'], 'pts'] = 3
    club_games.loc[club_games['own_goals'] == club_games['opponent_goals'], 'pts'] = 1

    group = club_games.groupby('club_id')['pts']
    club_games['own_streak_2'] = group.transform(lambda x: x.rolling(2, min_periods=1).sum().shift(1)).fillna(0)
    club_games['own_streak_5'] = group.transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1)).fillna(0)
    club_games['own_restday'] = (club_games['date'] - club_games.groupby('club_id')['date'].shift(1)).dt.days.fillna(7)

    club_games = club_games.merge(club_mv, on=['game_id', 'club_id'], how='left')
    club_games['own_market_value'] = club_games.groupby('club_id')['agg_market_value'].ffill().fillna(1e6)
    club_games['is_home'] = club_games['hosting'].map({'Home': 1, 'Away': 0})

    print("--- 4. Merging Opponent Perspective ---")
    # เลือกฟีเจอร์ฝั่งตรงข้าม
    # เราเลือก opponent_goals มาด้วยเพื่อให้มั่นใจว่ามันจะติดไปใน final_df
    opp_features = club_games[[
        'game_id', 'club_id', 'own_restday', 'own_market_value',
        'own_position', 'own_streak_2', 'own_streak_5', 'own_goals'
    ]].copy()

    # เปลี่ยนชื่อเพื่อให้กลายเป็นมุมมอง "คู่แข่ง" ของเรา
    opp_features.columns = [
        'game_id', 'opponent_id', 'opponent_restday', 'opponent_market_value',
        'opponent_position', 'opponent_streak_2', 'opponent_streak_5', 'opponent_goals'
    ]

    # Merge: กรองคอลัมน์ที่อาจจะซ้ำใน club_games ออกก่อน merge เพื่อไม่ให้เกิด _x, _y
    cols_to_keep = [
        'game_id', 'date', 'club_id', 'opponent_id', 'is_home',
        'own_restday', 'own_market_value', 'own_position',
        'own_streak_2', 'own_streak_5', 'own_goals'
    ]

    final_df = club_games[cols_to_keep].merge(opp_features, on=['game_id', 'opponent_id'], how='inner')

    print("--- 5. Generating Targets and Cleaning ---")
    final_df['target_result'] = 1
    final_df.loc[final_df['own_goals'] > final_df['opponent_goals'], 'target_result'] = 2
    final_df.loc[final_df['own_goals'] < final_df['opponent_goals'], 'target_result'] = 0

    result_columns = [
        'game_id', 'date', 'is_home',
        'own_restday', 'opponent_restday',
        'own_market_value', 'opponent_market_value',
        'own_position', 'opponent_position',
        'own_streak_2', 'opponent_streak_2',
        'own_streak_5', 'opponent_streak_5',
        'own_goals', 'opponent_goals', 'target_result'
    ]

    final_df = final_df[result_columns].dropna()
    print(f"✅ Dataset Ready! Total rows: {len(final_df)}")

    processed_data = os.path.join(data_dir, "processed_data.csv")
    final_df.to_csv(processed_data, index=False)

def load_transformed_dataset():
    """
    1. load selected_data.csv
    2. split Features/Targets
    3. split Train/Test
    4. Do Preprocessing Pipeline
    """
    # 1. กำหนด Path ของไฟล์ selected_data.csv
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "..", "raw_data", "processed_data.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ {file_path} กรุณารันฟังก์ชัน load_data() ก่อน")

    print(f"--- 🔄 Loading and Preprocessing Dataset from {file_path} ---")

    # 2. load dataset
    df = pd.read_csv(file_path, parse_dates=['date'])

    # 3. สร้าง Dataset (X, y_result, y_score)
    X, y_result, y_score = create_datasets(df)

    # 4. (Train 80% / Test 20%)
    X_train, X_test, y_train_res, y_train_sco, y_test_res, y_test_sco = split_data(X, y_result, y_score)

    # 5. Run Preprocessing Pipeline
    pipeline = create_preprocessing_pipeline()
    X_train_final, X_test_final = fit_transform_pipeline(pipeline, X_train, X_test)

    # --- Save Pipeline ---
    model_dir = "./models/"
    os.makedirs(model_dir, exist_ok=True)
    pipeline_path = os.path.join(model_dir, "football_pipeline.pkl")
    joblib.dump(pipeline, pipeline_path)
    print(f"✅ Pipeline (fitted) saved successfully at: {pipeline_path}")

    # return transformed data with targets and pipeline for future use (like Streamlit)
    return X_train_final, X_test_final, y_train_res, y_train_sco, y_test_res, y_test_sco, pipeline
