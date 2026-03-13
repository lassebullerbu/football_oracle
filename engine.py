# engine.py
import pandas as pd
import numpy as np
import datetime
import os

data_dir = "./raw_data/"
clubs = pd.read_csv(os.path.join(data_dir, "clubs.csv"))

def find_club(search_string, clubs_df):
    search_string_lower = search_string.lower()
    matching_clubs = clubs_df[
        clubs_df['club_code'].str.lower().str.contains(search_string_lower, na=False) |
        clubs_df['name'].str.lower().str.contains(search_string_lower, na=False)
    ]
    return matching_clubs[['club_id', 'name']] if not matching_clubs.empty else None

def extract_club_features(df, clubs_df):
    club_features = {}

    df['date'] = pd.to_datetime(df['date'])
    # Ensure the dataframe is sorted by club and date for correct 'last game' and 'most recent' calculations
    df_sorted = df.sort_values(by=['club_id', 'date']).copy()

    for club_id in df_sorted['club_id'].unique():
        club_df = df_sorted[df_sorted['club_id'] == club_id]

        # Get club name
        club_name = clubs[clubs['club_id'] == club_id]['name'].iloc[0] if club_id in clubs['club_id'].values else None

        # Last game date
        last_game_date = club_df['date'].max()

        # Market value (average of last 3 games)
        # Get the last 3 market values, handling cases where there are fewer than 3 games
        market_values = club_df.tail(3)['own_market_value']
        avg_market_value = market_values.mean() if not market_values.empty else None

        # Most recent position
        most_recent_position = club_df['own_position'].iloc[-1] if not club_df.empty else None

        # Most recent streak_2
        most_recent_streak_2 = club_df['own_streak_2'].iloc[-1] if not club_df.empty else None

        # Most recent streak_5
        most_recent_streak_5 = club_df['own_streak_5'].iloc[-1] if not club_df.empty else None

        club_features[club_id] = {
            'club_name': club_name,
            'last_game': last_game_date,
            'market_value_avg_last_3': avg_market_value,
            'position_most_recent': most_recent_position,
            'streak_2_most_recent': most_recent_streak_2,
            'streak_5_most_recent': most_recent_streak_5
        }
    return club_features

def get_match_features(club_stats_dict, club_id_1, club_id_2, input_date):
    match_features = {}

    # Convert input_date to datetime if it's not already
    if not isinstance(input_date, (pd.Timestamp, datetime.datetime)): # Corrected to datetime.datetime
        input_date = pd.to_datetime(input_date)

    # Get features for club 1 (home team)
    club_1_data = club_stats_dict.get(club_id_1)
    if club_1_data is None:
        raise ValueError(f"Club ID {club_id_1} not found in club_stats_dict")

    # Get features for club 2 (away team)
    club_2_data = club_stats_dict.get(club_id_2)
    if club_2_data is None:
        raise ValueError(f"Club ID {club_id_2} not found in club_stats_dict")

    # Calculate rest days
    own_restday = (input_date - club_1_data['last_game']).days
    opponent_restday = (input_date - club_2_data['last_game']).days

    match_features['is_home'] = 1 # As requested, home team perspective
    match_features['own_restday'] = own_restday
    match_features['opponent_restday'] = opponent_restday
    match_features['own_market_value'] = club_1_data['market_value_avg_last_3']
    match_features['opponent_market_value'] = club_2_data['market_value_avg_last_3']
    match_features['own_position'] = club_1_data['position_most_recent']
    match_features['opponent_position'] = club_2_data['position_most_recent']
    match_features['own_streak_2'] = club_1_data['streak_2_most_recent']
    match_features['opponent_streak_2'] = club_2_data['streak_2_most_recent']
    match_features['own_streak_5'] = club_1_data['streak_5_most_recent']
    match_features['opponent_streak_5'] = club_2_data['streak_5_most_recent']

    return match_features

def predict_match_result_dict(home_name, away_name, match_date,
                              clubs_df, club_stats_dict,
                              pipeline, model):
    """
    fine ID -> สร้าง Features -> Transform -> Predict
    """

    # --- 1. find ID ---
    # --- 1. find ID and (Official Name) ---
    try:
        home_search = find_club(home_name, clubs_df)
        away_search = find_club(away_name, clubs_df)

        if home_search is None or away_search is None:
            return {"error": "ไม่พบข้อมูลทีมที่ระบุ"}

        # get ID and Realname DataFrame
        home_id = home_search.iloc[0]['club_id']
        home_official_name = home_search.iloc[0]['name'] #

        away_id = away_search.iloc[0]['club_id']
        away_official_name = away_search.iloc[0]['name'] #

    except Exception as e:
        return {"error": f"เกิดข้อผิดพลาด: {str(e)}"}

    # --- 2.  Match Features and Transform ---
    try:
        features = get_match_features(club_stats_dict, home_id, away_id, match_date)
        X_input = pd.DataFrame.from_dict([features])

        # use Pipeline for Feature Engineering และ Scaling
        X_input_transform = pipeline.transform(X_input)
    except Exception as e:
        return {"error": f"No expected error: {str(e)}"}

    # --- 3. predict ---
    stack_raw_scores = model.predict(X_input_transform)
    h_raw, a_raw = stack_raw_scores[0]

    # Decision Logic (Margin = 0.250)
    def get_res_label(h, a, m=0.250):
        if (h - a) > m: return "Home Win"
        if (a - h) > m: return "Away Win"
        return "Draw"

    result_label = get_res_label(h_raw, a_raw)

    # --- 4. Adjust scores (Logical Adjustment) ---
    h_int, a_int = int(np.round(h_raw)), int(np.round(a_raw))

    if result_label == "Home Win" and h_int <= a_int:
        h_int = a_int + 1
    elif result_label == "Away Win" and a_int <= h_int:
        a_int = h_int + 1
    elif result_label == "Draw" and h_int != a_int:
        avg_val = int(np.round((h_raw + a_raw) / 2))
        h_int, a_int = avg_val, avg_val

    # --- 5. Return Results ---
    return {
        "home_team": home_official_name,
        "away_team": away_official_name,
        "score_display": f"{h_int} - {a_int}",
        "home_score": h_int,
        "away_score": a_int,
        "raw_h": round(float(h_raw), 2),
        "raw_a": round(float(a_raw), 2),
        "result": result_label,
        "match_date": str(match_date),
    }
