import streamlit as st
import pandas as pd
import requests
import joblib
import os
import datetime
# input Logic from engine.py
from engine import extract_club_features, predict_match_result_dict

# --- 2. DATA INITIALIZATION ---
@st.cache_resource
def init_all_stats():
    # load processed data and clubs for stats dict creation
    proc_data = pd.read_csv("raw_data/processed_data.csv")
    c_df = pd.read_csv("raw_data/clubs.csv")
    # create stats_dict for API and Local Engine use
    stats_dict = extract_club_features(proc_data, c_df)
    return stats_dict, c_df

# call s_dict and clubs_df at the start so they are cached and ready for both Preview and Prediction
try:
    s_dict, clubs_df = init_all_stats()
    club_names = sorted(clubs_df['name'].unique())
except Exception as e:
    st.error(f"Error loading initial data: {e}")
    st.stop()

# --- 1. CONFIGURATION ---
# Swap Mode: 'API' or 'LOCAL'
PREDICTION_MODE = 'API'

# Call API_URL safely Error: StreamlitSecretNotFoundError
try:
    API_URL = st.secrets["API_URL"]
except (st.errors.StreamlitSecretNotFoundError, KeyError):
    API_URL = "http://localhost:8080/predict"

st.set_page_config(page_title="Football Oracle", layout="wide")

# --- 2. HELPERS ---
def get_logo_url(club_id):
    """get URL from Transfermarkt CDN by using club_id"""
    return f"https://tmssl.akamaized.net/images/wappen/head/{club_id}.png"

@st.cache_data
def load_ui_data():
    df = pd.read_csv("raw_data/clubs.csv")
    return df, sorted(df['name'].unique())

clubs_df, club_names = load_ui_data()

# --- 3. UI HEADER ---
st.title("⚽ Football Oracle Predictor")
#st.markdown(f"**Current Mode:** `{PREDICTION_MODE}` | **Backend:** `{"We can not show this HAHA!" if PREDICTION_MODE == 'API' else 'Local Engine'}`")
if PREDICTION_MODE == 'API':
    # ดึงแค่ชื่อโดเมนหลักมาโชว์ (เช่น football-service-ew.a.run.app)
    # ตัดส่วน https:// และ /predict ออก
    display_backend = API_URL.replace("https://", "").split("/")[0]
    backend_text = f"🌐 Cloud API"
else:
    backend_text = "💻 Local Engine"

st.markdown(f"**Current Mode:** `{PREDICTION_MODE}` | **Backend:** {backend_text}")
# --- 4. SELECTION AREA (With Logos) ---
st.write("### 🏟️ Match Selection")
col1, space, col2 = st.columns([10, 1, 10])

with col1:
    home_team = st.selectbox("🏠 Home Team", club_names, index=club_names.index("FC Bayern München") if "FC Bayern München" in club_names else 0)
    home_id = clubs_df[clubs_df['name'] == home_team]['club_id'].values[0]
    st.image(get_logo_url(home_id), width=120)

with col2:
    away_team = st.selectbox("🚌 Away Team", club_names, index=club_names.index("Borussia Dortmund") if "Borussia Dortmund" in club_names else 1)
    away_id = clubs_df[clubs_df['name'] == away_team]['club_id'].values[0]
    st.image(get_logo_url(away_id), width=120)

match_date = st.date_input("📅 Match Date", value=datetime.date(2026, 3, 15))

# --- 5. PREVIEW STATS (แสดงก่อน Predict) ---
st.write("### 📊 Team Comparison (Preview)")

# ดึงข้อมูลสถิติมาเตรียมไว้ก่อน
try:
    # คำนวณ Features เบื้องต้นเพื่อมาโชว์ Preview
    home_id = clubs_df[clubs_df['name'] == home_team]['club_id'].values[0]
    away_id = clubs_df[clubs_df['name'] == away_team]['club_id'].values[0]

    from engine import get_match_features
    # จำลองการดึง Features ออกมา
    preview_features = get_match_features(s_dict, home_id, away_id, match_date)

    # show preview features in a nice format (you can customize this part)
    prev_col1, prev_col2, prev_col3 = st.columns([3, 2, 3])

    with prev_col1:
        st.metric("Market Value (Avg)", f"€{preview_features['own_market_value']:,.0f}")
        st.metric("Rest Days", f"{preview_features['own_restday']} days")
        st.caption(f"Current 2 games Streak: {preview_features['own_streak_2']}")
        st.caption(f"Current 5 games Streak: {preview_features['own_streak_5']}")

    with prev_col2:
        st.markdown("<h3 style='text-align: center;'>VS</h3>", unsafe_allow_html=True)

    with prev_col3:
        st.metric("Market Value (Avg)", f"€{preview_features['opponent_market_value']:,.0f}")
        st.metric("Rest Days", f"{preview_features['opponent_restday']} days")
        st.caption(f"Current 2 games Streak: {preview_features['opponent_streak_2']}")
        st.caption(f"Current 5 games Streak: {preview_features['opponent_streak_5']}")

except Exception as e:
    st.info("Please select valid teams to preview stats.")

# --- 5. PREDICTION LOGIC ---
if st.button("🚀 Predict Result", use_container_width=True):
    result = None
    date_str = match_date.strftime("%Y-%m-%d")

    if PREDICTION_MODE == 'API':
        with st.spinner("📡 Requesting from Cloud API..."):
            try:
                payload = {"home_team": home_team, "away_team": away_team, "date": date_str}
                resp = requests.post(API_URL, json=payload, timeout=10)
                if resp.status_code == 200:
                    result = resp.json()
                else:
                    st.error(f"API Error: {resp.text}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")
    else:
        with st.spinner("💻 Calculating Locally..."):
            try:
                # Load Assets Local
                m = joblib.load("models/football_stack_reg_model.pkl")
                p = joblib.load("models/football_pipeline.pkl")
                proc_data = pd.read_csv("raw_data/processed_data.csv")
                s_dict = extract_club_features(proc_data, clubs_df)

                result = predict_match_result_dict(home_team, away_team, date_str,
                                                 clubs_df, s_dict, p, m)
            except Exception as e:
                st.error(f"Local Calculation Error: {e}")

    # --- 6. DISPLAY RESULTS (SCOREBOARD STYLE) ---
    if result and "error" not in result:
        st.balloons()
        st.markdown("---")

        # Display Scores in a Scoreboard Style
        res_col1, res_col2, res_col3 = st.columns([2, 1, 2])

        with res_col1:
            st.title(f" {result['home_score']} ")

        with res_col2:
            st.markdown("<h1 style='text-align: center; padding-top: 40px;'>-</h1>", unsafe_allow_html=True)

        with res_col3:
            st.title(f" {result['away_score']} ")

        st.markdown(f"<h2 style='text-align: center; color: #FF4B4B;'>RESULT: {result['result']}</h2>", unsafe_allow_html=True)

        # Technical Details (Show Raw Model Output)
        with st.expander("🔍 Technical Details (Model Raw Output and Raw Input)"):
            st.json(result)
