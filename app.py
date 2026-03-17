import streamlit as st
import pandas as pd
import requests
import joblib
import os
import datetime
import time
# input Logic from engine.py
from engine import extract_club_features, predict_match_result_dict

# DATA INITIALIZATION
@st.cache_resource
def init_all_stats():
    # load processed data and clubs for stats dict creation
    proc_data = pd.read_csv("raw_data/processed_data.csv")
    c_df = pd.read_csv("raw_data/clubs.csv")
    comp_df = pd.read_csv("raw_data/competitions.csv")
    # create stats_dict for API and Local Engine use
    stats_dict = extract_club_features(proc_data, c_df)
    return stats_dict, c_df, comp_df, proc_data

# call s_dict and clubs_df at the start so they are cached and ready for both Preview and Prediction
try:
    s_dict, clubs_df, comp_df, proc_data = init_all_stats()
    league_names = sorted(comp_df[comp_df['type'] == 'domestic_league']['name'].unique())
    club_names = sorted(clubs_df['name'].unique())
except Exception as e:
    st.error(f"Error loading initial data: {e}")
    st.stop()

# CONFIGURATION
# Swap Mode: 'API' or 'LOCAL'
PREDICTION_MODE = 'API'

# Call API_URL safely Error: StreamlitSecretNotFoundError
try:
    API_URL = st.secrets["API_URL"]
except (st.errors.StreamlitSecretNotFoundError, KeyError):
    API_URL = "http://localhost:8080/predict"

st.set_page_config(page_title="Football Oracle", layout="wide")

#  HELPERS
def get_logo_url(club_id):
    """get URL from Transfermarkt CDN by using club_id"""
    return f"https://tmssl.akamaized.net/images/wappen/head/{club_id}.png"

@st.cache_data
def load_ui_data():
    df = pd.read_csv("raw_data/clubs.csv")
    return df, sorted(df['name'].unique())

clubs_df, club_names = load_ui_data()

# UI HEADER
#st.image("https://img.freepik.com/premium-vector/oracle-symbol-ethnic-protection-sign-spiritual-eye_543062-8378.jpg", width=80)
st.title("⚽ Football Oracle 🧙🏻‍♀️🔮🪄")

if PREDICTION_MODE == 'API':
    display_backend = API_URL.replace("https://", "").split("/")[0]
    backend_text = f"🌐 Cloud API"
else:
    backend_text = "💻 Local Engine"

st.markdown(f"**Current Mode:** `{PREDICTION_MODE}` | **Backend:** {backend_text}")
# SELECTION AREA (With Logos)

# tabs
tab1, tab2, tab3 = st.tabs(["🤝 Intro", "📊 Statistics", "🪄 Prediction"])

with tab1:
    st.header("Team")
    col11, space, col12, space, col13 = st.columns([10, 1, 10, 1, 10])
    with col11:
        st.subheader("Lasse")
        st.image("https://d26jy9fbi4q9wx.cloudfront.net/rails/active_storage/representations/proxy/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNmFRQkE9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--19523149968279a8f8ee4baa07e1a49c7a32e113/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaDdCem9MWm05eWJXRjBTU0lJYW5CbkJqb0dSVlE2RTNKbGMybDZaVjkwYjE5bWFXeHNXd2hwQWNocEFjaDdCam9KWTNKdmNEb09ZWFIwWlc1MGFXOXUiLCJleHAiOm51bGwsInB1ciI6InZhcmlhdGlvbiJ9fQ==--b67d9ded4d28d0969fbb98b4c21b79257705a99a/IMG_4827.jpg", width=80)
        st.markdown("*The business guy*")
    with col12:
        st.subheader("Cong")
        st.image("https://d26jy9fbi4q9wx.cloudfront.net/rails/active_storage/representations/proxy/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNDgyQlE9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--399c25ed125b8ef6db808dec4087c32e487eb0f0/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaDdCem9MWm05eWJXRjBTU0lJYW5CbkJqb0dSVlE2RTNKbGMybDZaVjkwYjE5bWFXeHNXd2hwQWNocEFjaDdCam9KWTNKdmNEb09ZWFIwWlc1MGFXOXUiLCJleHAiOm51bGwsInB1ciI6InZhcmlhdGlvbiJ9fQ==--b67d9ded4d28d0969fbb98b4c21b79257705a99a/IMG_4766.jpg", width=80)
        st.markdown("*The Chinese coding brain*")
    with col13:
        st.subheader("Siwalak")
        st.image("https://d26jy9fbi4q9wx.cloudfront.net/rails/active_storage/representations/proxy/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBMUEyQlE9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--29ef512c3baf6f950050c19cfccbb0c26e9994d0/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaDdCem9MWm05eWJXRjBTU0lKYW5CbFp3WTZCa1ZVT2hOeVpYTnBlbVZmZEc5ZlptbHNiRnNJYVFISWFRSElld1k2Q1dOeWIzQTZEbUYwZEdWdWRHbHZiZz09IiwiZXhwIjpudWxsLCJwdXIiOiJ2YXJpYXRpb24ifX0=--23cdbdf9871e44adeb4d843a03b0793a5f08394b/87DDABED-4B78-43EB-99A9-4CC56FDF3269_1_201_a.jpeg", width=80)
        st.markdown("*The Thai army knife*")
    st.divider()
    st.header("Problem")
    st.markdown("""
    Football is one of the <strong>most unpredictable major sports</strong>: a single goal can flip a match, and low-scoring dynamics make human intuition and simple stats models extremely noisy.<br>
    Fans, bettors, and analysts often rely on gut feeling or fragmented data views, which leads to <strong>inconsistent decisions and poor calibration</strong> of true match probabilities.
    """, unsafe_allow_html=True)
    st.divider()
    st.header("Solution")
    st.markdown("""
    <strong>We try to predict football matches! (Win, Draw, Lose)</strong> <br>
    We do this by <strong>predicting the number of goals per team</strong> using a <strong>stacking regression model</strong>.<br>
    We tried classification models too, but they came short to the <strong>>60% accuracy</strong> of our final regression model.<br>
    """, unsafe_allow_html=True)
    st.divider()
    st.header("Dataset")
    st.markdown("""
    Our model is powered by a Kaggle dataset sourced from Transfermarkt.de which is updated twice per week!<br>
    The dataset includes <strong>78k matches, 451 clubs, over 508k updated valuations of 33k players!</strong>
    Our key feature is therefore the team market value. The dataset allowed us to further engineer features such as home advantage,
    form streaks, and rest days capture both squad strength and situational context for every game.
    """, unsafe_allow_html=True)
    st.divider()
with tab2:
    st.header("A dog")
with tab3:
    st.write("### 🏟️ Match Selection")

    # Narrow down to 3 functional categories to simplify data mapping
    category = st.radio("Select Category",
                        ["Domestic League", "Domestic Cup", "Fantasy Match"],
                        horizontal=True)

    filtered_names = []

    if category == "Fantasy Match":
        # Show all clubs worldwide for cross-league matchups
        filtered_names = club_names
    else:
        # Map UI categories to 'type' column in competitions.csv
        type_map = {
            "Domestic League": "domestic_league",
            "Domestic Cup": "domestic_cup"
        }
        target_type = type_map.get(category)

        # Filter competitions by the selected type
        league_in_cat = comp_df[comp_df['type'].fillna('').str.strip() == target_type]
        league_options = sorted(league_in_cat['name'].unique())

        if league_options:
            selected_league = st.selectbox("Select League/Competition", league_options)

            # Extract metadata for the chosen competition
            comp_row = comp_df[comp_df['name'] == selected_league].iloc[0]

            # Use domestic_league_code (e.g., 'GB1') to pull all relevant teams for that country
            # This ensures Cup selections (like FA Cup) display all primary league clubs
            target_code = comp_row['domestic_league_code']

            if pd.notna(target_code) and target_code != "":
                # Filter clubs based on the domestic_competition_id column (verified from CSV)
                filtered_names = sorted(clubs_df[clubs_df['domestic_competition_id'] == target_code]['name'].unique())

        # Fallback if no clubs are filtered
        if not filtered_names:
            st.warning(f"No clubs found for the selected {category}.")
            filtered_names = club_names

    # UI DISPLAY (Dropdowns & Logos)
    col1, space, col2 = st.columns([10, 1, 10])

    with col1:
        # Home Team Selection
        home_team = st.selectbox("🏠 Home Team", filtered_names, key="home_select")
        home_id = clubs_df[clubs_df['name'] == home_team]['club_id'].values[0]
        st.image(get_logo_url(home_id), width=100)

    with col2:
        # Away Team Selection (default to second item to avoid initial match with Home)
        away_idx = 1 if len(filtered_names) > 1 else 0
        away_team = st.selectbox("🚌 Away Team", filtered_names, index=away_idx, key="away_select")
        away_id = clubs_df[clubs_df['name'] == away_team]['club_id'].values[0]
        st.image(get_logo_url(away_id), width=100)

    # Set default match date to the near future
    match_date = st.date_input("📅 Match Date", value=datetime.date(2026, 3, 15))

    # PREVIEW STATS
    st.write("### 📊 Team Comparison (Preview)")

    # get stats for preview
    try:
        # calculate features for preview
        home_id = clubs_df[clubs_df['name'] == home_team]['club_id'].values[0]
        away_id = clubs_df[clubs_df['name'] == away_team]['club_id'].values[0]

        from engine import get_match_features
        # get Features
        preview_features = get_match_features(s_dict, home_id, away_id, match_date)

        # show preview features in a nice format (you can customize this part)
        prev_col1, prev_col2, prev_col3 = st.columns([3, 2, 3])

        with prev_col1:
            st.metric("Market Value (Avg Last 3 Games):", f"€{preview_features['own_market_value']:,.0f}")
            st.metric("Position:", f"{int(preview_features['own_position'])}")
            st.metric("Rest Days:", f"{preview_features['own_restday']} days")
            st.metric("Current 2 games Streak:", f"{int(preview_features['own_streak_2'])} /6 Points")
            st.metric("Current 5 games Streak:", f"{int(preview_features['own_streak_5'])} /15 Points")

        with prev_col2:
            st.markdown("<h3 style='text-align: center;'>VS</h3>", unsafe_allow_html=True)

        with prev_col3:
            st.metric("Market Value (Avg Last 3 Games):", f"€{preview_features['opponent_market_value']:,.0f}")
            st.metric("Position:", f"{int(preview_features['opponent_position'])}")
            st.metric("Rest Days:", f"{preview_features['opponent_restday']} days")
            st.metric("Current 2 games Streak:", f"{int(preview_features['opponent_streak_2'])} /6 Points")
            st.metric("Current 5 games Streak:", f"{int(preview_features['opponent_streak_5'])} /15 Points")

    except Exception as e:
        st.info("Please select valid teams to preview stats.")

    # PREDICTION LOGIC
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

        #  DISPLAY RESULTS (SCOREBOARD STYLE)

        if result and "error" not in result:
            latest_iteration = st.empty()
            bar = st.progress(0)

            for i in range(100):
            # Update the progress bar with each iteration.
                latest_iteration.text(f'Loading Results {i+1}')
                bar.progress(i + 1)
                time.sleep(0.005)

            st.balloons()
            st.markdown("---")

            # Logic to display team win
            display_result = result['result']
            if result['result'] == "Home Win":
                display_result = f"🏆 {home_team} WIN!"
            elif result['result'] == "Away Win":
                display_result = f"🏆 {away_team} WIN!"
            elif result['result'] == "Draw":
                display_result = "🤝 IT'S A DRAW!"

            # Display Scores in a Scoreboard Style
            res_col1, res_col2, res_col3 = st.columns([2, 1, 2])

            with res_col1:
                st.markdown(f"<h1 style='text-align: center;'>{result['home_score']}</h1>", unsafe_allow_html=True)
                st.caption(f"<p style='text-align: center;'>{home_team}</p>", unsafe_allow_html=True)

            with res_col2:
                st.markdown("<h1 style='text-align: center; padding-top: 10px;'>-</h1>", unsafe_allow_html=True)

            with res_col3:
                st.markdown(f"<h1 style='text-align: center;'>{result['away_score']}</h1>", unsafe_allow_html=True)
                st.caption(f"<p style='text-align: center;'>{away_team}</p>", unsafe_allow_html=True)

            # Show Result Label Below the Scores
            st.markdown(f"<h2 style='text-align: center; color: #ae43a3;'>{display_result}</h2>", unsafe_allow_html=True)

            # Technical Details (Show Raw Model Output)
            with st.expander("🔍 Technical Details (Model Raw Output and Raw Input)"):
                st.json(result)
