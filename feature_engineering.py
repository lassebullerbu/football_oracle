def add_marketvalue():
    path = "raw_data"

    import pandas as pd
    appearances = pd.read_csv(path + "/appearances.csv")
    #club_games = pd.read_csv(path + "/club_games.csv")
    #clubs = pd.read_csv(path + "/clubs.csv")
    #competitions = pd.read_csv(path + "/competitions.csv")
    #game_events = pd.read_csv(path + "/game_events.csv")
    #game_lineups = pd.read_csv(path + "/game_lineups.csv")
    #games = pd.read_csv(path + "/games.csv")
    player_valuations = pd.read_csv(path + "/player_valuations.csv")
    #players = pd.read_csv(path + "/players.csv")
    #transfers = pd.read_csv(path + "/transfers.csv")

    appearances = appearances[['appearance_id', "game_id", "player_id", "player_club_id", "date", "minutes_played"]]
    player_valuations = player_valuations[['player_id', "date", "market_value_in_eur"]]

    #games['date'] = pd.to_datetime(games['date'])
    appearances['date'] = pd.to_datetime(appearances['date'])
    player_valuations['date'] = pd.to_datetime(player_valuations['date'])

    appearances_with_val = appearances.merge(
        player_valuations,
        on='player_id',
        how='left',
        suffixes=('', '_val')
    )

    appearances_with_val = appearances_with_val.query('date >= date_val')

    appearances_with_val = appearances_with_val.loc[
        appearances_with_val.groupby(['game_id', 'player_id'])['date_val'].idxmax()
    ].drop_duplicates(subset=['game_id', 'player_id'])

    total_minutes_per_game = appearances_with_val.groupby('game_id')['minutes_played'].max().reset_index()
    total_minutes_per_game = total_minutes_per_game.rename(columns={'minutes_played': 'total_minutes_per_game'})

    appearances_with_val = appearances_with_val.merge(total_minutes_per_game, on='game_id', how='left')
    appearances_with_val.head()

    appearances_with_val['weighted_market_value'] = (
        appearances_with_val['minutes_played'] *
        appearances_with_val['market_value_in_eur']
    ) / appearances_with_val['total_minutes_per_game']

    appearances_with_val.head()

    # Now aggregate market value per club per game
    club_mv_per_game = appearances_with_val.groupby(['game_id', 'player_club_id'])['weighted_market_value'].sum().reset_index()

    club_mv_per_game.rename(columns={'weighted_market_value': 'aggregate_market_value'}, inplace=True)

    games = pd.read_csv(path + "/games.csv")

    games_enhanced = games.merge(
        club_mv_per_game,
        left_on=['game_id', 'home_club_id'],
        right_on=['game_id', 'player_club_id'],
        how='left',
        suffixes=('', '_home')
    ).drop(columns=['player_club_id'])

    games_enhanced = games_enhanced.merge(
        club_mv_per_game,
        left_on=['game_id', 'away_club_id'],
        right_on=['game_id', 'player_club_id'],
        how='left',
        suffixes=('', '_away')
    ).drop(columns=['player_club_id'])

    # Rename for clarity
    games_enhanced.rename(columns={'aggregate_market_value': 'home_aggregate_market_value'}, inplace=True)
    games_enhanced.rename(columns={'aggregate_market_value_away': 'away_aggregate_market_value'}, inplace=True)

    return games_enhanced

def streak_and_restday_feature():
    """
    streak_and_restday_feature def wil return dataframe
    """

    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("davidcariboo/player-scores")

    print("Path to dataset files:", path)

    import pandas as pd

    club_games = pd.read_csv(path + "/club_games.csv")
    games = pd.read_csv(path + "/games.csv")

    """Home, TeamValue, TeamValue Away, Streak Home, Streak Away, Last Game within 4 days, Last Game within 4 days Away"""

    # 2. transform to datetime
    games['date'] = pd.to_datetime(games['date'])
    if 'date' in club_games.columns:
        club_games = club_games.drop(columns=['date'])
    club_games = club_games.merge(games[['game_id', 'date']], on='game_id', how='left')

    # sort by club and date
    club_games = club_games.sort_values(['club_id', 'date'])

    # 3. transform score (win=3, draw=1, lose=0)
    def get_pts(row):
        if row['is_win'] == 1: return 3
        if row['own_goals'] == row['opponent_goals']: return 1
        return 0

    club_games['match_points'] = club_games.apply(get_pts, axis=1)

    # 4. คำนวณ Streak (ไม่ต้อง fillna!)
    # ใช้ shift(1) เพื่อไม่ให้โมเดลแอบดูผลนัดปัจจุบัน
    club_games['streak_5'] = club_games.groupby('club_id')['match_points'].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum().shift(1)
    )
    #4. Calculate Streak for 2 last two games, sum point, shift 1 is not include current game(date)
    club_games['streak_2'] = club_games.groupby('club_id')['match_points'].transform(
        lambda x: x.rolling(window=2, min_periods=1).sum().shift(1)
    )

    # 5. Creat New Dataframe
    df_oracle = games[['game_id', 'date', 'home_club_id', 'away_club_id',
                    'home_club_goals', 'away_club_goals']].copy()

    # Streak to Home
    lookup = club_games[['game_id', 'club_id', 'streak_2','streak_5']]
    df_oracle = df_oracle.merge(lookup, left_on=['game_id', 'home_club_id'],
                            right_on=['game_id', 'club_id'], how='left')
    df_oracle = df_oracle.rename(columns={'streak_2': 'home_streak_2',
                                        'streak_5': 'home_streak_5'}).drop(columns=['club_id'])

    #  Streak to Away
    df_oracle = df_oracle.merge(lookup, left_on=['game_id', 'away_club_id'],
                            right_on=['game_id', 'club_id'], how='left')
    df_oracle = df_oracle.rename(columns={'streak_2': 'away_streak_2',
                                        'streak_5': 'away_streak_5'}).drop(columns=['club_id'])

    # 6. create win draw lose column
    def get_result(row):
        if row['home_club_goals'] > row['away_club_goals']: return 2 # Win
        if row['home_club_goals'] == row['away_club_goals']: return 1 # Draw
        return 0 # Loss
    df_oracle['target_result'] = df_oracle.apply(get_result, axis=1)
    df_oracle = df_oracle.dropna(subset=['home_streak_2', 'away_streak_2'])

    games['date'] = pd.to_datetime(games['date'])
    club_games_for_restday = club_games.sort_values(['club_id', 'date'])

    club_games_for_restday['last_game'] = club_games.groupby('club_id')['date'].shift(1)

    club_games_for_restday['rest_day'] = (club_games_for_restday['date'] - club_games_for_restday['last_game']).dt.days
    club_games_for_restday['rest_day'] = club_games_for_restday['rest_day'].fillna(0)

    # Streak to Home
    lookup = club_games_for_restday[['game_id', 'club_id', 'rest_day']]
    df_oracle = df_oracle.merge(lookup, left_on=['game_id', 'home_club_id'],
                            right_on=['game_id', 'club_id'], how='left')
    df_oracle = df_oracle.rename(columns={'rest_day': 'home_restday'}).drop(columns=['club_id'])

    #  Streak to Away
    df_oracle = df_oracle.merge(lookup, left_on=['game_id', 'away_club_id'],
                            right_on=['game_id', 'club_id'], how='left')
    df_oracle = df_oracle.rename(columns={'rest_day': 'away_restday'}).drop(columns=['club_id'])
    df_oracle.head(100)



    return df_oracle

def add_ishone_isaway():
    import kagglehub
    path = kagglehub.dataset_download("davidcariboo/player-scores")
    import pandas as pd
    games = pd.read_csv(path + "/games.csv")

    games = pd.read_csv(path + "/games.csv")
    games['ishome'] = (games['home_club_id'] == games['homeid']).astype(int)
    games['isaway'] = (games['away_club_id'] == games['homeid']).astype(int)
    df = games[['ishome','isaway']]
    return df

def add_win_loss():
    import kagglehub
    path = kagglehub.dataset_download("davidcariboo/player-scores")
    import pandas as pd
    games = pd.read_csv(path + "/games.csv")


    games["win_lose"] = games["home_club_goals"] - games["away_club_goals"]
    games["win_lose"] = games["win_lose"].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1) # win for 1 ,equal for 0,lose for -1
    df =games[["win_lose"]]
    return df

def add_club_opponent_position():
    import kagglehub
    path = kagglehub.dataset_download("davidcariboo/player-scores")
    import pandas as pd
    games = pd.read_csv(path + "/games.csv")

    club_games = pd.read_csv(path + "/club_games.csv")


    club_games2 = club_games.merge(
    games[['game_id', 'home_club_position', 'away_club_position']],
    on='game_id',
    how='left')

    games['own_position'] = club_games2.apply(
    lambda row: row['home_club_position'] if row.get('is_home', False) or row['club_id'] == row.get('home_club_id')
               else row['away_club_position'],
    axis=1)

    games['opponent_position'] = club_games2.apply(
    lambda row: row['away_club_position'] if row['club_id'] == row.get('home_club_id') else row['home_club_position'],
    axis=1)

    df =games[["own_position","opponent_position"]]
    return df
