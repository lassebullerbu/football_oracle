import pandas as pd

path = "raw_data"

appearances = pd.read_csv(path + "/appearances.csv")
club_games = pd.read_csv(path + "/club_games.csv")
clubs = pd.read_csv(path + "/clubs.csv")
competitions = pd.read_csv(path + "/competitions.csv")
game_events = pd.read_csv(path + "/game_events.csv")
game_lineups = pd.read_csv(path + "/game_lineups.csv")
games = pd.read_csv(path + "/games.csv")
player_valuations = pd.read_csv(path + "/player_valuations.csv")
players = pd.read_csv(path + "/players.csv")
transfers = pd.read_csv(path + "/transfers.csv")

games['date'] = pd.to_datetime(games['date'])
appearances['date'] = pd.to_datetime(appearances['date'])
player_valuations['date'] = pd.to_datetime(player_valuations['date'])

club_games = club_games.merge(games[['game_id', 'date']], on='game_id', how='left')

# MARKET VALUE
appearances = appearances[['appearance_id', "game_id", "player_id", "player_club_id", "date", "minutes_played"]]
player_valuations = player_valuations[['player_id', "date", "market_value_in_eur"]]

appearances_with_val = appearances.merge(
player_valuations,
on='player_id',
how='left',
suffixes=('', '_val')
).query('date >= date_val')

appearances_with_val = appearances_with_val.loc[
appearances_with_val.groupby(['game_id', 'player_id'])['date_val'].idxmax()
].drop_duplicates(subset=['game_id', 'player_id'])

# Calculate total minutes per game to calculate weighted market value
total_minutes_per_game = appearances_with_val.groupby('game_id')['minutes_played'].max().reset_index()
total_minutes_per_game = total_minutes_per_game.rename(columns={'minutes_played': 'total_minutes_per_game'})
appearances_with_val = appearances_with_val.merge(total_minutes_per_game, on='game_id', how='left')
appearances_with_val['weighted_market_value'] = (
    appearances_with_val['minutes_played'] *
    appearances_with_val['market_value_in_eur']
) / appearances_with_val['total_minutes_per_game']

# Now aggregate market value per club per game
club_mv_per_game = appearances_with_val.groupby(['game_id', 'player_club_id'])['weighted_market_value'].sum().reset_index()
club_mv_per_game.rename(columns={'weighted_market_value': 'aggregate_market_value'}, inplace=True)

# Merge with games df
final_df = club_games.merge(
    club_mv_per_game,
    left_on=['game_id', 'club_id'],
    right_on=['game_id', 'player_club_id'],
    how='left',
    suffixes=('', '_home')
).drop(columns=['player_club_id'])

final_df = final_df.merge(
    club_mv_per_game,
    left_on=['game_id', 'opponent_id'],
    right_on=['game_id', 'player_club_id'],
    how='left',
    suffixes=('', '_away')
).drop(columns=['player_club_id'])

# Rename for clarity
final_df.rename(columns={'aggregate_market_value': 'home_aggregate_market_value'}, inplace=True)
final_df.rename(columns={'aggregate_market_value_away': 'away_aggregate_market_value'}, inplace=True)


# STREAK
final_df = final_df.sort_values(['club_id', 'date'])
# Transform score (win=3, draw=1, lose=0)
def get_pts(row):
    if row['is_win'] == 1: return 3
    if row['own_goals'] == row['opponent_goals']: return 1
    return 0
final_df['match_points'] = final_df.apply(get_pts, axis=1)

# Calculate Streak for 5 last two games, sum point, shift 1 is not include current game(date)
final_df['streak_5'] = final_df.groupby('club_id')['match_points'].transform(
    lambda x: x.rolling(window=5, min_periods=1).sum().shift(1)
)
# Calculate Streak for 2 last two games, sum point, shift 1 is not include current game(date)
final_df['streak_2'] = final_df.groupby('club_id')['match_points'].transform(
    lambda x: x.rolling(window=2, min_periods=1).sum().shift(1)
)
final_df["streak_2"] = final_df["streak_2"].fillna(0)
final_df["streak_5"] = final_df["streak_5"].fillna(0)


# RESTDAYS
final_df['last_game'] = final_df.groupby('club_id')['date'].shift(1)
final_df['rest_days'] = (final_df['date'] - final_df['last_game']).dt.days
final_df['rest_days'] = final_df['rest_days'].fillna(7)


# WIN_LOSE
final_df["win_lose"] = final_df["own_goals"] - final_df["opponent_goals"]
final_df["win_lose"] = final_df["win_lose"].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1) # win for 1 ,equal for 0,lose for -1

# HOME_GAME BINARY
final_df['home_game'] = (final_df['hosting'] == 'Home').astype(int)

# CLEANING
final_df = final_df.drop(columns=['own_manager_name',"opponent_manager_name","is_win","match_points","last_game","hosting"])

final_df.to_csv('final_df.csv', index=False)
