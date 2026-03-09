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


def add_fatigue():
    CODE HERE

    return df
