import os
import pandas as pd
import numpy as np


# Define target seasons at the top
SEASONS = [
    '2018-19', '2019-20', '2020-21',
    '2021-22', '2022-23', '2023-24', '2024-25', '2025-26', '2026-27'
]

# Rolling stat windows for grid search: 1, 5, 15, and 38 games
ROLLING_WINDOWS = [1, 5, 15, 38]




def engineerCols(seasons=SEASONS, windows=ROLLING_WINDOWS):
    """
    Engineers rolling statistics (5, 15, 38) for player, team, and opponent features,
    drops 0-minute rows, and returns the engineered DataFrame directly in-memory.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_data_dir = os.path.join(base_dir, 'Models', 'ModelData')

    player_data_path = os.path.join(model_data_dir, 'playerData.csv')
    team_data_path = os.path.join(model_data_dir, 'teamData.csv')

    print("Re-engineering feature dataset in-memory")
    df_player = pd.read_csv(player_data_path)
    df_team = pd.read_csv(team_data_path)

    # Filter data for requested seasons
    df_player = df_player[df_player['season'].isin(seasons)].copy()
    df_team = df_team[df_team['season'].isin(seasons)].copy()

    # Merge kickoff_time from teamData onto df_player for exact chronological ordering
    fixture_times = df_team[['season', 'id', 'kickoff_time']].drop_duplicates(subset=['season', 'id'])
    df_player = pd.merge(
        df_player,
        fixture_times,
        left_on=['season', 'fixture'],
        right_on=['season', 'id'],
        how='left'
    )

    # -------------------------------------------------------------
    # 1. PLAYER-LEVEL ROLLING & SHIFTED STATS
    # -------------------------------------------------------------
    df_player = df_player.sort_values(by=['name', 'season', 'kickoff_time', 'GW']).reset_index(drop=True)

    player_roll_metrics = [
        'xG', 'xA', 'Goals', 'Assists', 'MinutesPlayed',
        'FPL_Points', 'BonusPoints'
    ]

    player_rolling_cols = []
    for metric in player_roll_metrics:
        for w in windows:
            roll_col = f'{metric}_last_{w}'
            df_player[roll_col] = df_player.groupby('name')[metric].transform(
                lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
            )
            player_rolling_cols.append(roll_col)

    # Player cost feature without leakage (shifted by 1 game)
    df_player['New_Cost_last_1'] = df_player.groupby('name')['New_Cost'].shift(1)
    player_rolling_cols.append('New_Cost_last_1')

    # Overall career per-game average for YellowCards (shifted by 1 game)
    df_player['YellowCards_per_game'] = df_player.groupby('name')['YellowCards'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    ).fillna(0.0)

    card_avg_cols = ['YellowCards_per_game']

    # -------------------------------------------------------------
    # 2. TEAM & OPPONENT CHRONOLOGICAL ROLLING & SHIFTED STATS
    # -------------------------------------------------------------
    df_team = df_team.sort_values(by=['season', 'event', 'kickoff_time']).reset_index(drop=True)

    # Split fixtures into Home and Away team match records
    home_records = df_team[[
        'season', 'event', 'id', 'team_h', 'team_h_score', 'team_a_score',
        'h_xg', 'a_xg', 'home_pos', 'away_pos'
    ]].copy()
    home_records.columns = [
        'season', 'event', 'fixture', 'team_id', 'team_goals_scored',
        'opp_goals_conceded', 'team_xg',
        'opp_xg_allowed', 'h_pos', 'a_pos'
    ]
    home_records['isHome'] = True

    away_records = df_team[[
        'season', 'event', 'id', 'team_a', 'team_a_score', 'team_h_score',
        'a_xg', 'h_xg', 'home_pos', 'away_pos'
    ]].copy()
    away_records.columns = [
        'season', 'event', 'fixture', 'team_id', 'team_goals_scored',
        'opp_goals_conceded', 'team_xg',
        'opp_xg_allowed', 'h_pos', 'a_pos'
    ]
    away_records['isHome'] = False

    team_matches = pd.concat([home_records, away_records], ignore_index=True)

    # -------------------------------------------------------------
    # PROMOTED TEAM REMAPPING (Inherit Relegated Team Historical Stats)
    # -------------------------------------------------------------
    # Map promoted team IDs to relegated team benchmark IDs for smooth continuous rolling statistics
    team_matches['effective_team_id'] = team_matches['team_id']
    team_matches = team_matches.sort_values(by=['effective_team_id', 'season', 'event']).reset_index(drop=True)



    team_roll_metrics = [
        'team_goals_scored', 'opp_goals_conceded',
        'team_xg', 'opp_xg_allowed'
    ]

    team_rolling_cols = []
    for metric in team_roll_metrics:
        for w in windows:
            roll_col = f'{metric}_last_{w}'
            team_matches[roll_col] = team_matches.groupby('team_id')[metric].transform(
                lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
            )
            team_rolling_cols.append(roll_col)

    # -------------------------------------------------------------
    # 3. MERGE PLAYER & TEAM DATASETS
    # -------------------------------------------------------------
    df_merged = pd.merge(
        df_player,
        team_matches,
        left_on=['season', 'fixture', 'was_home'],
        right_on=['season', 'fixture', 'isHome'],
        how='left'
    )

    # Determine position column (and fill 2026-27 UNK positions from prior seasons)
    if 'position_x' in df_merged.columns:
        df_merged['position'] = df_merged['position_x']
    elif 'position' not in df_merged.columns:
        df_merged['position'] = 'UNK'

    pos_lookup = df_merged[df_merged['position'] != 'UNK'][['name', 'position']].drop_duplicates(subset=['name'], keep='last')
    df_merged = pd.merge(df_merged, pos_lookup, on='name', how='left', suffixes=('', '_known'))
    df_merged['position'] = df_merged['position'].replace('UNK', np.nan).fillna(df_merged['position_known']).fillna('UNK')
    df_merged.drop(columns=['position_known'], inplace=True, errors='ignore')


    # One-hot encode positions (FWD, MID, DEF, GK)
    df_merged['pos_FWD'] = (df_merged['position'] == 'FWD').astype(int)
    df_merged['pos_MID'] = (df_merged['position'] == 'MID').astype(int)
    df_merged['pos_DEF'] = (df_merged['position'] == 'DEF').astype(int)
    df_merged['pos_GK'] = (df_merged['position'] == 'GKP').astype(int) | (df_merged['position'] == 'GK').astype(int)

    player_header_cols = [
        'season', 'GW', 'name', 'position', 'pos_FWD', 'pos_MID', 'pos_DEF', 'pos_GK',
        'element', 'fixture', 'was_home', 'xG', 'xA',
        'Goals', 'Assists', 'MinutesPlayed', 'FPL_Points', 'YellowCards',
        'RedCards', 'BonusPoints', 'New_Cost'
    ]

    raw_match_cols = [
        'isHome', 'h_pos', 'a_pos',
        'team_goals_scored', 'team_xg',
        'opp_goals_conceded', 'opp_xg_allowed'
    ]

    # Deduplicate final column list preserving order
    all_requested_cols = player_header_cols + raw_match_cols + player_rolling_cols + team_rolling_cols + card_avg_cols
    final_cols = list(dict.fromkeys(all_requested_cols))
    df_features = df_merged[final_cols].sort_values(by=['season', 'GW', 'name']).reset_index(drop=True)


    # DROP ALL 0-MINUTE ROWS (except upcoming season GW1 placeholders)
    df_features = df_features[(df_features['MinutesPlayed'] > 0) | ((df_features['season'] == '2026-27') & (df_features['GW'] == 1))].reset_index(drop=True)

    print("Feature engineering complete.")
    return df_features

def loadCols(minutes=40, seasons=SEASONS):
    """
    Engineers feature dataset directly in memory and filters for game records where
    MinutesPlayed >= minutes (defaults to 40 minutes).
    """
    df_features = engineerCols(seasons=seasons)

    # Filter for seasons
    df_features = df_features[df_features['season'].isin(seasons)]

    # Filter for MinutesPlayed >= minutes (or season 2026-27 GW 1 placeholders)
    df_filtered = df_features[(df_features['MinutesPlayed'] >= minutes) | ((df_features['season'] == '2026-27') & (df_features['GW'] == 1))].reset_index(drop=True)
    print(f"Loaded dataset filtered for MinutesPlayed >= {minutes}: {len(df_filtered)} rows x {len(df_filtered.columns)} cols.")
    return df_filtered


# if __name__ == '__main__':
#     pd.set_option('display.max_columns', None)
#     pd.set_option('display.width', 1000)

#     # Re-engineer to strip 0-minute rows from model_data.csv
#     engineerCols()

#     # Load with default minutes=40 threshold
#     df = loadCols(minutes=40)

#     print("\nPreview Bukayo Saka 2025-26 games (MinutesPlayed >= 40):")
#     saka_2526 = df[(df['name'] == 'Bukayo Saka') & (df['season'] == '2025-26')]
#     print(saka_2526[['season', 'GW', 'fixture', 'MinutesPlayed', 'xG', 'xA', 'FPL_Points', 'opp_league_pos', 'isHome']].head(1000))
