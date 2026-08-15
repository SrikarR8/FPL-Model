import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

# 1. Load data directly from ModelData and RawData (for FPL ownership / selected)
base_dir = os.path.dirname(os.path.abspath(__file__))
player_data_path = os.path.join(base_dir, 'ModelData', 'playerData.csv')
team_data_path = os.path.join(base_dir, 'ModelData', 'teamData.csv')
project_root = os.path.dirname(base_dir)

df_player = pd.read_csv(player_data_path)
df_team = pd.read_csv(team_data_path)

seasons = ['2022-23', '2023-24', '2024-25', '2025-26', '2026-27']
df_player = df_player[df_player['season'].isin(seasons)].copy()
df_team = df_team[df_team['season'].isin(seasons)].copy()

# Populate 2026-27 player positions & teams from historical seasons / RawData players_raw
raw_base = os.path.join(project_root, 'RawData', 'csv_data')
pr_file = os.path.join(raw_base, '2026-27', 'players_raw.csv')
if os.path.exists(pr_file):
    pr = pd.read_csv(pr_file, encoding='latin-1')
    pr['name'] = pr['first_name'] + ' ' + pr['second_name']
    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    pr_pos = dict(zip(pr['name'], pr['element_type'].map(pos_map)))
    df_player['position'] = df_player['position'].replace('UNK', np.nan)
    df_player['position'] = df_player['position'].fillna(df_player['name'].map(pr_pos))

    # Align 2026-27 player team and fixture/venue mapping
    player_team_map = dict(zip(pr['name'], pr['team']))
    gw1_fixtures = df_team[(df_team['season'] == '2026-27') & (df_team['event'] == 1)]
    home_teams = dict(zip(gw1_fixtures['team_h'], gw1_fixtures['id']))
    away_teams = dict(zip(gw1_fixtures['team_a'], gw1_fixtures['id']))

    p26_mask = df_player['season'] == '2026-27'
    player_teams = df_player.loc[p26_mask, 'name'].map(player_team_map)
    is_home_26 = player_teams.isin(home_teams.keys())
    df_player.loc[p26_mask, 'was_home'] = is_home_26.astype(int)
    df_player.loc[p26_mask, 'fixture'] = player_teams.map(home_teams).fillna(player_teams.map(away_teams))

known_positions = df_player[df_player['position'] != 'UNK'].groupby('name')['position'].last()
df_player['position'] = df_player['position'].fillna(df_player['name'].map(known_positions))

# Add season_id (1, 2, 3, 4, 5)
season_id_map = {s: i + 1 for i, s in enumerate(seasons)}
df_player['season_id'] = df_player['season'].map(season_id_map)

# Load FPL ownership (selected) & raw bonus system points (bps) from RawData
raw_selected_list = []
for s in seasons:
    mgw_file = os.path.join(raw_base, s, 'gws', 'merged_gw.csv')
    if os.path.exists(mgw_file):
        d = pd.read_csv(mgw_file, encoding='latin-1', low_memory=False)
        d['season'] = s
        d['GW'] = pd.to_numeric(d['GW'], errors='coerce')
        d['fixture'] = pd.to_numeric(d['fixture'], errors='coerce')
        cols_to_pull = ['name', 'season', 'GW', 'fixture']
        if 'selected' in d.columns: cols_to_pull.append('selected')
        if 'bps' in d.columns: cols_to_pull.append('bps')
        raw_selected_list.append(d[cols_to_pull])

if raw_selected_list:
    raw_selected_df = pd.concat(raw_selected_list, ignore_index=True).drop_duplicates(subset=['name', 'season', 'GW', 'fixture'])
    df_player['GW'] = pd.to_numeric(df_player['GW'], errors='coerce')
    df_player['fixture'] = pd.to_numeric(df_player['fixture'], errors='coerce')
    df_player = pd.merge(df_player, raw_selected_df, on=['name', 'season', 'GW', 'fixture'], how='left')
else:
    df_player['selected'] = 0
    df_player['bps'] = 0

# -------------------------------------------------------------
# 1. Opponent League Position (GW - 1 Lagged, Zero Look-Ahead)
# -------------------------------------------------------------
h_team = df_team[['season', 'event', 'team_h', 'home_pos']].rename(columns={'team_h': 'team', 'home_pos': 'pos'})
a_team = df_team[['season', 'event', 'team_a', 'away_pos']].rename(columns={'team_a': 'team', 'away_pos': 'pos'})
team_pos_table = pd.concat([h_team, a_team]).drop_duplicates(subset=['season', 'event', 'team'])

# Map previous gameweek (event - 1) standings onto fixtures
df_team['prev_event'] = np.maximum(1, df_team['event'] - 1)

df_team = pd.merge(
    df_team,
    team_pos_table[['season', 'event', 'team', 'pos']],
    left_on=['season', 'prev_event', 'team_h'],
    right_on=['season', 'event', 'team'],
    how='left'
).rename(columns={'pos': 'home_league_pos'})

df_team = pd.merge(
    df_team,
    team_pos_table[['season', 'event', 'team', 'pos']],
    left_on=['season', 'prev_event', 'team_a'],
    right_on=['season', 'event', 'team'],
    how='left'
).rename(columns={'pos': 'away_league_pos'})

# For 2026-27 GW1, anchor to 2025-26 final season standings
prev_season_final = team_pos_table[(team_pos_table['season'] == '2025-26') & (team_pos_table['event'] == 38)]
prev_season_map = dict(zip(prev_season_final['team'], prev_season_final['pos']))
df_team.loc[df_team['season'] == '2026-27', 'home_league_pos'] = df_team.loc[df_team['season'] == '2026-27', 'team_h'].map(prev_season_map).fillna(18.0)
df_team.loc[df_team['season'] == '2026-27', 'away_league_pos'] = df_team.loc[df_team['season'] == '2026-27', 'team_a'].map(prev_season_map).fillna(18.0)

# -------------------------------------------------------------
# 2. Opponent xG Allowed (Cumulative Season Avg)
# -------------------------------------------------------------
home_records = df_team[['season', 'event', 'id', 'kickoff_time', 'team_h', 'team_a', 'h_xga']].copy()
home_records.columns = ['season', 'event', 'fixture', 'kickoff_time', 'team_id', 'opp_id', 'match_xga']
home_records['isHome'] = True

away_records = df_team[['season', 'event', 'id', 'kickoff_time', 'team_a', 'team_h', 'a_xga']].copy()
away_records.columns = ['season', 'event', 'fixture', 'kickoff_time', 'team_id', 'opp_id', 'match_xga']
away_records['isHome'] = False

team_matches = pd.concat([home_records, away_records], ignore_index=True)
team_matches['kickoff_time'] = team_matches['kickoff_time'].fillna('').astype(str)
team_matches = team_matches.sort_values(by=['team_id', 'season', 'kickoff_time', 'event']).reset_index(drop=True)

# Cumulative Season Average xGA (shifted by 1 to exclude current match)
team_matches['opp_xga_season_avg'] = team_matches.groupby(['team_id', 'season'])['match_xga'].transform(
    lambda x: x.shift(1).expanding(min_periods=1).mean()
)

# Fallback to career average xGA for early season matches
team_career_xga = team_matches.groupby('team_id')['match_xga'].transform(
    lambda x: x.shift(1).expanding(min_periods=1).mean()
)
team_matches['opp_xga_season_avg'] = team_matches['opp_xga_season_avg'].fillna(team_career_xga).fillna(1.35)

# Merge opponent defense onto df_player
df_player['was_home_bool'] = df_player['was_home'].astype(bool)

opp_defense = team_matches[['season', 'fixture', 'isHome', 'opp_xga_season_avg']].copy()
opp_defense.columns = ['season', 'fixture', 'opp_isHome', 'opp_xga_season_avg']

df_player = pd.merge(
    df_player,
    opp_defense,
    left_on=['season', 'fixture', ~df_player['was_home_bool']],
    right_on=['season', 'fixture', 'opp_isHome'],
    how='left'
)

# Merge league positions onto df_player and compute opp_league_pos
diff_df = df_team[['season', 'id', 'kickoff_time', 'home_league_pos', 'away_league_pos']].drop_duplicates(subset=['season', 'id'])
df_player = pd.merge(
    df_player,
    diff_df,
    left_on=['season', 'fixture'],
    right_on=['season', 'id'],
    how='left'
)

# Opponent league pos: If player is home, opp is away; if player is away, opp is home
df_player['opp_league_pos'] = np.where(df_player['was_home_bool'], df_player['away_league_pos'], df_player['home_league_pos'])
df_player['opp_league_pos'] = df_player['opp_league_pos'].fillna(10.0)
df_player['opp_xga_season_avg'] = df_player['opp_xga_season_avg'].fillna(1.35)

# Construct Composite Opponent Difficulty Index (0 to 1, higher = easier fixture)
df_player['composite_opp_difficulty'] = (
    (df_player['opp_league_pos'] / 20.0) * 0.5 + 
    (df_player['opp_xga_season_avg'] / 2.5) * 0.5
)

# Sort strictly by name and kickoff_time for chronological rolling calculation
df_player['kickoff_time'] = df_player['kickoff_time'].fillna('').astype(str)
df_player['GW'] = pd.to_numeric(df_player['GW'], errors='coerce')
df_player = df_player.sort_values(by=['name', 'season', 'kickoff_time', 'GW']).reset_index(drop=True)

# Define double gameweeks
df_player['is_double_gameweek'] = (df_player.groupby(['name', 'season', 'GW'])['name'].transform('count') > 1).astype(int)

df_player['FPL_Points'] = pd.to_numeric(df_player['FPL_Points'], errors='coerce')
df_player['MinutesPlayed'] = pd.to_numeric(df_player['MinutesPlayed'], errors='coerce')
df_player['xG'] = pd.to_numeric(df_player['xG'], errors='coerce')
df_player['xA'] = pd.to_numeric(df_player['xA'], errors='coerce')
df_player['BonusPoints'] = pd.to_numeric(df_player['BonusPoints'], errors='coerce').fillna(0)
df_player['bps'] = pd.to_numeric(df_player['bps'], errors='coerce').fillna(df_player['BonusPoints'] * 10)

# Short-term rolling metrics (5 games)
df_player['rolling_avg_points_last_5'] = df_player.groupby('name')['FPL_Points'].transform(
    lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
)
df_player['rolling_avg_mins_last_5'] = df_player.groupby('name')['MinutesPlayed'].transform(
    lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
)
df_player['rolling_avg_xG_last_5'] = df_player.groupby('name')['xG'].transform(
    lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
)
df_player['rolling_avg_xA_last_5'] = df_player.groupby('name')['xA'].transform(
    lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
)
df_player['rolling_avg_bps_last_5'] = df_player.groupby('name')['bps'].transform(
    lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
)

# Long-term 38-game baseline anchors (captures proven full-season player quality & bonus magnet rating)
df_player['rolling_avg_points_last_38'] = df_player.groupby('name')['FPL_Points'].transform(
    lambda x: x.shift(1).rolling(window=38, min_periods=3).mean()
)
df_player['rolling_avg_points_last_38'] = df_player['rolling_avg_points_last_38'].fillna(df_player['rolling_avg_points_last_5'])

df_player['rolling_avg_xG_last_38'] = df_player.groupby('name')['xG'].transform(
    lambda x: x.shift(1).rolling(window=38, min_periods=3).mean()
)
df_player['rolling_avg_xG_last_38'] = df_player['rolling_avg_xG_last_38'].fillna(df_player['rolling_avg_xG_last_5'])

df_player['rolling_avg_bps_last_38'] = df_player.groupby('name')['bps'].transform(
    lambda x: x.shift(1).rolling(window=38, min_periods=3).mean()
)
df_player['rolling_avg_bps_last_38'] = df_player['rolling_avg_bps_last_38'].fillna(df_player['rolling_avg_bps_last_5'])

# Drop first season warm-up data (season_id > 1)
df_model = df_player[df_player['season_id'] > 1].copy()

df_model['target_points'] = pd.to_numeric(df_model['FPL_Points'], errors='coerce')
df_model['was_home'] = df_model['was_home'].astype(int)

# Filter for regular starter Forwards priced >= £3.8m
MW_df = df_model.rename(columns={'GW': 'match_week'}).copy()
MW_df = MW_df[MW_df['New_Cost'] >= 3.8].copy()
FWD_df = MW_df[MW_df['position'] == 'FWD'].copy()

FWD_starters_df = FWD_df[FWD_df['rolling_avg_mins_last_5'] >= 40].copy()
FWD_starters_df['target_points'] = FWD_starters_df['target_points'].clip(lower=0)

# Feature set: Short-term form + 38-GW Talent/xG/BPS Anchors + Composite Fixture Difficulty
feature_columns = [
    'rolling_avg_points_last_5',
    'rolling_avg_points_last_38',
    'rolling_avg_mins_last_5',
    'rolling_avg_xG_last_5',
    'rolling_avg_xA_last_5',
    'rolling_avg_xG_last_38',
    'rolling_avg_bps_last_38',
    'composite_opp_difficulty',
    'was_home'
]

test_metadata_list = []
target_season = '2025-26'

print("=" * 115)
print("🏃 RUNNING WALK-FORWARD BACKTEST FOR SEASON 2025-26 (GW 1 to 38)...")
print("=" * 115)
print(f"Model Architecture: max_depth=3, colsample_bytree=0.8, learning_rate=0.05, n_estimators=100")
print(f"Features: {feature_columns}\n")

# Walk-Forward Backtest Loop (Gameweeks 1 to 38 of 2025-26)
for gw in range(1, 39):
    test_mask = (FWD_starters_df['season'] == target_season) & (FWD_starters_df['match_week'] == gw)
    X_test = FWD_starters_df.loc[test_mask, feature_columns]

    if X_test.empty:
        continue

    if gw >= 4:
        val_mask = (FWD_starters_df['season'] == target_season) & \
                   (FWD_starters_df['match_week'] >= gw - 3) & \
                   (FWD_starters_df['match_week'] < gw)
        train_inner_mask = (FWD_starters_df['season'] < target_season) | \
                           ((FWD_starters_df['season'] == target_season) & (FWD_starters_df['match_week'] < gw - 3))
    else:
        val_mask = (FWD_starters_df['season'] == '2024-25') & (FWD_starters_df['match_week'] >= 36)
        train_inner_mask = (FWD_starters_df['season'] < '2024-25') | \
                           ((FWD_starters_df['season'] == '2024-25') & (FWD_starters_df['match_week'] < 36))

    X_train_inner = FWD_starters_df.loc[train_inner_mask, feature_columns]
    y_train_inner = FWD_starters_df.loc[train_inner_mask, 'target_points']

    # XGBoost Regularized Model (max_depth=3, colsample_bytree=0.8)
    model = xgb.XGBRegressor(
        objective='reg:tweedie',
        tweedie_variance_power=1.2,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1
    )
    model.fit(X_train_inner, y_train_inner, verbose=False)

    gw_metadata = FWD_starters_df.loc[test_mask, ['name', 'season', 'match_week', 'target_points', 'rolling_avg_points_last_5', 'selected']].copy()
    gw_metadata['predicted_points'] = model.predict(X_test)
    test_metadata_list.append(gw_metadata)

test_metadata = pd.concat(test_metadata_list, ignore_index=True)

# Compute Gameweek-by-Gameweek Performance Metrics (Model vs Baseline vs Most Popular)
gw_stats = []
for gw, group in test_metadata.groupby('match_week'):
    y_true = group['target_points'].values
    y_pred = group['predicted_points'].values
    y_base = group['rolling_avg_points_last_5'].values

    # Spearman correlations and NDCGs
    corr_model, _ = spearmanr(y_true, y_pred) if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1 else (0.0, 1.0)
    corr_base, _ = spearmanr(y_true, y_base) if len(np.unique(y_true)) > 1 and len(np.unique(y_base)) > 1 else (0.0, 1.0)
    ndcg_model = ndcg_score([y_true], [y_pred]) if len(y_true) > 1 else 0.0
    ndcg_base = ndcg_score([y_true], [y_base]) if len(y_true) > 1 else 0.0
    k_val = min(5, len(y_true))
    ndcg_5_model = ndcg_score([y_true], [y_pred], k=k_val) if len(y_true) > 1 else 0.0
    ndcg_5_base = ndcg_score([y_true], [y_base], k=k_val) if len(y_true) > 1 else 0.0

    # Top 5 Picks Actual Points Scored:
    top5_model = group.sort_values(by='predicted_points', ascending=False).head(5)
    model_top5_pts = top5_model['target_points'].mean()

    top5_base = group.sort_values(by='rolling_avg_points_last_5', ascending=False).head(5)
    base_top5_pts = top5_base['target_points'].mean()

    top5_popular = group.sort_values(by='selected', ascending=False).head(5)
    popular_top5_pts = top5_popular['target_points'].mean()

    gw_stats.append({
        'GW': int(gw),
        'Players': len(group),
        'Model Corr': corr_model,
        'Base Corr': corr_base,
        'Model NDCG': ndcg_model,
        'Base NDCG': ndcg_base,
        'Model Top5 Pts': model_top5_pts,
        'Base Top5 Pts': base_top5_pts,
        'Popular Top5 Pts': popular_top5_pts,
    })

results_df = pd.DataFrame(gw_stats)
overall_corr_model = spearmanr(test_metadata['target_points'], test_metadata['predicted_points'])
overall_corr_base = spearmanr(test_metadata['target_points'], test_metadata['rolling_avg_points_last_5'])

print("=" * 115)
print(f"📊 GAMEWEEK-BY-GAMEWEEK EVALUATION: TOP 5 PICKS POINTS COMPARISON (2025-26 GW 1 - 38)")
print("=" * 115)
print(f"{'GW':<5}{'Players':<9}{'Model Corr':<13}{'Base Corr':<13}{'Model NDCG':<13}{'Model Top5 Pts':<18}{'Base Top5 Pts':<18}{'Popular Top5 Pts':<18}")
print("-" * 115)
for _, r in results_df.iterrows():
    print(f"GW {int(r['GW']):<2} {int(r['Players']):<9}{r['Model Corr']:<13.4f}{r['Base Corr']:<13.4f}{r['Model NDCG']:<13.4f}{r['Model Top5 Pts']:<18.2f}{r['Base Top5 Pts']:<18.2f}{r['Popular Top5 Pts']:<18.2f}")

print("\n" + "=" * 80)
print("📈 SEASON 2025-26 OVERALL SUMMARY (MODEL VS BASELINE VS MOST POPULAR)")
print("=" * 80)
print(f"{'Metric / Strategy':<42}{'Score / Pts'}")
print("-" * 80)
print(f"{'Average Weekly Spearman Correlation (Model):':<42}{results_df['Model Corr'].mean():.4f}")
print(f"{'Average Weekly Spearman Correlation (Baseline):':<42}{results_df['Base Corr'].mean():.4f}")
print(f"{'Average Weekly NDCG (Model):':<42}{results_df['Model NDCG'].mean():.4f}")
print(f"{'Average Weekly NDCG (Baseline):':<42}{results_df['Base NDCG'].mean():.4f}")
print(f"{'Overall Pooled Spearman Correlation (Model):':<42}{overall_corr_model.statistic:.4f}")
print(f"{'Overall Pooled Spearman Correlation (Baseline):':<42}{overall_corr_base.statistic:.4f}")
print("-" * 80)
print(f"{'⭐ Top 5 Model Picks (Avg Points/Player/GW):':<42}{results_df['Model Top5 Pts'].mean():.2f} pts")
print(f"{'⭐ Top 5 Baseline Form Picks (Avg Points/GW):':<42}{results_df['Base Top5 Pts'].mean():.2f} pts")
print(f"{'⭐ Top 5 Most Popular FPL Picks (Avg Points/GW):':<42}{results_df['Popular Top5 Pts'].mean():.2f} pts")
print("=" * 80)

# Feature Importances
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop Features by Built-in Importance (Gain):")
print(importance_df.to_string(index=False))

# -------------------------------------------------------------
# 🚀 2026-27 GAMEWEEK 1 FORWARD PREDICTIONS (UPCOMING FIXTURES)
# -------------------------------------------------------------
# Train final model on all completed historical seasons (2023-24, 2024-25, 2025-26)
train_all_mask = (FWD_starters_df['season'].isin(['2023-24', '2024-25', '2025-26'])) & (FWD_starters_df['target_points'].notna())
X_train_final = FWD_starters_df.loc[train_all_mask, feature_columns]
y_train_final = FWD_starters_df.loc[train_all_mask, 'target_points']

final_model = xgb.XGBRegressor(
    objective='reg:tweedie',
    tweedie_variance_power=1.2,
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=1
)
final_model.fit(X_train_final, y_train_final, verbose=False)

# Predict 2026-27 GW1
pred_2627_mask = (FWD_starters_df['season'] == '2026-27') & (FWD_starters_df['match_week'] == 1)
df_pred_2627 = FWD_starters_df.loc[pred_2627_mask].copy()

if not df_pred_2627.empty:
    df_pred_2627['predicted_points'] = final_model.predict(df_pred_2627[feature_columns])
    rankings_2627 = df_pred_2627[[
        'name', 'New_Cost', 'predicted_points', 
        'rolling_avg_points_last_38', 'rolling_avg_points_last_5', 
        'rolling_avg_xG_last_5', 'composite_opp_difficulty', 'was_home'
    ]].sort_values(by='predicted_points', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 115)
    print("🚀 2026-27 GAMEWEEK 1 FORWARD PROJECTIONS (TOP PICKS FOR UPCOMING FIXTURES)")
    print("=" * 115)
    print(f"{'Rank':<6}{'Player':<32}{'Price':<8}{'Pred Pts':<12}{'38-GW Anchor':<15}{'5-GW Form':<13}{'5-GW xG':<11}{'Venue':<8}")
    print("-" * 115)
    for idx, r in rankings_2627.head(15).iterrows():
        venue_str = "Home" if r['was_home'] == 1 else "Away"
        print(f"#{idx+1:<5}{r['name']:<32}£{r['New_Cost']:<7.1f}{r['predicted_points']:<12.2f}{r['rolling_avg_points_last_38']:<15.2f}{r['rolling_avg_points_last_5']:<13.2f}{r['rolling_avg_xG_last_5']:<11.2f}{venue_str:<8}")
    print("=" * 115)

