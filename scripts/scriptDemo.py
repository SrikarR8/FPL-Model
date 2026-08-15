import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr


seasons = [
    '2018-19', '2019-20', '2020-21',
    '2021-22', '2022-23', '2023-24', '2024-25', '2025-26'
]

all_seasons_data = []

for i, season in enumerate(seasons):
    gw_path = f'Data/csv_data/{season}/gws/merged_gw.csv'
    fixtures_path = f'Data/csv_data/{season}/fixtures.csv'
    
    try:
        df_gw = pd.read_csv(gw_path, encoding='utf-8')
    except UnicodeDecodeError:
        df_gw = pd.read_csv(gw_path, encoding='latin-1')
        
    try:
        fixtures_df = pd.read_csv(fixtures_path, encoding='utf-8')
    except UnicodeDecodeError:
        fixtures_df = pd.read_csv(fixtures_path, encoding='latin-1')
    
    df_gw['season'] = season
    df_gw['season_id'] = i + 1
    
    home_diff_map = fixtures_df.set_index('id')['team_h_difficulty'].to_dict()
    away_diff_map = fixtures_df.set_index('id')['team_a_difficulty'].to_dict()
    
    df_gw['team_h_difficulty'] = df_gw['fixture'].map(home_diff_map)
    df_gw['team_a_difficulty'] = df_gw['fixture'].map(away_diff_map)
    
    df_gw['fixture_difficulty'] = np.where(
        df_gw['was_home'] == True,
        df_gw['team_h_difficulty'],  
        df_gw['team_a_difficulty']   
    )
    
    df_gw = df_gw.drop(columns=['team_h_difficulty', 'team_a_difficulty'])
    
    all_seasons_data.append(df_gw)

df_combined = pd.concat(all_seasons_data, ignore_index=True)

df_combined['kickoff_time'] = pd.to_datetime(df_combined['kickoff_time'])
df_combined['GW'] = pd.to_numeric(df_combined['GW'], errors='coerce')
df_combined = df_combined.sort_values(by=['name', 'kickoff_time']).reset_index(drop=True)

df_combined['is_double_gameweek'] = df_combined.groupby(['name', 'season', 'GW'])['name'].transform('count') > 1

rolling_metrics = {
    'total_points': 'rolling_avg_points_last_5',
    'minutes': 'rolling_avg_mins_last_5',
    'expected_goals': 'rolling_avg_xG_last_5',
    'expected_assists': 'rolling_avg_xA_last_5'
}

for orig_col, roll_col in rolling_metrics.items():
    if orig_col in df_combined.columns:
        df_combined[orig_col] = pd.to_numeric(df_combined[orig_col], errors='coerce')
        df_combined[roll_col] = df_combined.groupby('name')[orig_col].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )


df_model = df_combined[df_combined['season_id'] > 1].copy()

df_model['target_points'] = pd.to_numeric(df_model['total_points'], errors='coerce')

df_model['was_home'] = df_model['was_home'].astype(int)
df_model['is_double_gameweek'] = df_model['is_double_gameweek'].astype(int)

df_model['position'] = df_model['position'].replace({'GKP': 'GK'})

pos_dummies = pd.get_dummies(df_model['position'], prefix='is')

pos_dummies.columns = [col.replace('is_', 'is_') for col in pos_dummies.columns]

df_model = pd.concat([df_model, pos_dummies], axis=1)

feature_cols = [
    "name", "season", "GW", "kickoff_time", "minutes",
    "rolling_avg_points_last_5", "rolling_avg_mins_last_5", 
    "rolling_avg_xG_last_5", "rolling_avg_xA_last_5", 
    "fixture_difficulty", "was_home", "is_double_gameweek",
    "value", "is_GK", "is_DEF", "is_MID", "is_FWD",
    "target_points"
]

MW_df = df_model[feature_cols].copy()
MW_df = MW_df.rename(columns={"value": "new_cost", "GW": "match_week"})
MW_df = MW_df[MW_df['new_cost'] >= 38].copy()
pos_cols = ['is_GK', 'is_DEF', 'is_MID', 'is_FWD']
MW_df[pos_cols] = MW_df[pos_cols].astype(int)

FWD_df = MW_df[MW_df['is_FWD'] == 1]


# Define Starters: Players averaging at least 45 minutes over their last 5 games
FWD_starters_df = FWD_df[FWD_df['rolling_avg_mins_last_5'] >= 44].copy()

print(f"\nFiltered records: {len(FWD_df)} total forward entries -> {len(FWD_starters_df)} starter forward entries")


feature_columns = [
    "rolling_avg_points_last_5", 
    "rolling_avg_mins_last_5", 
    "rolling_avg_xG_last_5", 
    "rolling_avg_xA_last_5", 
    "fixture_difficulty", 
    "was_home", 
    "is_double_gameweek",
    "new_cost", 
    "is_GK", "is_DEF", "is_MID", "is_FWD"
]
test_metadata_list = []
target_season = '2025-26'
for gw in range(20, 39):
    
    # FIX 1: Use FWD_starters_df instead of FWD_df to keep out the noise
    train_mask = (FWD_starters_df['season'] < target_season) | ((FWD_starters_df['season'] == target_season) & (FWD_starters_df['match_week'] < gw))
    test_mask = (FWD_starters_df['season'] == target_season) & (FWD_starters_df['match_week'] == gw)
    
    X_test = FWD_starters_df.loc[test_mask, feature_columns]
    
    if X_test.empty:
        continue

    val_mask = (FWD_starters_df['season'] == target_season) & \
                (FWD_starters_df['match_week'] >= gw - 3) & \
                (FWD_starters_df['match_week'] < gw)
                
    train_inner_mask = (FWD_starters_df['season'] < target_season) | \
                        ((FWD_starters_df['season'] == target_season) & (FWD_starters_df['match_week'] < gw - 3))

    X_train_inner = FWD_starters_df.loc[train_inner_mask, feature_columns]
    y_train_inner = FWD_starters_df.loc[train_inner_mask, 'target_points']

    X_val = FWD_starters_df.loc[val_mask, feature_columns]
    y_val = FWD_starters_df.loc[val_mask, 'target_points']
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        early_stopping_rounds=20,
        random_state=42
    )
    model.fit(
        X_train_inner, 
        y_train_inner, 
        eval_set=[(X_val, y_val)], 
        verbose=False
    )

    gw_metadata = FWD_starters_df.loc[test_mask, ["name", "season", "match_week", "target_points", "rolling_avg_points_last_5"]].copy()
    gw_metadata['predicted_points'] = model.predict(X_test)
    test_metadata_list.append(gw_metadata)

test_metadata = pd.concat(test_metadata_list, ignore_index=True)

eval_df = test_metadata.dropna(subset=['target_points', 'predicted_points', 'rolling_avg_points_last_5'])

model_test_vals = spearmanr(eval_df["target_points"], eval_df["predicted_points"])
baseline_vals = spearmanr(eval_df['target_points'], eval_df['rolling_avg_points_last_5'])

print(f"Model Correlation: {model_test_vals.statistic:.4f} (p={model_test_vals.pvalue:.4f})")
print(f"Baseline Correlation: {baseline_vals.statistic:.4f} (p={baseline_vals.pvalue:.4f})")