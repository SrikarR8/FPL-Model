import os
from datetime import datetime, timedelta
import pandas as pd

# Define paths
csv_input = "Data/csv_data/match_info_understat.csv"
csv_output = "Data/csv_data/tables.csv"

# Load the matches
df = pd.read_csv(csv_input)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Build a lookup of teams by season
season_teams = {}
all_seasons = sorted(df['season'].unique())
for s in all_seasons:
    season_df = df[df['season'] == s]
    teams = set(season_df['team_h']).union(set(season_df['team_a']))
    season_teams[s] = teams

# We want snapshots from 2018/19 (season 2018) to 2025/26 (season 2025)
target_seasons = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
table_records = []

def get_team_matches_as_of(team, date, season, is_gw1=False):
    if is_gw1:
        prev_season = season - 1
        was_in_prev = (prev_season in season_teams) and (team in season_teams[prev_season])
        if was_in_prev:
            # Filter matches involving team in the previous season S-1
            team_df = df[((df['team_h'] == team) | (df['team_a'] == team)) & (df['season'] == prev_season)]
            return team_df
        else:
            # Promoted team - return empty DataFrame
            return pd.DataFrame(columns=df.columns)
    else:
        # Season-wide calculation: filter matches in the current season up to date D
        team_df = df[((df['team_h'] == team) | (df['team_a'] == team)) & 
                     (df['season'] == season) & 
                     (df['date'] <= date)]
        return team_df

def compute_team_stats(team, team_matches):
    played = len(team_matches)
    wins = 0
    draws = 0
    losses = 0
    goals_for = 0
    goals_against = 0
    points = 0
    
    for idx, row in team_matches.iterrows():
        is_home = row['team_h'] == team
        if is_home:
            gf = int(row['h_goals'])
            ga = int(row['a_goals'])
            w = int(row['h_w'])
            d = int(row['h_d'])
            l = int(row['h_l'])
        else:
            gf = int(row['a_goals'])
            ga = int(row['h_goals'])
            w = int(row['h_l'])
            d = int(row['h_d'])
            l = int(row['h_w'])
            
        goals_for += gf
        goals_against += ga
        wins += w
        draws += d
        losses += l
        points += (3 * w + 1 * d)
        
    gd = goals_for - goals_against
    ppg = points / played if played > 0 else 0.0
    
    return {
        'played': played,
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'goals_for': goals_for,
        'goals_against': goals_against,
        'goal_difference': gd,
        'points': points,
        'ppg': ppg
    }

for season in target_seasons:
    season_str = f"{season}/{str(season+1)[2:]}"
    print(f"Generating tables for season {season_str}...")
    
    # 1. Get matches in the current season
    season_matches = df[df['season'] == season]
    if len(season_matches) == 0:
        continue
        
    active_teams = sorted(list(season_teams[season]))
    
    # 2. Get unique weeks (Sunday dates) of this season
    season_matches_dates = season_matches['date']
    first_match_date = season_matches_dates.min()
    
    # We find all unique ISO weeks in the season matches
    unique_weeks = sorted(list(set(d.isocalendar()[:2] for d in season_matches_dates)))
    
    # Convert ISO weeks to Sunday end dates
    sundays = []
    for yr, wk in unique_weeks:
        sundays.append(datetime.fromisocalendar(yr, wk, 7))
    sundays = sorted(list(set(sundays)))
    
    # 3. Snapshot 0: GW1 (Start of the season, before any matches of this season are played)
    # We set the snapshot date to the end of the day before the first match of the season
    d0 = first_match_date - timedelta(days=1)
    d0_end = datetime(d0.year, d0.month, d0.day, 23, 59, 59)
    
    gw1_rows = []
    for team in active_teams:
        team_matches = get_team_matches_as_of(team, d0_end, season, is_gw1=True)
        stats = compute_team_stats(team, team_matches)
        stats['team'] = team
        stats['season'] = season_str
        stats['week'] = f"GW1"
        gw1_rows.append(stats)
        
    # Sort GW1 table by PPG, Goal Difference, Goals For
    # Promoted teams will have 0 PPG and sort to the bottom
    gw1_rows.sort(key=lambda x: (x['ppg'], x['goal_difference'], x['goals_for']), reverse=True)
    for pos, row in enumerate(gw1_rows, 1):
        row['position'] = pos
        table_records.append(row)
        
    # 4. Snapshots for each subsequent week
    for idx, sunday in enumerate(sundays, 2):
        sunday_end = datetime(sunday.year, sunday.month, sunday.day, 23, 59, 59)
        gw_label = f"GW{idx}"
        
        gw_rows = []
        for team in active_teams:
            team_matches = get_team_matches_as_of(team, sunday_end, season, is_gw1=False)
            stats = compute_team_stats(team, team_matches)
            stats['team'] = team
            stats['season'] = season_str
            stats['week'] = gw_label
            gw_rows.append(stats)
            
        gw_rows.sort(key=lambda x: (x['ppg'], x['goal_difference'], x['goals_for']), reverse=True)
        for pos, row in enumerate(gw_rows, 1):
            row['position'] = pos
            table_records.append(row)

# Save to CSV
os.makedirs(os.path.dirname(csv_output), exist_ok=True)
out_df = pd.DataFrame(table_records)
# Reorder columns
columns_order = [
    'season', 'week', 'position', 'team', 'played', 'wins', 'draws', 'losses',
    'goals_for', 'goals_against', 'goal_difference', 'points', 'ppg'
]
out_df = out_df[columns_order]
out_df.to_csv(csv_output, index=False)

print(f"\nDone! Successfully wrote {len(out_df)} table records to {csv_output}.")
