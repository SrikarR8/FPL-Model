import glob
import os
import re
import unicodedata
import pandas as pd

# Comprehensive alias map for player name variations between FPL and Understat
ALIAS_MAP = {
    # Gabriel family
    'gabriel magalhaes': 'gabriel',
    'gabriel dos santos magalhaes': 'gabriel',
    'gabriel fernando de jesus': 'gabriel jesus',
    'gabriel teodoro martinelli silva': 'gabriel martinelli',
    'gabriel martinelli silva': 'gabriel martinelli',
    # Single-name / Portuguese / Spanish / Latin full names
    'heung min son': 'son heung min',
    'rodrigo hernandez': 'rodri',
    'rodrigo hernandez cascante': 'rodri',
    'emerson aparecido leite de souza junior': 'emerson',
    'thiago emiliano da silva': 'thiago silva',
    'diogo teixeira da silva': 'diogo jota',
    'bruno guimaraes rodriguez moura': 'bruno guimaraes',
    'richarlison de andrade': 'richarlison',
    'frederico rodrigues de paula santos': 'fred',
    'fabio henrique tavares': 'fabinho',
    'carlos enrique casimiro': 'casemiro',
    'antony matheus dos santos': 'antony',
    'ederson santana de moraes': 'ederson',
    'jose malheiro de sa': 'jose sa',
    'alisson ramses becker': 'alisson',
    'joao pedro cavaco cancelo': 'joao cancelo',
    'bruno miguel borges fernandes': 'bruno fernandes',
    'joao filipe iria santos moutinho': 'joao moutinho',
    'raphael dias belloli': 'raphinha',
    'oriol romeu vidal': 'oriol romeu',
    'bernardo mota veiga de carvalho e silva': 'bernardo silva',
    'joao pedro junqueira de jesus': 'joao pedro',
    'kepa arrizabalaga': 'kepa',
    'jonny castro otto': 'jonny',
    'pedro lomba neto': 'pedro neto',
    'matheus luiz nunes': 'matheus nunes',
    'matheus santos carneiro da cunha': 'matheus cunha',
    'joao maria lobo alves palhinha goncalves': 'joao palhinha',
    'nelson cabral semedo': 'nelson semedo',
    'ruben diogo da silva neves': 'ruben neves',
    'ruben santos gato alves dias': 'ruben dias',
    'adama traore diarra': 'adama traore',
    'lucas rodrigues moura da silva': 'lucas moura',
}

def normalize_str(s):
    if not isinstance(s, str):
        return ''
    s = re.sub(r'_[0-9]+$', '', s)
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8')
    s = s.replace('_', ' ').replace('-', ' ').lower().strip()
    return ALIAS_MAP.get(s, s)

def load_understat_data(raw_data_dir):
    all_u_files = glob.glob(os.path.join(raw_data_dir, '*', 'understat', '*_[0-9]*.csv'))
    if not all_u_files:
        return pd.DataFrame()

    dfs = []
    for f in all_u_files:
        fname = os.path.basename(f)
        p_name = '_'.join(fname.split('_')[:-1])
        df = pd.read_csv(f)
        df['u_clean_name'] = normalize_str(p_name)
        df['match_date'] = pd.to_datetime(df['date']).dt.date
        dfs.append(df[['u_clean_name', 'match_date', 'xG', 'xA']])

    big_u = pd.concat(dfs, ignore_index=True)
    big_u.drop_duplicates(subset=['match_date', 'u_clean_name'], inplace=True)
    return big_u

def build_player_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(base_dir, 'RawData', 'csv_data')
    model_data_dir = os.path.join(base_dir, 'Models', 'ModelData')
    os.makedirs(model_data_dir, exist_ok=True)

    print("Loading Understat player match dataset for 2018-2022 xG/xA backfilling...")
    understat_df = load_understat_data(raw_data_dir)
    print(f"Loaded {len(understat_df)} unique Understat player-match records.")

    season_paths = [
        s for s in sorted(glob.glob(os.path.join(raw_data_dir, '20*')))
        if os.path.exists(os.path.join(s, 'gws', 'merged_gw.csv')) and os.path.basename(s) >= '2018-19'
    ]

    all_dfs = []

    for s_path in season_paths:
        season_str = os.path.basename(s_path)
        mgw_file = os.path.join(s_path, 'gws', 'merged_gw.csv')

        df = pd.read_csv(mgw_file, encoding='latin-1', low_memory=False)
        df['season'] = season_str

        # Ensure was_home boolean is present
        if 'was_home' in df.columns:
            df['was_home'] = df['was_home'].astype(bool)
        else:
            df['was_home'] = True

        # Add match date and clean name for joining
        df['u_clean_name'] = df['name'].apply(normalize_str)
        df['match_date'] = pd.to_datetime(df['kickoff_time']).dt.date

        # Determine xG and xA
        if 'expected_goals' in df.columns:
            df['xG'] = pd.to_numeric(df['expected_goals'], errors='coerce').fillna(0.0)
        else:
            df['xG'] = 0.0

        if 'expected_assists' in df.columns:
            df['xA'] = pd.to_numeric(df['expected_assists'], errors='coerce').fillna(0.0)
        else:
            df['xA'] = 0.0

        # If xG / xA are missing (e.g. 2018-19 to 2021-22), merge from Understat
        if not understat_df.empty and (season_str < '2022-23' or df['xG'].sum() == 0):
            merged = pd.merge(
                df,
                understat_df,
                on=['u_clean_name', 'match_date'],
                how='left',
                suffixes=('', '_understat')
            )
            df['xG'] = merged['xG_understat'].combine_first(df['xG']).fillna(0.0)
            df['xA'] = merged['xA_understat'].combine_first(df['xA']).fillna(0.0)

        df['Goals'] = pd.to_numeric(df['goals_scored'], errors='coerce').fillna(0).astype(int)
        df['Assists'] = pd.to_numeric(df['assists'], errors='coerce').fillna(0).astype(int)
        df['MinutesPlayed'] = pd.to_numeric(df['minutes'], errors='coerce').fillna(0).astype(int)
        df['FPL_Points'] = pd.to_numeric(df['total_points'], errors='coerce').fillna(0).astype(int)
        df['YellowCards'] = pd.to_numeric(df['yellow_cards'], errors='coerce').fillna(0).astype(int)
        df['RedCards'] = pd.to_numeric(df['red_cards'], errors='coerce').fillna(0).astype(int)
        df['BonusPoints'] = pd.to_numeric(df['bonus'], errors='coerce').fillna(0).astype(int)

        # New_Cost (convert FPL value in tenths to millions, e.g. 55 -> 5.5)
        df['New_Cost'] = (pd.to_numeric(df['value'], errors='coerce').fillna(0) / 10.0)

        # 5_FPL_Points boolean metric (True if FPL_Points >= 5, else False)
        df['5_FPL_Points'] = df['FPL_Points'] >= 5

        if 'position' not in df.columns:
            df['position'] = 'UNK'

        output_cols = [
            'season', 'GW', 'name', 'position', 'element', 'fixture', 'was_home',
            'xG', 'xA', 'Goals', 'Assists', 'MinutesPlayed',
            'FPL_Points', 'YellowCards', 'RedCards', 'BonusPoints',
            'New_Cost', '5_FPL_Points'
        ]

        all_dfs.append(df[output_cols])

    # Build GW1 placeholder rows for upcoming unplayed seasons (e.g. 2026-27)
    upcoming_season_paths = [
        s for s in sorted(glob.glob(os.path.join(raw_data_dir, '20*')))
        if os.path.exists(os.path.join(s, 'cleaned_players.csv')) and os.path.exists(os.path.join(s, 'fixtures.csv'))
        and not os.path.exists(os.path.join(s, 'gws', 'merged_gw.csv'))
    ]

    for u_path in upcoming_season_paths:
        season_str = os.path.basename(u_path)
        players_csv = os.path.join(u_path, 'cleaned_players.csv')
        fixtures_csv = os.path.join(u_path, 'fixtures.csv')
        teams_csv = os.path.join(u_path, 'teams.csv')

        df_players = pd.read_csv(players_csv, encoding='latin-1')
        df_fixtures = pd.read_csv(fixtures_csv, encoding='latin-1')

        # Filter for GW 1 fixtures
        gw1_fixtures = df_fixtures[df_fixtures['event'] == 1].copy()

        # Map element_type to position string (FPL standard: 1: GKP, 2: DEF, 3: MID, 4: FWD)
        pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}

        # Map teams if teams.csv exists
        team_id_map = {}
        if os.path.exists(teams_csv):
            df_teams = pd.read_csv(teams_csv, encoding='latin-1')
            team_id_map = dict(zip(df_teams['id'], df_teams['name']))

        # Build player GW1 rows
        gw1_rows = []
        for idx, row in df_players.iterrows():
            p_name = f"{row['first_name']} {row['second_name']}"
            p_team = row.get('team', None)
            pos_str = pos_map.get(row.get('element_type', 0), 'UNK')

            # Find matching GW1 fixture for this player's team
            was_home = True
            fixture_id = 1
            if p_team:
                home_fix = gw1_fixtures[gw1_fixtures['team_h'] == p_team]
                away_fix = gw1_fixtures[gw1_fixtures['team_a'] == p_team]
                if not home_fix.empty:
                    was_home = True
                    fixture_id = home_fix.iloc[0]['id']
                elif not away_fix.empty:
                    was_home = False
                    fixture_id = away_fix.iloc[0]['id']

            cost_val = float(row.get('now_cost', 50)) / 10.0

            gw1_rows.append({
                'season': season_str,
                'GW': 1,
                'name': p_name,
                'position': pos_str,
                'element': idx,
                'fixture': fixture_id,
                'was_home': was_home,
                'xG': 0.0,
                'xA': 0.0,
                'Goals': 0,
                'Assists': 0,
                'MinutesPlayed': 90,  # Set default 90 so loadCols minutes=40 filter includes them
                'FPL_Points': 0,
                'YellowCards': 0,
                'RedCards': 0,
                'BonusPoints': 0,
                'New_Cost': cost_val,
                '5_FPL_Points': False
            })

        df_gw1 = pd.DataFrame(gw1_rows)
        all_dfs.append(df_gw1[output_cols])
        print(f"Created {len(df_gw1)} GW1 placeholder rows for upcoming season {season_str}.")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    out_file = os.path.join(model_data_dir, 'playerData.csv')
    combined_df.to_csv(out_file, index=False)
    print(f"Successfully saved {len(combined_df)} rows to {out_file}")
    return combined_df

if __name__ == '__main__':
    build_player_data()
