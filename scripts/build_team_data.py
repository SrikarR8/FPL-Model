import glob
import os
import pandas as pd

def build_team_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(base_dir, 'RawData', 'csv_data')
    model_data_dir = os.path.join(base_dir, 'Models', 'ModelData')
    os.makedirs(model_data_dir, exist_ok=True)

    tables_file = os.path.join(raw_data_dir, 'tables.csv')
    understat_file = os.path.join(raw_data_dir, 'match_info_understat.csv')
    mtl_file = os.path.join(raw_data_dir, 'master_team_list.csv')

    tables_df = pd.read_csv(tables_file) if os.path.exists(tables_file) else pd.DataFrame()
    understat_df = pd.read_csv(understat_file) if os.path.exists(understat_file) else pd.DataFrame()
    mtl_df = pd.read_csv(mtl_file) if os.path.exists(mtl_file) else pd.DataFrame()

    # Name mapping to align FPL team names with tables.csv and understat names
    name_map = {
        'Man City': 'Manchester City',
        'Man Utd': 'Manchester United',
        'Newcastle': 'Newcastle United',
        "Nott'm Forest": 'Nottingham Forest',
        'Sheffield Utd': 'Sheffield United',
        'Spurs': 'Tottenham',
        'West Brom': 'West Bromwich Albion',
        'Wolves': 'Wolverhampton Wanderers',
        'Ipswich Town': 'Ipswich',
        'Hull City': 'Hull',
    }

    # Standard output columns order
    standard_cols = [
        'season', 'code', 'event', 'finished', 'finished_provisional', 'id',
        'kickoff_time', 'minutes', 'provisional_start_time', 'started',
        'team_a', 'team_a_score', 'team_h', 'team_h_score', 'stats',
        'team_h_difficulty', 'team_a_difficulty', 'pulse_id',
        'home_pos', 'away_pos', 'h_ppda', 'a_ppda',
        'h_xg', 'a_xg', 'h_xga', 'a_xga'
    ]

    season_paths = [
        s for s in sorted(glob.glob(os.path.join(raw_data_dir, '20*')))
        if os.path.exists(os.path.join(s, 'fixtures.csv')) and os.path.basename(s) >= '2018-19'
    ]

    all_fixture_dfs = []

    for s_path in season_paths:
        season_str = os.path.basename(s_path) # e.g. 2018-19
        season_slash = season_str.replace('-', '/') # 2018/19
        season_year = int(season_str[:4]) # 2018

        # 1. Build team ID to standard team name lookup dictionary for this season
        id_to_name = {}
        if not mtl_df.empty and season_str in mtl_df['season'].unique():
            sub = mtl_df[mtl_df['season'] == season_str]
            for _, r in sub.iterrows():
                raw_name = r['team_name']
                id_to_name[r['team']] = name_map.get(raw_name, raw_name)

        teams_csv_p = os.path.join(s_path, 'teams.csv')
        if os.path.exists(teams_csv_p):
            tdf = pd.read_csv(teams_csv_p)
            for _, r in tdf.iterrows():
                raw_name = r['name']
                id_to_name[r['id']] = name_map.get(raw_name, raw_name)

        # 2. Load fixtures
        fix_df = pd.read_csv(os.path.join(s_path, 'fixtures.csv'))
        fix_df['season'] = season_str

        # Filter tables data for this season
        t_season = tables_df[tables_df['season'] == season_slash] if not tables_df.empty else pd.DataFrame()
        available_gws = t_season['week'].unique() if not t_season.empty else []

        # Filter understat data for this season
        u_season = understat_df[understat_df['season'] == season_year] if not understat_df.empty else pd.DataFrame()

        home_pos_list = []
        away_pos_list = []
        h_ppda_list = []
        a_ppda_list = []
        h_xg_list = []
        a_xg_list = []
        h_xga_list = []
        a_xga_list = []

        for _, r in fix_df.iterrows():
            h_id = r['team_h']
            a_id = r['team_a']
            h_name = id_to_name.get(h_id)
            a_name = id_to_name.get(a_id)

            # --- Lookup home_pos & away_pos from tables.csv ---
            h_pos, a_pos = None, None
            if pd.notna(r['event']) and len(available_gws) > 0:
                event_num = int(r['event'])
                # Handle 2019-20 COVID restart gameweeks (39..47 -> GW30..GW38)
                if season_str == '2019-20' and event_num >= 39:
                    gw_num = event_num - 9
                else:
                    gw_num = event_num

                max_gw = max([int(w.replace('GW', '')) for w in available_gws])
                target_gw = f"GW{min(gw_num, max_gw)}"

                t_gw = t_season[t_season['week'] == target_gw]
                if not t_gw.empty:
                    hp_vals = t_gw[t_gw['team'] == h_name]['position'].values
                    ap_vals = t_gw[t_gw['team'] == a_name]['position'].values
                    if len(hp_vals) > 0: h_pos = int(hp_vals[0])
                    if len(ap_vals) > 0: a_pos = int(ap_vals[0])

            home_pos_list.append(h_pos)
            away_pos_list.append(a_pos)

            # --- Lookup h_ppda, a_ppda, h_xg, a_xg from match_info_understat.csv ---
            h_ppda, a_ppda = None, None
            h_xg, a_xg = None, None
            h_xga, a_xga = None, None

            if not u_season.empty and h_name and a_name:
                u_match = u_season[(u_season['team_h'] == h_name) & (u_season['team_a'] == a_name)]
                if len(u_match) == 1:
                    h_ppda = u_match['h_ppda'].values[0]
                    a_ppda = u_match['a_ppda'].values[0]
                    h_xg = u_match['h_xg'].values[0]
                    a_xg = u_match['a_xg'].values[0]
                    h_xga = a_xg # Home team xGA is Away team xG
                    a_xga = h_xg # Away team xGA is Home team xG

            h_ppda_list.append(h_ppda)
            a_ppda_list.append(a_ppda)
            h_xg_list.append(h_xg)
            a_xg_list.append(a_xg)
            h_xga_list.append(h_xga)
            a_xga_list.append(a_xga)

        fix_df['home_pos'] = home_pos_list
        fix_df['away_pos'] = away_pos_list
        fix_df['h_ppda'] = h_ppda_list
        fix_df['a_ppda'] = a_ppda_list
        fix_df['h_xg'] = h_xg_list
        fix_df['a_xg'] = a_xg_list
        fix_df['h_xga'] = h_xga_list
        fix_df['a_xga'] = a_xga_list

        # Ensure all standard columns exist
        for col in standard_cols:
            if col not in fix_df.columns:
                fix_df[col] = None

        all_fixture_dfs.append(fix_df[standard_cols])

    combined_df = pd.concat(all_fixture_dfs, ignore_index=True)
    out_file = os.path.join(model_data_dir, 'teamData.csv')
    combined_df.to_csv(out_file, index=False)
    print(f"Successfully saved {len(combined_df)} rows to {out_file}")
    return combined_df

if __name__ == '__main__':
    build_team_data()
