import time
import os
import pandas as pd
from understatapi import UnderstatClient
from concurrent.futures import ThreadPoolExecutor

csv_filename = "Data/csv_data/match_info_understat.csv"

columns = [
    "id", "fid", "h", "a", "date", "league_id", "season", "h_goals", "a_goals",
    "team_h", "team_a", "h_xg", "a_xg", "h_w", "h_d", "h_l", "league",
    "h_shot", "a_shot", "a_ppda", "h_ppda", "h_xa", "a_xa"
]

# Run for all seasons from 2014 to 2025 to create a clean, consistent dataset
seasons = ["2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
league_name = "EPL"
all_matches = []

os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

def fetch_roster_data(match_id):
    """Fetch roster data for a single match with retry logic."""
    with UnderstatClient() as understat_thread:
        for attempt in range(3):
            try:
                data = understat_thread.match(match_id).get_roster_data()
                return match_id, data
            except Exception as e:
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
                else:
                    print(f"Warning: Failed to fetch roster for match {match_id}: {e}")
                    return match_id, None

with UnderstatClient() as understat:
    for season in seasons:
        print(f"\n{'='*50}")
        print(f"Fetching {league_name} schedule for {season}")
        print(f"{'='*50}")
        
        # 1. Get the high-level match schedule
        schedule = understat.league(league=league_name).get_match_data(season=season)
        
        # 2. Build a lookup dictionary for PPDA via the League Team Endpoint
        print("Pre-fetching PPDA stats via the League Team Endpoint...")
        team_match_lookup = {}
        try:
            team_data = understat.league(league=league_name).get_team_data(season=season)
            for team_id, data in team_data.items():
                for match in data.get('history', []):
                    date = match.get('date')
                    ppda = match.get('ppda', {})
                    ppda_allowed = match.get('ppda_allowed', {})
                    
                    ppda_val = ppda.get('att', 0) / max(ppda.get('def', 1), 1)
                    ppda_allowed_val = ppda_allowed.get('att', 0) / max(ppda_allowed.get('def', 1), 1)
                    
                    team_match_lookup[(team_id, date)] = {
                        'ppda': ppda_val,
                        'ppda_allowed': ppda_allowed_val
                    }
        except Exception as e:
            print(f"Warning: Failed to fetch league team data: {e}")
            
        time.sleep(1) # Respect the rate limit
        
        # 3. Parallel fetch match roster data (which contains player shots and xA)
        played_match_ids = [m.get("id") for m in schedule if m.get("isResult")]
        print(f"Fetching rosters in parallel for {len(played_match_ids)} played matches...")
        roster_lookup = {}
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(fetch_roster_data, played_match_ids))
            
        for m_id, roster_data in results:
            if roster_data:
                roster_lookup[m_id] = roster_data
                
        # 4. Iterate through schedule and map values
        print(f"Compiling {len(schedule)} matches...")
        for match in schedule:
            match_id = match.get("id")
            is_played = match.get("isResult")
            
            # Calculate W/D/L mathematically
            try:
                h_g = int(match["goals"]["h"])
                a_g = int(match["goals"]["a"])
                h_w = 1 if h_g > a_g else 0
                h_d = 1 if h_g == a_g else 0
                h_l = 1 if h_g < a_g else 0
            except (TypeError, ValueError):
                h_w, h_d, h_l = 0, 0, 0
                
            row = {
                "id": match_id,
                "fid": match.get("fid", ""),
                "h": match["h"].get("id"),
                "a": match["a"].get("id"),
                "date": match.get("datetime"),
                "league_id": match.get("league_id", ""),
                "season": season,
                "h_goals": match["goals"]["h"],
                "a_goals": match["goals"]["a"],
                "team_h": match["h"]["title"],
                "team_a": match["a"]["title"],
                "h_xg": match["xG"]["h"],
                "a_xg": match["xG"]["a"],
                "h_w": h_w,
                "h_d": h_d,
                "h_l": h_l,
                "league": league_name,
            }
            
            if is_played:
                h_id = match["h"].get("id")
                date = match.get("datetime")
                
                # Map PPDA from our pre-fetched dictionary
                stats = team_match_lookup.get((h_id, date), {})
                h_ppda = stats.get('ppda', '')
                a_ppda = stats.get('ppda_allowed', '')
                
                # Map shots and xA from our pre-fetched roster lookup
                roster = roster_lookup.get(match_id, {})
                h_shot = ""
                a_shot = ""
                h_xa = ""
                a_xa = ""
                
                if roster:
                    try:
                        h_shot = sum(int(p.get("shots", 0)) for p in roster.get("h", {}).values())
                        a_shot = sum(int(p.get("shots", 0)) for p in roster.get("a", {}).values())
                        h_xa = sum(float(p.get("xA", 0.0)) for p in roster.get("h", {}).values())
                        a_xa = sum(float(p.get("xA", 0.0)) for p in roster.get("a", {}).values())
                    except Exception as e:
                        print(f"Error calculating roster totals for match {match_id}: {e}")
                
                row.update({
                    "h_ppda": h_ppda,
                    "a_ppda": a_ppda,
                    "h_shot": h_shot,
                    "a_shot": a_shot,
                    "h_xa": h_xa,
                    "a_xa": a_xa
                })
            else:
                row.update({
                    "h_ppda": "",
                    "a_ppda": "",
                    "h_shot": "",
                    "a_shot": "",
                    "h_xa": "",
                    "a_xa": ""
                })

            all_matches.append(row)

# Save the final data with headers
df_new = pd.DataFrame(all_matches, columns=columns)
df_new.to_csv(csv_filename, index=False)

print(f"\nDone! Successfully wrote {len(df_new)} matches to {csv_filename}.")