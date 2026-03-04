"""
Collect 2025-26 mid-season player stats and append to player_stats_by_season.csv.
Uses the same LeagueLeaders endpoint as nba_player_data_collector_v2.py.
"""

import pandas as pd
import requests
import time
import warnings
warnings.filterwarnings('ignore')

OUTPUT_FILE = 'player_stats_by_season.csv'
SEASON = '2025-26'

COUNTING_STATS = ['FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
                  'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

REQUEST_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/121.0.0.0 Safari/537.36'),
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'Connection': 'keep-alive',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
}


def main():
    # Check if already collected
    if pd.io.common.file_exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE, usecols=['SEASON'])
        if SEASON in existing['SEASON'].unique():
            print(f'{SEASON} already in {OUTPUT_FILE} — nothing to do.')
            return

    print(f'Fetching {SEASON} via LeagueLeaders...', flush=True)

    url = (
        'https://stats.nba.com/stats/leagueleaders'
        f'?LeagueID=00&PerMode=Totals&Scope=S'
        f'&Season={SEASON}&SeasonType=Regular+Season&StatCategory=MIN'
    )

    session = requests.Session()
    for attempt in range(3):
        try:
            r = session.get(url, headers=REQUEST_HEADERS, timeout=30)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            print(f'  Attempt {attempt+1} failed: {str(e)[:80]}')
            if attempt < 2:
                time.sleep(5)
            else:
                print('  All attempts failed.')
                return

    result_set = data.get('resultSet', {})
    cols = result_set.get('headers', [])
    rows = result_set.get('rowSet', [])
    if not rows:
        print('No data returned.')
        return

    df = pd.DataFrame(rows, columns=cols)
    print(f'  Raw rows: {len(df)}')

    df = df.rename(columns={
        'PLAYER': 'PLAYER_NAME',
        'TEAM':   'TEAM_ABBREVIATION',
        'MIN':    'TOTAL_MIN',
    })

    gp_safe = df['GP'].clip(lower=1)
    for col in COUNTING_STATS:
        if col in df.columns:
            df[col] = df[col] / gp_safe
    df['MIN'] = df['TOTAL_MIN'] / gp_safe

    df['SEASON']      = SEASON
    df['SEASON_YEAR'] = 2026
    df['birth_year']  = 0

    for col in ['AGE', 'PIE', 'NET_RATING', 'OFF_RATING', 'DEF_RATING',
                'TS_PCT', 'USG_PCT', 'AST_PCT', 'REB_PCT', 'OREB_PCT', 'DREB_PCT']:
        df[col] = None

    df['OVERALL_PICK'] = 999
    df['DRAFT_YEAR']   = 0

    keep = [
        'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION',
        'SEASON', 'SEASON_YEAR', 'AGE', 'birth_year',
        'GP', 'MIN', 'TOTAL_MIN',
        'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
        'FG_PCT', 'FG3_PCT', 'FT_PCT',
        'PIE', 'NET_RATING', 'OFF_RATING', 'DEF_RATING',
        'TS_PCT', 'USG_PCT', 'AST_PCT', 'REB_PCT', 'OREB_PCT', 'DREB_PCT',
        'OVERALL_PICK', 'DRAFT_YEAR',
    ]
    df = df[[col for col in keep if col in df.columns]].copy()

    # Match column order to existing file
    existing_df = pd.read_csv(OUTPUT_FILE, nrows=0)
    existing_cols = existing_df.columns.tolist()
    for col in existing_cols:
        if col not in df.columns:
            df[col] = None
    df = df[existing_cols]

    df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
    print(f'Appended {len(df)} rows for {SEASON} to {OUTPUT_FILE}')

    # Report max GP so we know how far into the season we are
    max_gp = df['GP'].max()
    print(f'Max GP in dataset: {max_gp} (out of 82 in a full season)')
    print(f'Season is ~{max_gp/82*100:.0f}% complete')


if __name__ == '__main__':
    main()
