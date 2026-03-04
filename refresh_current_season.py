"""
Refresh 2025-26 Season Data
Fetches fresh team and player stats for the current NBA season and
updates both nba_team_stats_clean.csv and player_stats_by_season.csv.

Run this before re-running the pipeline mid-season to pick up new games.

Author: GY Mehta
Date: February 2026
"""

import pandas as pd
import time
import warnings
from nba_api.stats.endpoints import leaguedashteamstats, leaguedashplayerstats
from nba_api.stats.library import http as nba_http

warnings.filterwarnings('ignore')

SEASON = '2025-26'
SEASON_YEAR = 2026  # end year of the season

# Custom headers required by some NBA API endpoints
nba_http.NBAStatsHTTP.headers['x-nba-stats-origin'] = 'stats'
nba_http.NBAStatsHTTP.headers['x-nba-stats-token'] = 'true'

TEAM_CLEAN_FILE   = 'nba_team_stats_clean.csv'
PLAYER_STATS_FILE = 'player_stats_by_season.csv'


# ── Helpers ──────────────────────────────────────────────────────────────────

def fetch_with_retry(fetch_fn, label='', max_retries=3, retry_sleep=5):
    """Execute an API fetch function with retry logic."""
    for attempt in range(max_retries):
        try:
            return fetch_fn()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f'  Retry {attempt + 1}/{max_retries - 1} ({label}): {str(e)[:80]}')
                time.sleep(retry_sleep)
            else:
                print(f'  Failed after {max_retries} attempts ({label}): {str(e)[:80]}')
                return None


# ── Team Stats ────────────────────────────────────────────────────────────────

def fetch_team_stats():
    """
    Fetch 2025-26 team stats via 4 LeagueDashTeamStats calls.
    Returns a cleaned DataFrame matching nba_team_stats_clean.csv columns.
    """
    print('Fetching team Base stats ...')
    base_df = fetch_with_retry(
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=SEASON,
            measure_type_detailed_defense='Base',
            per_mode_detailed='PerGame',
            season_type_all_star='Regular Season',
            timeout=120
        ).get_data_frames()[0],
        label='Team Base'
    )
    if base_df is None or base_df.empty:
        print('  ERROR: could not fetch team Base stats.')
        return None
    time.sleep(0.6)

    print('Fetching team Advanced stats ...')
    adv_df = fetch_with_retry(
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=SEASON,
            measure_type_detailed_defense='Advanced',
            per_mode_detailed='PerGame',
            season_type_all_star='Regular Season',
            timeout=120
        ).get_data_frames()[0],
        label='Team Advanced'
    )
    time.sleep(0.6)

    print('Fetching team Four Factors stats ...')
    ff_df = fetch_with_retry(
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=SEASON,
            measure_type_detailed_defense='Four Factors',
            per_mode_detailed='PerGame',
            season_type_all_star='Regular Season',
            timeout=120
        ).get_data_frames()[0],
        label='Team Four Factors'
    )
    time.sleep(0.6)

    print('Fetching team Opponent stats ...')
    opp_df = fetch_with_retry(
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=SEASON,
            measure_type_detailed_defense='Opponent',
            per_mode_detailed='PerGame',
            season_type_all_star='Regular Season',
            timeout=120
        ).get_data_frames()[0],
        label='Team Opponent'
    )
    time.sleep(0.6)

    # ── Merge ──
    merged = base_df.copy()

    if adv_df is not None and not adv_df.empty:
        adv_keep = ['TEAM_ID', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE',
                    'PIE', 'AST_PCT', 'AST_TO', 'AST_RATIO', 'OREB_PCT', 'DREB_PCT',
                    'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT']
        adv_sub = adv_df[[c for c in adv_keep if c in adv_df.columns]]
        merged = merged.merge(adv_sub, on='TEAM_ID', how='left', suffixes=('', '_ADV'))

    if ff_df is not None and not ff_df.empty:
        ff_keep = ['TEAM_ID', 'EFG_PCT', 'FTA_RATE', 'TM_TOV_PCT', 'OREB_PCT']
        ff_sub = ff_df[[c for c in ff_keep if c in ff_df.columns]]
        merged = merged.merge(ff_sub, on='TEAM_ID', how='left', suffixes=('', '_FF'))

    if opp_df is not None and not opp_df.empty:
        opp_prefixed = opp_df.add_prefix('OPP_')
        opp_prefixed = opp_prefixed.rename(columns={'OPP_TEAM_ID': 'TEAM_ID'})
        merged = merged.merge(opp_prefixed, on='TEAM_ID', how='left')

    # ── Rename to match clean CSV columns ──
    col_map = {
        'TEAM_NAME':    'Team',
        'OFF_RATING':   'ORtg',
        'DEF_RATING':   'DRtg',
        'NET_RATING':   'NRtg',
        'PACE':         'Pace',
        'FG_PCT':       'FG%',
        'FG3_PCT':      '3P%',
        'FT_PCT':       'FT%',
        'FG3A':         '3PA',
        'FG3M':         'FG3M',
        'AST_PCT':      'AST%',
        'OREB_PCT':     'ORB%',
        'DREB_PCT':     'DRB%',
        'TM_TOV_PCT':   'TOV%',
        'EFG_PCT':      'eFG%',
        'TS_PCT':       'TS%',
        'FTA_RATE':     'FTA_RATE',
        'AST_TO':       'AST_TO',
        'AST_RATIO':    'AST_RATIO',
        'REB_PCT':      'REB_PCT',
        'PIE':          'PIE',
        'PTS':          'PTS',
        'OPP_PTS':      'OPP_OPP_PTS',
        'OPP_FG_PCT':   'OPP_OPP_FG_PCT',
        'OPP_FG3_PCT':  'OPP_OPP_3P_PCT',
    }
    merged = merged.rename(columns=col_map)

    # ── Derived columns ──
    if 'W' in merged.columns and 'L' in merged.columns:
        merged['WIN_PCT'] = merged['W'] / (merged['W'] + merged['L'])
    elif 'W_PCT' in merged.columns:
        merged['WIN_PCT'] = merged['W_PCT']

    merged['SEASON'] = SEASON
    merged['YEAR']   = SEASON_YEAR
    merged['ERA']    = '2020s'

    # ── Keep only clean CSV columns ──
    clean_cols = [
        'Team', 'SEASON', 'YEAR', 'ERA', 'GP', 'W', 'L', 'WIN_PCT',
        'ORtg', 'DRtg', 'NRtg', 'eFG%', 'TS%', '3PA', '3P%', 'FG3M',
        'Pace', 'AST%', 'AST_TO', 'AST_RATIO', 'TOV%', 'ORB%', 'DRB%',
        'REB_PCT', 'FTA_RATE', 'PIE', 'PTS', 'FG%', 'FT%',
        'OPP_OPP_PTS', 'OPP_OPP_FG_PCT',
    ]
    result = merged[[c for c in clean_cols if c in merged.columns]].copy()

    # Add any missing columns as NaN
    for c in clean_cols:
        if c not in result.columns:
            result[c] = None

    print(f'  ✓ {len(result)} teams fetched')
    return result[clean_cols]


def update_team_clean_csv(new_df):
    """Replace 2025-26 rows in nba_team_stats_clean.csv."""
    try:
        existing = pd.read_csv(TEAM_CLEAN_FILE)
    except FileNotFoundError:
        print(f'  ERROR: {TEAM_CLEAN_FILE} not found — run nba_data_collector.py first.')
        return

    before = len(existing[existing['SEASON'] == SEASON])
    existing = existing[existing['SEASON'] != SEASON].copy()
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values(['SEASON', 'Team']).reset_index(drop=True)
    combined.to_csv(TEAM_CLEAN_FILE, index=False)
    print(f'  ✓ {TEAM_CLEAN_FILE} updated  '
          f'(replaced {before} old rows → {len(new_df)} fresh rows;  '
          f'total {len(combined)} rows)')


# ── Player Stats ──────────────────────────────────────────────────────────────

def fetch_player_stats():
    """
    Fetch 2025-26 player Base + Advanced stats.
    Returns merged DataFrame matching player_stats_by_season.csv columns.
    """
    print('Fetching player Base stats ...')
    base_df = fetch_with_retry(
        lambda: leaguedashplayerstats.LeagueDashPlayerStats(
            season=SEASON,
            measure_type_detailed_defense='Base',
            per_mode_detailed='PerGame',
            season_type_all_star='Regular Season',
            timeout=120
        ).get_data_frames()[0],
        label='Player Base'
    )
    if base_df is None or base_df.empty:
        print('  ERROR: could not fetch player Base stats.')
        return None
    time.sleep(0.6)

    print('Fetching player Advanced stats ...')
    adv_df = fetch_with_retry(
        lambda: leaguedashplayerstats.LeagueDashPlayerStats(
            season=SEASON,
            measure_type_detailed_defense='Advanced',
            per_mode_detailed='PerGame',
            season_type_all_star='Regular Season',
            timeout=120
        ).get_data_frames()[0],
        label='Player Advanced'
    )
    time.sleep(0.6)

    # Drop TOT aggregate rows for traded players
    base_df = base_df[base_df['TEAM_ABBREVIATION'] != 'TOT'].copy()

    # Normalise age column name
    age_col = 'PLAYER_AGE' if 'PLAYER_AGE' in base_df.columns else 'AGE'
    base_keep = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION',
                 age_col, 'GP', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
                 'FG_PCT', 'FG3_PCT', 'FT_PCT']
    base_df = base_df[[c for c in base_keep if c in base_df.columns]].copy()
    base_df = base_df.rename(columns={'PLAYER_AGE': 'AGE'})
    if 'AGE' not in base_df.columns:
        base_df['AGE'] = None

    # Merge Advanced
    if adv_df is not None and not adv_df.empty:
        adv_df = adv_df[adv_df['TEAM_ABBREVIATION'] != 'TOT'].copy()
        adv_keep = ['PLAYER_ID', 'TEAM_ID', 'PIE', 'NET_RATING', 'OFF_RATING',
                    'DEF_RATING', 'TS_PCT', 'USG_PCT', 'AST_PCT', 'REB_PCT',
                    'OREB_PCT', 'DREB_PCT']
        adv_df = adv_df[[c for c in adv_keep if c in adv_df.columns]].copy()
        merged = base_df.merge(adv_df, on=['PLAYER_ID', 'TEAM_ID'], how='left')
    else:
        merged = base_df.copy()
        for col in ['PIE', 'NET_RATING', 'OFF_RATING', 'DEF_RATING',
                    'TS_PCT', 'USG_PCT', 'AST_PCT', 'REB_PCT', 'OREB_PCT', 'DREB_PCT']:
            merged[col] = None
        print('  WARNING: Advanced stats unavailable — those columns will be NaN.')

    # Derived columns
    merged['SEASON']      = SEASON
    merged['SEASON_YEAR'] = SEASON_YEAR
    merged['TOTAL_MIN']   = merged['MIN'] * merged['GP']
    merged['birth_year']  = (SEASON_YEAR - merged['AGE'].fillna(0)).astype(int)

    print(f'  ✓ {len(merged)} player-team stints fetched  '
          f'(PIE non-null: {merged["PIE"].notna().sum()})')
    return merged


def update_player_stats_csv(new_df):
    """
    Replace 2025-26 rows in player_stats_by_season.csv.
    Reattaches OVERALL_PICK and DRAFT_YEAR from historical rows.
    """
    try:
        existing = pd.read_csv(PLAYER_STATS_FILE)
    except FileNotFoundError:
        print(f'  ERROR: {PLAYER_STATS_FILE} not found — run nba_player_data_collector.py first.')
        return

    # Build draft lookup from all historical rows (non-2025-26)
    hist = existing[existing['SEASON'] != SEASON]
    draft_lkp = (
        hist[['PLAYER_ID', 'OVERALL_PICK', 'DRAFT_YEAR']]
        .dropna(subset=['PLAYER_ID'])
        .drop_duplicates('PLAYER_ID')
        .set_index('PLAYER_ID')
    )

    # Merge draft info onto new rows
    new_df = new_df.merge(
        draft_lkp.reset_index(),
        on='PLAYER_ID', how='left'
    )
    if 'OVERALL_PICK' not in new_df.columns:
        new_df['OVERALL_PICK'] = 999
    if 'DRAFT_YEAR' not in new_df.columns:
        new_df['DRAFT_YEAR'] = 0
    new_df['OVERALL_PICK'] = new_df['OVERALL_PICK'].fillna(999).astype(int)
    new_df['DRAFT_YEAR']   = new_df['DRAFT_YEAR'].fillna(0).astype(int)

    # Align columns to match existing file
    all_cols = list(existing.columns)
    for c in all_cols:
        if c not in new_df.columns:
            new_df[c] = None
    new_df = new_df[[c for c in all_cols if c in new_df.columns]]

    before = len(existing[existing['SEASON'] == SEASON])
    combined = pd.concat([hist, new_df], ignore_index=True)
    combined.to_csv(PLAYER_STATS_FILE, index=False)
    print(f'  ✓ {PLAYER_STATS_FILE} updated  '
          f'(replaced {before} old rows → {len(new_df)} fresh rows;  '
          f'total {len(combined)} rows)')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 65)
    print('NBA 2025-26 SEASON DATA REFRESH')
    print('=' * 65)
    print(f'Season: {SEASON}')
    print()

    # Force fresh HTTP session
    import requests as _requests
    from nba_api.stats.library.http import NBAStatsHTTP
    NBAStatsHTTP.set_session(_requests.Session())

    # ── Team stats ──
    print('── TEAM STATS ──')
    team_df = fetch_team_stats()
    if team_df is not None:
        update_team_clean_csv(team_df)
    print()

    # ── Player stats ──
    print('── PLAYER STATS ──')
    player_df = fetch_player_stats()
    if player_df is not None:
        update_player_stats_csv(player_df)
    print()

    print('=' * 65)
    print('Refresh complete. Next steps:')
    print('  python player_rating_engine.py')
    print('  python team_profile_builder.py')
    print('  python generate_multiyear_similarity.py')
    print('  python generate_multiyear_report.py')
    print('=' * 65)


if __name__ == '__main__':
    main()
