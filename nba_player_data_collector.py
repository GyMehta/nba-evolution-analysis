"""
NBA Player Stats Data Collector
Collects player-level stats for 1995-2025 using nba_api

Outputs:
    player_stats_by_season.csv  — one row per player-team stint per season

Author: GY Mehta
Date: February 2026
"""

import pandas as pd
import time
import warnings
from nba_api.stats.endpoints import leaguedashplayerstats, drafthistory
from nba_api.stats.library import http as nba_http  # for header patching
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Some stats.nba.com endpoints require these custom headers (sent by the
# NBA website's own JavaScript). Add them to the existing nba_api defaults
# rather than replacing the full header dict.
# ---------------------------------------------------------------------------
nba_http.NBAStatsHTTP.headers['x-nba-stats-origin'] = 'stats'
nba_http.NBAStatsHTTP.headers['x-nba-stats-token'] = 'true'

START_YEAR = 1995
END_YEAR = 2024  # 2024 = start of 2024-25 season


def get_season_string(year):
    """Convert year to NBA season format (e.g., 2023 -> '2023-24')"""
    next_year = str(year + 1)[-2:]
    return f"{year}-{next_year}"


def fetch_with_retry(fetch_fn, label="", max_retries=3, retry_sleep=5):
    """Execute an API fetch function with retry logic"""
    for attempt in range(max_retries):
        try:
            return fetch_fn()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}/{max_retries - 1} ({label}): {str(e)[:80]}")
                time.sleep(retry_sleep)
            else:
                print(f"    Failed after {max_retries} attempts ({label}): {str(e)[:80]}")
                return None


def collect_player_stats_for_season(season):
    """
    Fetch Base + Advanced player stats for a single season.
    Returns one row per player-team stint (traded players get multiple rows).
    """
    season_year = int(season[:4]) + 1  # '2023-24' -> 2024

    # --- Base stats ---
    def fetch_base():
        return leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            measure_type_detailed_defense='Base',
            per_mode_detailed='PerGame',
            season_type_all_star='Regular Season',
            timeout=120
        ).get_data_frames()[0]

    base_df = fetch_with_retry(fetch_base, label=f"{season} Base")
    if base_df is None or base_df.empty:
        return None

    time.sleep(0.6)

    # --- Advanced stats ---
    def fetch_advanced():
        return leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            measure_type_detailed_defense='Advanced',
            per_mode_detailed='PerGame',
            season_type_all_star='Regular Season',
            timeout=120
        ).get_data_frames()[0]

    advanced_df = fetch_with_retry(fetch_advanced, label=f"{season} Advanced")
    time.sleep(0.6)

    # --- Drop TOT aggregate rows (traded players appear once per team + once as TOT) ---
    base_df = base_df[base_df['TEAM_ABBREVIATION'] != 'TOT'].copy()

    # Base columns to keep — handle both PLAYER_AGE (common) and AGE (some API versions)
    age_col = 'PLAYER_AGE' if 'PLAYER_AGE' in base_df.columns else 'AGE'
    base_keep = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION',
                 age_col, 'GP', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
                 'FG_PCT', 'FG3_PCT', 'FT_PCT']
    # Keep TEAM_NAME if the API returns it (not all versions do)
    if 'TEAM_NAME' in base_df.columns:
        base_keep.append('TEAM_NAME')

    base_df = base_df[[col for col in base_keep if col in base_df.columns]].copy()
    # Normalize age column name to 'AGE'
    base_df = base_df.rename(columns={'PLAYER_AGE': 'AGE'})
    if 'AGE' not in base_df.columns:
        base_df['AGE'] = None

    # --- Merge advanced stats ---
    if advanced_df is not None and not advanced_df.empty:
        advanced_df = advanced_df[advanced_df['TEAM_ABBREVIATION'] != 'TOT'].copy()
        adv_keep = ['PLAYER_ID', 'TEAM_ID', 'PIE', 'NET_RATING', 'OFF_RATING',
                    'DEF_RATING', 'TS_PCT', 'USG_PCT', 'AST_PCT', 'REB_PCT',
                    'OREB_PCT', 'DREB_PCT']
        advanced_df = advanced_df[[col for col in adv_keep if col in advanced_df.columns]].copy()
        merged = base_df.merge(advanced_df, on=['PLAYER_ID', 'TEAM_ID'], how='left')
    else:
        merged = base_df.copy()
        for col in ['PIE', 'NET_RATING', 'OFF_RATING', 'DEF_RATING',
                    'TS_PCT', 'USG_PCT', 'AST_PCT', 'REB_PCT', 'OREB_PCT', 'DREB_PCT']:
            merged[col] = None

    # --- Add derived columns ---
    merged['SEASON'] = season
    merged['SEASON_YEAR'] = season_year
    merged['TOTAL_MIN'] = merged['MIN'] * merged['GP']
    merged['birth_year'] = (season_year - merged['AGE'].fillna(0)).astype(int)

    return merged


def collect_draft_history():
    """
    Fetch all-time NBA draft history.
    Returns a DataFrame with PERSON_ID, DRAFT_YEAR, OVERALL_PICK.
    """
    print("Fetching draft history (one-time call)...")

    def fetch():
        return drafthistory.DraftHistory(league_id='00', timeout=120).get_data_frames()[0]

    df = fetch_with_retry(fetch, label="DraftHistory")
    time.sleep(0.6)

    if df is None or df.empty:
        print("  Warning: Could not fetch draft history — all picks will be marked undrafted")
        return pd.DataFrame()

    # Keep relevant columns
    keep = ['PERSON_ID', 'SEASON', 'ROUND_NUMBER', 'ROUND_PICK']
    df = df[[col for col in keep if col in df.columns]].copy()

    # Compute overall pick position (approximate — uses 30 picks/round for all eras)
    df['ROUND_NUMBER'] = pd.to_numeric(df['ROUND_NUMBER'], errors='coerce').fillna(1)
    df['ROUND_PICK'] = pd.to_numeric(df['ROUND_PICK'], errors='coerce').fillna(60)
    df['OVERALL_PICK'] = ((df['ROUND_NUMBER'] - 1) * 30 + df['ROUND_PICK']).astype(int)
    df = df.rename(columns={'SEASON': 'DRAFT_YEAR'})

    print(f"  ✓ Draft history: {len(df)} picks across {df['DRAFT_YEAR'].nunique()} draft classes")
    return df


def main():
    print("=" * 70)
    print("NBA PLAYER STATS DATA COLLECTION")
    print("=" * 70)
    print(f"Seasons: {START_YEAR}-{START_YEAR + 1} to {END_YEAR}-{END_YEAR + 1}")
    print("Estimated time: ~3-5 minutes")
    print("=" * 70)
    print()

    # Force a fresh HTTP session — avoids inheriting a broken connection
    # from any previous failed run (nba_api reuses sessions across calls)
    import requests as _requests
    from nba_api.stats.library.http import NBAStatsHTTP
    NBAStatsHTTP.set_session(_requests.Session())

    # 1. Collect draft history (single call, covers all years)
    draft_df = collect_draft_history()

    output_file = 'player_stats_by_season.csv'

    # 2. Resume support: load any seasons already collected
    already_collected = set()
    if pd.io.common.file_exists(output_file):
        existing = pd.read_csv(output_file, usecols=['SEASON'])
        already_collected = set(existing['SEASON'].unique())
        print(f"Resuming — {len(already_collected)} seasons already in {output_file}, skipping them.")

    failed_seasons = []

    for year in range(START_YEAR, END_YEAR + 1):
        season = get_season_string(year)

        if season in already_collected:
            print(f"Skipping {season} (already collected)")
            continue

        print(f"Fetching {season}...")
        df = collect_player_stats_for_season(season)

        if df is not None and len(df) > 0:
            # Merge draft data immediately so each append is self-contained
            if not draft_df.empty:
                df = df.merge(
                    draft_df[['PERSON_ID', 'DRAFT_YEAR', 'OVERALL_PICK']],
                    left_on='PLAYER_ID', right_on='PERSON_ID', how='left'
                )
                df.drop(columns=['PERSON_ID'], errors='ignore', inplace=True)
            if 'OVERALL_PICK' not in df.columns:
                df['OVERALL_PICK'] = 999
            df['OVERALL_PICK'] = df['OVERALL_PICK'].fillna(999).astype(int)
            if 'DRAFT_YEAR' not in df.columns:
                df['DRAFT_YEAR'] = 0
            df['DRAFT_YEAR'] = df['DRAFT_YEAR'].fillna(0).astype(int)

            # Append to CSV incrementally — safe even if script is interrupted
            write_header = not pd.io.common.file_exists(output_file)
            df.to_csv(output_file, mode='a', header=write_header, index=False)
            print(f"  ✓ {len(df)} player-team stints (saved)")
        else:
            print(f"  ✗ Failed to collect {season}")
            failed_seasons.append(season)

        # Slightly longer sleep to reduce rate-limit hits
        time.sleep(2)

    if failed_seasons:
        print(f"\n⚠ {len(failed_seasons)} seasons failed: {failed_seasons}")
        print("  Re-run the script to retry — already-collected seasons will be skipped.")

    if not pd.io.common.file_exists(output_file):
        print("\n✗ No data collected. Check NBA API connectivity.")
        return

    player_df = pd.read_csv(output_file)
    print(f"\n✓ Total player-season stints in file: {len(player_df)}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total rows:       {len(player_df)}")
    print(f"Seasons covered:  {player_df['SEASON'].nunique()}")
    print(f"Unique players:   {player_df['PLAYER_ID'].nunique()}")
    print(f"Columns:          {list(player_df.columns)}")

    # Quick sanity check: superstars should be recognizable
    if 'PIE' in player_df.columns:
        top_seasons = player_df.nlargest(5, 'PIE')[['PLAYER_NAME', 'SEASON', 'PIE', 'TEAM_ABBREVIATION']]
        print("\nTop PIE seasons (sanity check):")
        print(top_seasons.to_string(index=False))

    print("\n✓ Next step: run player_rating_engine.py")


if __name__ == "__main__":
    main()
