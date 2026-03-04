"""
2024-25 Full League Breakdown
Generates two CSV files covering all 30 NBA teams in the 2024-25 season:

  nba_2024_25_similarity.csv  — top 5 historical matches per team (150 rows)
  nba_2024_25_rosters.csv     — top 10 players by minutes per team (~300 rows)

Runs in seconds — no API calls required (reads from pre-built CSVs).

Author: GY Mehta
Date: February 2026
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Feature weights — must match franchise_similarity_engine.py exactly
# ---------------------------------------------------------------------------
FEATURE_WEIGHTS = {
    'n_superstars':        2.5,
    'top_player_score':    1.5,
    'depth_score':         1.2,
    'roster_potential':    1.2,
    'net_rating':          1.0,
    'second_player_score': 1.0,
    'win_pct':             0.8,
}
FEATURE_COLUMNS = list(FEATURE_WEIGHTS.keys())

FEATURE_LABELS = {
    'n_superstars':        'superstar presence',
    'top_player_score':    'top player quality',
    'depth_score':         'roster depth',
    'roster_potential':    'roster potential',
    'net_rating':          'net rating',
    'second_player_score': '2nd player quality',
    'win_pct':             'win percentage',
}


# ---------------------------------------------------------------------------
# Similarity helpers (copied from franchise_similarity_engine.py)
# ---------------------------------------------------------------------------

def normalize_features(df, feature_cols):
    means, stds = {}, {}
    df_norm = df.copy()
    for col in feature_cols:
        mean = df[col].mean()
        std  = df[col].std()
        means[col] = mean
        stds[col]  = std if std > 0 else 1.0
        df_norm[col] = (df[col] - means[col]) / stds[col]
    return df_norm, means, stds


def compute_similarity_scores(query_row, candidate_df, feature_cols, weights):
    query_vec = np.array([query_row[col] * weights[col] for col in feature_cols])
    distances = []
    for _, row in candidate_df.iterrows():
        cand_vec = np.array([row[col] * weights[col] for col in feature_cols])
        distances.append(np.sqrt(np.sum((query_vec - cand_vec) ** 2)))
    distances = np.array(distances)
    similarity = 100 / (1 + distances)
    if similarity.max() > 0:
        similarity = similarity / similarity.max() * 100
    return pd.Series(similarity, index=candidate_df.index)


def get_similarity_reason(query_row, match_row, feature_cols):
    diffs = {col: abs(query_row[col] - match_row[col]) for col in feature_cols}
    sorted_dims = sorted(diffs.items(), key=lambda x: x[1])
    top2 = [FEATURE_LABELS.get(col, col) for col, _ in sorted_dims[:2]]
    return f"Similar {top2[0]} + {top2[1]}"


def fmt_record(w, l):
    try:
        return f"{int(w)}-{int(l)}"
    except (ValueError, TypeError):
        return 'N/A'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('=' * 70)
    print('2024-25 NBA FULL LEAGUE BREAKDOWN')
    print('=' * 70)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    try:
        profiles = pd.read_csv('team_profiles.csv')
    except FileNotFoundError:
        print('team_profiles.csv not found — run team_profile_builder.py first')
        return

    try:
        ratings = pd.read_csv('player_ratings.csv')
    except FileNotFoundError:
        print('player_ratings.csv not found — run player_rating_engine.py first')
        return

    print(f'Loaded {len(profiles)} team profiles, {len(ratings)} player ratings')

    player_profiles = profiles[profiles['has_player_data'] == True].copy()

    # -----------------------------------------------------------------------
    # 2. Prepare features for similarity (fill NaN, normalize)
    # -----------------------------------------------------------------------
    working = player_profiles.copy()
    for col in FEATURE_COLUMNS:
        if col not in working.columns:
            working[col] = 0.0
        col_mean = working[col].mean()
        working[col] = working[col].fillna(col_mean)

    working_norm, _, _ = normalize_features(working, FEATURE_COLUMNS)

    # -----------------------------------------------------------------------
    # 3. Isolate 2024-25 teams and historical candidates
    # -----------------------------------------------------------------------
    mask_2425 = working['SEASON'] == '2024-25'
    teams_2425 = working[mask_2425].copy()
    teams_2425_norm = working_norm[mask_2425].copy()

    # Historical = everything NOT 2024-25
    hist_mask = ~mask_2425
    hist_raw  = working[hist_mask].copy()
    hist_norm = working_norm[hist_mask].copy()

    print(f'2024-25 teams: {len(teams_2425)}')
    print(f'Historical candidates: {len(hist_raw)}')

    # -----------------------------------------------------------------------
    # 4. Run similarity for each 2024-25 team
    # -----------------------------------------------------------------------
    similarity_rows = []

    for _, q_norm in teams_2425_norm.iterrows():
        team_name = teams_2425.loc[q_norm.name, 'Team']
        q_raw = player_profiles.loc[q_norm.name]

        scores = compute_similarity_scores(q_norm, hist_norm, FEATURE_COLUMNS, FEATURE_WEIGHTS)
        hist_raw_copy = hist_raw.copy()
        hist_raw_copy['_sim'] = scores

        top5 = hist_raw_copy.nlargest(5, '_sim').reset_index(drop=True)

        for rank_idx, match_row in top5.iterrows():
            match_norm_row = hist_norm.loc[match_row.name] \
                if match_row.name in hist_norm.index else hist_norm.iloc[0]
            reason = get_similarity_reason(q_norm, match_norm_row, FEATURE_COLUMNS)

            similarity_rows.append({
                # --- 2024-25 team info ---
                'team':                q_raw.get('Team', ''),
                'W':                   q_raw.get('W', ''),
                'L':                   q_raw.get('L', ''),
                'record':              fmt_record(q_raw.get('W'), q_raw.get('L')),
                'net_rating':          round(float(q_raw.get('net_rating', 0) or 0), 1),
                'win_pct':             round(float(q_raw.get('win_pct', 0) or 0), 3),
                'n_superstars':        int(q_raw.get('n_superstars', 0)),
                'n_stars':             int(q_raw.get('n_stars', 0)),
                'n_starters':          int(q_raw.get('n_starters', 0)),
                'n_role_players':      int(q_raw.get('n_role_players', 0)),
                'top_player':          q_raw.get('top_player_name', ''),
                'second_player':       q_raw.get('second_player_name', ''),
                'depth_score':         round(float(q_raw.get('depth_score', 0) or 0), 3),
                'roster_potential':    round(float(q_raw.get('roster_potential', 0) or 0), 3),
                # --- Match info ---
                'match_rank':          rank_idx + 1,
                'match_team':          match_row.get('Team', ''),
                'match_season':        match_row.get('SEASON', ''),
                'match_W':             match_row.get('W', ''),
                'match_L':             match_row.get('L', ''),
                'match_record':        fmt_record(match_row.get('W'), match_row.get('L')),
                'match_net_rating':    round(float(match_row.get('net_rating', 0) or 0), 1),
                'match_win_pct':       round(float(match_row.get('win_pct', 0) or 0), 3),
                'match_n_superstars':  int(match_row.get('n_superstars', 0)),
                'match_n_stars':       int(match_row.get('n_stars', 0)),
                'match_n_starters':    int(match_row.get('n_starters', 0)),
                'match_top_player':    match_row.get('top_player_name', ''),
                'match_second_player': match_row.get('second_player_name', ''),
                'match_depth_score':   round(float(match_row.get('depth_score', 0) or 0), 3),
                'similarity_score':    round(float(match_row['_sim']), 1),
                'similarity_reason':   reason,
            })

    sim_df = pd.DataFrame(similarity_rows)
    sim_df = sim_df.sort_values(['team', 'match_rank']).reset_index(drop=True)

    sim_df.to_csv('nba_2024_25_similarity.csv', index=False)
    print(f'\nSaved nba_2024_25_similarity.csv ({len(sim_df)} rows)')

    # -----------------------------------------------------------------------
    # 5. Build roster details for 2024-25
    # -----------------------------------------------------------------------
    ratings_2425 = ratings[ratings['SEASON'] == '2024-25'].copy()

    # Join per-game box score stats from player_stats_by_season.csv
    # (player_ratings.csv drops those columns after computing composite)
    try:
        stats_raw = pd.read_csv('player_stats_by_season.csv')
        stats_2425 = stats_raw[stats_raw['SEASON'] == '2024-25'][
            ['PLAYER_ID', 'TEAM_ID', 'GP', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        ].copy()
        ratings_2425 = ratings_2425.merge(
            stats_2425, on=['PLAYER_ID', 'TEAM_ID'], how='left'
        )
    except FileNotFoundError:
        for col in ['GP', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']:
            ratings_2425[col] = None

    # Build TEAM_ABBREVIATION → full team name via top-player matching
    abbr_to_name = {}
    for _, prof_row in player_profiles[player_profiles['SEASON'] == '2024-25'].iterrows():
        top_name = prof_row.get('top_player_name', '')
        if top_name:
            match = ratings_2425[ratings_2425['PLAYER_NAME'] == top_name]
            if len(match) > 0:
                abbr_to_name[match.iloc[0]['TEAM_ABBREVIATION']] = prof_row['Team']

    ratings_2425['team_name'] = (
        ratings_2425['TEAM_ABBREVIATION'].map(abbr_to_name)
        .fillna(ratings_2425['TEAM_ABBREVIATION'])
    )

    roster_rows = []
    for team_name_key, group in ratings_2425.groupby('team_name'):
        top10 = group.nlargest(10, 'TOTAL_MIN').reset_index(drop=True)
        for rank_idx, player in top10.iterrows():
            def _f(v, d=1): return round(float(v), d) if pd.notna(v) else None
            def _i(v):      return int(v) if pd.notna(v) else None
            roster_rows.append({
                'team':               team_name_key,
                'rank_on_team':       rank_idx + 1,
                'player_name':        player.get('PLAYER_NAME', ''),
                'tier':               player.get('tier', ''),
                'GP':                 _i(player.get('GP')),
                'total_min':          _i(player.get('TOTAL_MIN')),
                'PTS':                _f(player.get('PTS')),
                'REB':                _f(player.get('REB')),
                'AST':                _f(player.get('AST')),
                'STL':                _f(player.get('STL')),
                'BLK':                _f(player.get('BLK')),
                'TOV':                _f(player.get('TOV')),
                'composite_score':    _f(player.get('composite_score'), 4),
                'within_season_pct':  _f(player.get('within_season_pct')),
                'potential_score':    _f(player.get('potential_score'), 4),
            })

    roster_df = pd.DataFrame(roster_rows)
    roster_df = roster_df.sort_values(['team', 'rank_on_team']).reset_index(drop=True)

    roster_df.to_csv('nba_2024_25_rosters.csv', index=False)
    print(f'Saved nba_2024_25_rosters.csv ({len(roster_df)} rows)')

    # -----------------------------------------------------------------------
    # 6. Quick summary
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('SNAPSHOT: Top match for each 2024-25 team')
    print('=' * 70)
    top1 = sim_df[sim_df['match_rank'] == 1].copy()
    top1 = top1.sort_values('team')
    col_w = [28, 28, 8, 12, 10]
    headers = ['2024-25 Team', 'Best Historical Match', 'Season', 'Similarity', 'Net Rtg']
    print('  '.join(h.ljust(w) for h, w in zip(headers, col_w)))
    print('-' * 90)
    for _, row in top1.iterrows():
        vals = [
            str(row['team'])[:27],
            str(row['match_team'])[:27],
            str(row['match_season']),
            f"{row['similarity_score']:.1f}",
            f"{row['net_rating']:+.1f}",
        ]
        print('  '.join(v.ljust(w) for v, w in zip(vals, col_w)))

    print('\nDone. Files written:')
    print('  nba_2024_25_similarity.csv — 30 teams x 5 matches = 150 rows')
    print('  nba_2024_25_rosters.csv    — top 10 players per team')


if __name__ == '__main__':
    main()
