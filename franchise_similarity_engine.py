"""
Franchise Similarity Engine
Finds the most historically similar NBA team-seasons to a given query.

Usage:
    python franchise_similarity_engine.py "Toronto Raptors 2024-25"
    python franchise_similarity_engine.py "Golden State Warriors 2016-17"
    python franchise_similarity_engine.py "San Antonio Spurs 2023-24"

Similarity is computed using weighted Euclidean distance on a normalized
profile vector that captures roster quality, team performance, and potential.

Author: GY Mehta
Date: February 2026
"""

import sys
import re
import difflib
import argparse
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Feature weights for similarity computation
# Higher weight = that dimension matters more for defining "similar" teams
# ---------------------------------------------------------------------------
FEATURE_WEIGHTS = {
    'n_superstars':        2.5,   # Whether a team has a franchise cornerstone
    'top_player_score':    1.5,   # Quality of the best player
    'depth_score':         1.2,   # Overall roster strength
    'roster_potential':    1.2,   # Future upside of the roster
    'net_rating':          1.0,   # How good the team actually was
    'second_player_score': 1.0,   # Quality of the sidekick
    'win_pct':             0.8,   # Win percentage (correlated with net_rating)
    # avg_age_core / n_young_stars excluded: age data unavailable for most seasons
}

FEATURE_COLUMNS = list(FEATURE_WEIGHTS.keys())

# Human-readable labels for similarity reason output
FEATURE_LABELS = {
    'n_superstars':        'superstar presence',
    'top_player_score':    'top player quality',
    'depth_score':         'roster depth',
    'roster_potential':    'roster potential',
    'net_rating':          'net rating',
    'avg_age_core':        'team age',
    'n_young_stars':       'young star count',
    'second_player_score': '2nd player quality',
    'win_pct':             'win percentage',
}


def parse_query(args):
    """
    Parse a query like ['Toronto', 'Raptors', '2024-25'] into (team_name, season).
    Season must be in YYYY-YY format.
    """
    if not args:
        return None, None

    # Last element should be the season
    season_pattern = re.compile(r'^\d{4}-\d{2}$')

    if season_pattern.match(args[-1]):
        season = args[-1]
        team_name = ' '.join(args[:-1])
    else:
        # Try to find the season anywhere in the args
        for i, arg in enumerate(args):
            if season_pattern.match(arg):
                season = arg
                team_name = ' '.join(args[:i] + args[i+1:])
                return team_name.strip(), season
        return None, None

    return team_name.strip(), season


def normalize_features(df, feature_cols):
    """
    Z-score normalize each feature column across the full dataset.
    Columns with zero variance are set to 0.
    Returns normalized DataFrame and the scaler params (mean, std) for the query.
    """
    means = {}
    stds = {}
    df_norm = df.copy()

    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        means[col] = mean
        stds[col] = std if std > 0 else 1.0
        df_norm[col] = (df[col] - means[col]) / stds[col]

    return df_norm, means, stds


def compute_similarity_scores(query_row, candidate_df, feature_cols, weights):
    """
    Compute weighted Euclidean distance from query_row to each row in candidate_df.
    Returns Series of similarity scores (0-100, higher = more similar).
    """
    distances = []

    query_vec = np.array([query_row[col] * weights[col] for col in feature_cols])

    for _, row in candidate_df.iterrows():
        cand_vec = np.array([row[col] * weights[col] for col in feature_cols])
        dist = np.sqrt(np.sum((query_vec - cand_vec) ** 2))
        distances.append(dist)

    distances = np.array(distances)
    # Convert distance to 0-100 similarity score
    similarity = 100 / (1 + distances)
    # Rescale so the closest match gets close to 100
    if similarity.max() > 0:
        similarity = similarity / similarity.max() * 100

    return pd.Series(similarity, index=candidate_df.index)


def get_similarity_reason(query_row, match_row, feature_cols):
    """
    Identify the 2 feature dimensions where query and match are most similar.
    Returns a short descriptive string.
    """
    diffs = {col: abs(query_row[col] - match_row[col]) for col in feature_cols}
    # Sort by smallest difference (most similar dimensions)
    sorted_dims = sorted(diffs.items(), key=lambda x: x[1])
    top2 = [FEATURE_LABELS.get(col, col) for col, _ in sorted_dims[:2]]
    return f"Similar {top2[0]} + {top2[1]}"


def format_record(w, l):
    """Format W-L record, handling NaN."""
    try:
        return f"{int(w)}-{int(l)}"
    except (ValueError, TypeError):
        return "N/A"


def format_float(val, decimals=1):
    """Format a float, handling NaN."""
    try:
        return f"{float(val):+.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"


def print_results_table(query_team, query_season, query_info, results_df):
    """Print a formatted similarity table to stdout."""
    print("\n" + "=" * 100)
    # Query summary
    record = format_record(query_info.get('W'), query_info.get('L'))
    net_rtg = format_float(query_info.get('net_rating'))
    top_p = query_info.get('top_player_name', 'N/A')
    second_p = query_info.get('second_player_name', 'N/A')
    tier_summary = (f"{int(query_info.get('n_superstars', 0))}SS "
                    f"{int(query_info.get('n_stars', 0))}Star "
                    f"{int(query_info.get('n_starters', 0))}Start "
                    f"| Avg age {query_info.get('avg_age_core', 0):.1f}")

    print(f"QUERY: {query_team} {query_season}")
    print(f"Record: {record}  |  Net Rtg: {net_rtg}  |  "
          f"Top: {top_p}  |  2nd: {second_p}")
    print(f"Roster: {tier_summary}")
    print("=" * 100)

    # Header
    col_widths = [4, 28, 8, 12, 8, 8, 20, 20, 30]
    headers = ['Rank', 'Team', 'Season', 'Similarity', 'Record', 'Net Rtg',
               'Top Player', '2nd Player', 'Key Similarity']
    header_line = '  '.join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print('-' * 100)

    for _, row in results_df.iterrows():
        record_str = format_record(row.get('W'), row.get('L'))
        net_str = format_float(row.get('net_rating'))
        sim_str = f"{row['similarity_score']:.1f}"
        reason = row.get('similarity_reason', '')

        vals = [
            str(int(row['rank'])),
            str(row['Team'])[:27],
            str(row['SEASON']),
            sim_str,
            record_str,
            net_str,
            str(row.get('top_player_name', ''))[:19],
            str(row.get('second_player_name', ''))[:19],
            reason[:29],
        ]
        line = '  '.join(v.ljust(w) for v, w in zip(vals, col_widths))
        print(line)

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description='Find NBA teams historically similar to a given team-season.',
        epilog='Example: python franchise_similarity_engine.py "Toronto Raptors 2024-25"'
    )
    parser.add_argument('query', nargs='+',
                        help='Team name and season (e.g., "Toronto Raptors 2024-25")')
    parser.add_argument('--top', type=int, default=10,
                        help='Number of similar teams to show (default: 10)')
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Parse query
    # -----------------------------------------------------------------------
    team_name, season = parse_query(args.query)

    if not team_name or not season:
        print("✗ Could not parse query. Example: python franchise_similarity_engine.py "
              "\"Toronto Raptors 2024-25\"")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 2. Load team profiles
    # -----------------------------------------------------------------------
    try:
        profiles = pd.read_csv('team_profiles.csv')
    except FileNotFoundError:
        print("✗ team_profiles.csv not found — run team_profile_builder.py first")
        sys.exit(1)

    print(f"✓ Loaded team_profiles.csv: {len(profiles)} team-seasons")

    # Only use rows with player data for similarity computation
    player_profiles = profiles[profiles['has_player_data'] == True].copy()
    print(f"✓ Profiles with player data: {len(player_profiles)}")

    # -----------------------------------------------------------------------
    # 3. Validate query
    # -----------------------------------------------------------------------
    available_teams = player_profiles['Team'].unique().tolist()
    available_seasons = player_profiles['SEASON'].unique().tolist()

    # Check team name
    if team_name not in available_teams:
        close = difflib.get_close_matches(team_name, available_teams, n=5, cutoff=0.4)
        print(f"✗ Team '{team_name}' not found in profiles.")
        if close:
            print(f"  Did you mean one of: {close}")
        else:
            print(f"  Available teams (sample): {sorted(available_teams)[:10]}")
        sys.exit(1)

    # Check season
    if season not in available_seasons:
        close_seasons = [s for s in available_seasons if s.startswith(season[:4])]
        print(f"✗ Season '{season}' not found for {team_name}.")
        if close_seasons:
            print(f"  Available seasons near that year: {close_seasons}")
        sys.exit(1)

    # Find query row
    query_mask = (player_profiles['Team'] == team_name) & (player_profiles['SEASON'] == season)
    if not query_mask.any():
        print(f"✗ No profile found for {team_name} {season}")
        sys.exit(1)

    query_row_raw = player_profiles[query_mask].iloc[0]
    print(f"✓ Query: {team_name} {season}")

    # -----------------------------------------------------------------------
    # 4. Handle missing feature values
    # -----------------------------------------------------------------------
    working = player_profiles.copy()

    for col in FEATURE_COLUMNS:
        if col not in working.columns:
            working[col] = 0.0
        col_mean = working[col].mean()
        working[col] = working[col].fillna(col_mean)

    # -----------------------------------------------------------------------
    # 5. Z-score normalize
    # -----------------------------------------------------------------------
    working_norm, means, stds = normalize_features(working, FEATURE_COLUMNS)

    # -----------------------------------------------------------------------
    # 6. Compute similarity scores
    # -----------------------------------------------------------------------
    query_row_norm = working_norm[query_mask].iloc[0]

    # Exclude the query itself from candidates
    candidates_norm = working_norm[~query_mask].copy()
    candidates_raw = working[~query_mask].copy()

    similarity_scores = compute_similarity_scores(
        query_row_norm, candidates_norm, FEATURE_COLUMNS, FEATURE_WEIGHTS
    )

    # -----------------------------------------------------------------------
    # 7. Rank and select top N
    # -----------------------------------------------------------------------
    candidates_raw = candidates_raw.copy()
    candidates_raw['similarity_score'] = similarity_scores

    top_matches = (candidates_raw
                   .nlargest(args.top, 'similarity_score')
                   .reset_index(drop=True))

    top_matches['rank'] = top_matches.index + 1

    # Add similarity reason
    top_matches['similarity_reason'] = top_matches.apply(
        lambda row: get_similarity_reason(query_row_norm,
                                          working_norm.loc[row.name]
                                          if row.name in working_norm.index
                                          else working_norm.iloc[0],
                                          FEATURE_COLUMNS),
        axis=1
    )

    # -----------------------------------------------------------------------
    # 8. Print results
    # -----------------------------------------------------------------------
    print_results_table(team_name, season, query_row_raw.to_dict(), top_matches)

    # -----------------------------------------------------------------------
    # 9. Detailed breakdown of query profile
    # -----------------------------------------------------------------------
    print("\nQUERY PROFILE BREAKDOWN")
    print("-" * 50)
    profile_labels = {
        'n_superstars':        'Superstars',
        'n_stars':             'Stars',
        'n_starters':          'Starters',
        'n_role_players':      'Role Players',
        'top_player_score':    'Top Player Score',
        'second_player_score': '2nd Player Score',
        'depth_score':         'Depth Score',
        'net_rating':          'Net Rating',
        'win_pct':             'Win %',
        'avg_age_core':        'Avg Age (core 8)',
        'roster_potential':    'Roster Potential',
        'n_young_stars':       'Young Stars (<25)',
    }
    for col, label in profile_labels.items():
        val = query_row_raw.get(col, 'N/A')
        try:
            if col in ('n_superstars', 'n_stars', 'n_starters', 'n_role_players', 'n_young_stars'):
                print(f"  {label:<22} {int(val)}")
            elif col == 'win_pct':
                print(f"  {label:<22} {float(val):.1%}")
            elif col == 'avg_age_core':
                print(f"  {label:<22} {float(val):.1f}")
            else:
                print(f"  {label:<22} {float(val):.3f}")
        except (ValueError, TypeError):
            print(f"  {label:<22} N/A")

    print()


if __name__ == "__main__":
    main()
