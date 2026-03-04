"""
Team Profile Builder
Aggregates player ratings into a feature vector (profile) for each team-season.

Each team profile captures:
  - Tier counts (Superstars, Stars, Starters, Role Players in top 10 by minutes)
  - Top player quality and second player quality
  - Roster depth score (weighted sum of top 8)
  - Team performance (net rating, win%)
  - Roster age and potential

Joins player data to nba_team_stats_clean.csv for team performance metrics.

Outputs: team_profiles.csv

Author: GY Mehta
Date: February 2026
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Depth score weights for top 8 players (sorted by smoothed_composite desc)
# Position 1 = best player, position 8 = 8th best
# ---------------------------------------------------------------------------
DEPTH_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

# ---------------------------------------------------------------------------
# TEAM_ID -> Team name mapping for join
# Built from the raw team stats file; this dict is a fallback for edge cases.
# Keys are TEAM_ID integers from the NBA API.
# ---------------------------------------------------------------------------
TEAM_ID_FALLBACK = {
    1610612737: 'Atlanta Hawks',
    1610612738: 'Boston Celtics',
    1610612751: 'Brooklyn Nets',
    1610612766: 'Charlotte Hornets',
    1610612741: 'Chicago Bulls',
    1610612739: 'Cleveland Cavaliers',
    1610612742: 'Dallas Mavericks',
    1610612743: 'Denver Nuggets',
    1610612765: 'Detroit Pistons',
    1610612744: 'Golden State Warriors',
    1610612745: 'Houston Rockets',
    1610612754: 'Indiana Pacers',
    1610612746: 'Los Angeles Clippers',
    1610612747: 'Los Angeles Lakers',
    1610612763: 'Memphis Grizzlies',
    1610612748: 'Miami Heat',
    1610612749: 'Milwaukee Bucks',
    1610612750: 'Minnesota Timberwolves',
    1610612740: 'New Orleans Pelicans',
    1610612752: 'New York Knicks',
    1610612760: 'Oklahoma City Thunder',
    1610612753: 'Orlando Magic',
    1610612755: 'Philadelphia 76ers',
    1610612756: 'Phoenix Suns',
    1610612757: 'Portland Trail Blazers',
    1610612758: 'Sacramento Kings',
    1610612759: 'San Antonio Spurs',
    1610612761: 'Toronto Raptors',
    1610612762: 'Utah Jazz',
    1610612764: 'Washington Wizards',
    # Historical teams
    1610612745: 'Houston Rockets',
    1610612729: 'Minnesota Timberwolves',   # pre-relocation ID variant
}


def build_team_name_lookup(ratings_df, raw_team_file='nba_team_stats_1980_2025.csv'):
    """
    Build a (TEAM_ID, SEASON) -> Team full name lookup.

    Primary: load the raw team stats CSV which has TEAM_ID + Team columns.
    Fallback: use TEAM_ID_FALLBACK dict.
    """
    lookup = {}

    # Try loading raw team stats file
    try:
        raw = pd.read_csv(raw_team_file)
        if 'TEAM_ID' in raw.columns and 'Team' in raw.columns and 'SEASON' in raw.columns:
            for _, row in raw[['TEAM_ID', 'SEASON', 'Team']].iterrows():
                lookup[(int(row['TEAM_ID']), row['SEASON'])] = row['Team']
            print(f"  ✓ Team name lookup built from {raw_team_file} ({len(lookup)} entries)")
            return lookup
    except Exception:
        pass

    # Fallback: use static dict (covers current + recently relocated teams)
    print("  ⚠ Raw team stats file unavailable — using fallback TEAM_ID mapping")
    # Build from ratings file itself: same TEAM_ID in same season = same name
    for team_id, team_id_df in ratings_df.groupby('TEAM_ID'):
        name = TEAM_ID_FALLBACK.get(int(team_id), f'Team_{team_id}')
        for season in team_id_df['SEASON'].unique():
            lookup[(int(team_id), season)] = name

    return lookup


def build_team_profile(group):
    """
    Build a single team's profile dict from its qualified player rows for one season.
    `group` is pre-filtered to top 10 players by TOTAL_MIN.

    Uses smoothed_composite (career-blended) for quality scores so that tier
    counts and depth ratings are not skewed by single-season injury absences or
    one-year breakout spikes. Falls back to composite_score if column is absent
    (older runs of the rating engine).
    """
    score_col = 'smoothed_composite' if 'smoothed_composite' in group.columns else 'composite_score'

    # Sort by smoothed_composite for depth weighting
    by_score = group.sort_values(score_col, ascending=False).reset_index(drop=True)
    # Sort by TOTAL_MIN for age weighting
    by_min = group.sort_values('TOTAL_MIN', ascending=False).reset_index(drop=True)

    # --- Tier counts (from top 10 by minutes) ---
    tier_counts = group['tier'].value_counts()
    n_superstars = tier_counts.get('Superstar', 0)
    n_stars = tier_counts.get('Star', 0)
    n_starters = tier_counts.get('Starter', 0)
    n_role_players = tier_counts.get('Role Player', 0)

    # --- Quality scores (from top 8 by smoothed composite) ---
    top8 = by_score.head(8)
    top_player_score = by_score.iloc[0][score_col] if len(by_score) >= 1 else 0.0
    second_player_score = by_score.iloc[1][score_col] if len(by_score) >= 2 else 0.0
    top_player_name = by_score.iloc[0]['PLAYER_NAME'] if len(by_score) >= 1 else ''
    second_player_name = by_score.iloc[1]['PLAYER_NAME'] if len(by_score) >= 2 else ''

    # Depth score: weighted sum of top 8
    scores = top8[score_col].values
    weights = DEPTH_WEIGHTS[:len(scores)]
    depth_score = float(np.dot(scores, weights))

    # --- Potential features (from top 8 by minutes) ---
    top8_min = by_min.head(8)

    # Minutes-weighted average age (NaN when age data is unavailable)
    ages = top8_min['AGE'].dropna()
    if len(ages) > 0:
        valid = top8_min.dropna(subset=['AGE'])
        total_min_valid = valid['TOTAL_MIN'].sum()
        if total_min_valid > 0:
            avg_age_core = float((valid['AGE'] * valid['TOTAL_MIN']).sum() / total_min_valid)
        else:
            avg_age_core = float(ages.mean())
    else:
        avg_age_core = float('nan')  # no age data for this season

    roster_potential = float(top8_min['potential_score'].sum())

    # Young stars: Star or Superstar under 25
    young_stars_mask = (top8_min['tier'].isin(['Star', 'Superstar'])) & (top8_min['AGE'] < 25)
    n_young_stars = int(young_stars_mask.sum())

    return {
        'n_superstars': n_superstars,
        'n_stars': n_stars,
        'n_starters': n_starters,
        'n_role_players': n_role_players,
        'top_player_score': round(top_player_score, 4),
        'second_player_score': round(second_player_score, 4),
        'depth_score': round(depth_score, 4),
        'avg_age_core': round(avg_age_core, 2),
        'roster_potential': round(roster_potential, 4),
        'n_young_stars': n_young_stars,
        'top_player_name': top_player_name,
        'second_player_name': second_player_name,
        'qualified_player_count': len(group),
    }


def main():
    print("=" * 70)
    print("TEAM PROFILE BUILDER")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Load player ratings
    # -----------------------------------------------------------------------
    try:
        ratings = pd.read_csv('player_ratings.csv')
    except FileNotFoundError:
        print("✗ player_ratings.csv not found — run player_rating_engine.py first")
        return

    print(f"✓ Loaded player_ratings.csv: {len(ratings)} rows")

    # -----------------------------------------------------------------------
    # 2. Build team name lookup
    # -----------------------------------------------------------------------
    print("Building team name lookup...")
    team_name_lookup = build_team_name_lookup(ratings)

    ratings['Team'] = ratings.apply(
        lambda r: team_name_lookup.get((int(r['TEAM_ID']), r['SEASON']),
                                       TEAM_ID_FALLBACK.get(int(r['TEAM_ID']),
                                                            f"Team_{int(r['TEAM_ID'])}")),
        axis=1
    )

    # -----------------------------------------------------------------------
    # 3. Load team performance data
    # -----------------------------------------------------------------------
    try:
        team_stats = pd.read_csv('nba_team_stats_clean.csv')
    except FileNotFoundError:
        print("✗ nba_team_stats_clean.csv not found")
        return

    print(f"✓ Loaded nba_team_stats_clean.csv: {len(team_stats)} rows")

    # -----------------------------------------------------------------------
    # 4. Build profile for each team-season
    # -----------------------------------------------------------------------
    print("Building team profiles...")
    profiles = []

    for (team_name, season), group in ratings.groupby(['Team', 'SEASON']):
        # Select top 10 players by total minutes for this team-season
        top10 = group.nlargest(10, 'TOTAL_MIN')

        if len(top10) < 3:
            # Not enough qualified players for a meaningful profile
            continue

        profile = build_team_profile(top10)
        profile['Team'] = team_name
        profile['SEASON'] = season
        profiles.append(profile)

    profiles_df = pd.DataFrame(profiles)
    print(f"✓ Built {len(profiles_df)} team-season profiles")

    # -----------------------------------------------------------------------
    # 5. Join with team performance stats
    # -----------------------------------------------------------------------
    # Rename team stats columns for clarity
    perf_cols = ['Team', 'SEASON', 'YEAR', 'ERA', 'GP', 'W', 'L', 'WIN_PCT',
                 'NRtg', 'ORtg', 'DRtg']
    perf_cols = [c for c in perf_cols if c in team_stats.columns]
    perf = team_stats[perf_cols].copy()
    perf = perf.rename(columns={'NRtg': 'net_rating', 'WIN_PCT': 'win_pct'})

    merged = profiles_df.merge(perf, on=['Team', 'SEASON'], how='left')

    # Flag rows where team stats join failed
    merged['has_perf_data'] = merged['win_pct'].notna()
    n_missing = (~merged['has_perf_data']).sum()
    if n_missing > 0:
        print(f"  ⚠ {n_missing} team-seasons could not be matched to team stats")

    # Add SEASON_YEAR for display
    merged['SEASON_YEAR'] = merged['SEASON'].str[:4].astype(int) + 1

    # Mark all rows as having player data (since we just built them from player data)
    merged['has_player_data'] = True

    # -----------------------------------------------------------------------
    # 6. Append pre-1995 team-season rows (team stats only, no player data)
    # -----------------------------------------------------------------------
    pre_1995 = team_stats[team_stats['SEASON'].str[:4].astype(int) < 1995].copy()
    if len(pre_1995) > 0:
        pre_1995 = pre_1995.rename(columns={'NRtg': 'net_rating'})
        pre_1995['has_player_data'] = False
        pre_1995['SEASON_YEAR'] = pre_1995['SEASON'].str[:4].astype(int) + 1
        # Fill player-derived columns with NaN
        for col in profiles_df.columns:
            if col not in pre_1995.columns:
                pre_1995[col] = np.nan
        merged = pd.concat([merged, pre_1995], ignore_index=True)
        print(f"✓ Appended {len(pre_1995)} pre-1995 rows (team stats only, no player data)")

    # -----------------------------------------------------------------------
    # 7. Save
    # -----------------------------------------------------------------------
    output_file = 'team_profiles.csv'
    merged = merged.sort_values(['SEASON', 'Team']).reset_index(drop=True)
    merged.to_csv(output_file, index=False)
    print(f"✓ Saved {output_file} ({len(merged)} rows, {len(merged.columns)} columns)")

    # -----------------------------------------------------------------------
    # 8. Validation
    # -----------------------------------------------------------------------
    player_data_rows = merged[merged['has_player_data'] == True]

    print("\n" + "=" * 70)
    print("PROFILE SUMMARY")
    print("=" * 70)
    print(f"Total team-seasons: {len(merged)}")
    print(f"  With player data: {len(player_data_rows)}")
    print(f"  Team stats only:  {len(merged) - len(player_data_rows)}")

    print("\nTop 10 teams by depth_score (best balanced rosters):")
    top_depth = (player_data_rows
                 .nlargest(10, 'depth_score')
                 [['Team', 'SEASON', 'depth_score', 'n_superstars', 'n_stars',
                   'top_player_name', 'W', 'net_rating']])
    print(top_depth.to_string(index=False))

    print("\nHighest potential rosters (young and talented):")
    top_pot = (player_data_rows
               .nlargest(10, 'roster_potential')
               [['Team', 'SEASON', 'roster_potential', 'avg_age_core',
                 'n_young_stars', 'top_player_name', 'W']])
    print(top_pot.to_string(index=False))

    print("\n✓ Next step: run franchise_similarity_engine.py")


if __name__ == "__main__":
    main()
