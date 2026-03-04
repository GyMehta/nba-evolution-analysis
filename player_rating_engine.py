"""
Player Rating Engine
Reads player_stats_by_season.csv and produces player_ratings.csv with:
  - composite_score: weighted blend of PIE, NET_RATING, box score
  - tier: Superstar / Star / Starter / Role Player / Fringe (within-season percentile)
  - potential_score: composite adjusted for age curve and draft pedigree

Within-season percentile tiering ensures cross-era comparability —
a Superstar in 1996 is as rare as a Superstar in 2024.

Author: GY Mehta
Date: February 2026
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Tier thresholds (within-season percentile of composite_score)
# ---------------------------------------------------------------------------
TIER_THRESHOLDS = {
    'Superstar':   98.8,   # top ~1.2%  → ~4-5 players per season
    #              Raised from 98.5: the 98.5-98.8 band captured stat accumulators
    #              (e.g. Sabonis) whose counting numbers don't reflect team impact.
    #              True Superstars (top 4-5 per season) cluster at 99.0+.
    'Star':        90.0,   # top ~10%   → ~25-30 players per season
    'Starter':     60.0,   # top ~40%
    'Role Player': 30.0,   # top ~70%
    # 'Fringe': below 30th percentile
}


def normalize_within_season(series, clip_z=3.0):
    """
    Z-score normalize a series, clip extremes, then rescale to [0, 1].
    All operations are within the passed series (single season group).
    Returns 0.5 for all if series has zero variance.
    """
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.5, index=series.index)
    z = ((series - mean) / std).clip(-clip_z, clip_z)
    z_min, z_max = z.min(), z.max()
    if z_max == z_min:
        return pd.Series(0.5, index=series.index)
    return (z - z_min) / (z_max - z_min)


def compute_composite_for_group(group):
    """
    Compute composite score for a single season's qualified players.
    All normalization is within-season to avoid era bias.

    When PIE is available (rare — a few seasons with full advanced data):
        composite = 0.50 × PIE_norm + 0.30 × NET_RATING_norm + 0.20 × box_per36_norm

    Fallback (most seasons — box score + minutes + team net rating):
        composite = 0.50 × box_per36_norm + 0.30 × minutes_norm + 0.20 × team_nrtg_norm

    team_nrtg_norm: within-season normalized team net rating (proxy for player impact
    on winning). Rewards players on high-performing teams; penalizes stat accumulators
    on mediocre teams whose counting numbers don't translate to wins. Falls back to
    box+minutes only if team NRtg is unavailable.

    Box score weights dampened to reduce big-man inflation:
    REB 1.2 (was 1.5), AST/STL/BLK 1.5 (were 2.0).
    """
    g = group.copy()

    box_norm = normalize_within_season(g['box_per36'].fillna(g['box_per36'].median()))
    min_norm = normalize_within_season(g['TOTAL_MIN'])   # within-season relative playing time
    net_norm = normalize_within_season(g['NET_RATING'].fillna(g['NET_RATING'].median()))

    pie = g['PIE']
    has_pie = pie.notna() & (pie != 0)

    composite = pd.Series(np.nan, index=g.index)

    if has_pie.sum() > 0:
        pie_norm = normalize_within_season(pie.fillna(pie.median()))
        composite[has_pie] = (0.5 * pie_norm[has_pie]
                              + 0.3 * net_norm[has_pie]
                              + 0.2 * box_norm[has_pie])

    # Fallback: box efficiency + minutes + team net rating (winning context)
    no_pie = ~has_pie
    if 'team_nrtg' in g.columns and g['team_nrtg'].notna().sum() > 0:
        tnrtg_norm = normalize_within_season(
            g['team_nrtg'].fillna(g['team_nrtg'].median())
        )
        composite[no_pie] = (0.50 * box_norm[no_pie]
                             + 0.30 * min_norm[no_pie]
                             + 0.20 * tnrtg_norm[no_pie])
    else:
        composite[no_pie] = 0.60 * box_norm[no_pie] + 0.40 * min_norm[no_pie]

    return composite


def assign_tier(pct):
    """Assign tier label from within-season composite percentile (0-100)."""
    if pct > TIER_THRESHOLDS['Superstar']:
        return 'Superstar'
    elif pct >= TIER_THRESHOLDS['Star']:
        return 'Star'
    elif pct >= TIER_THRESHOLDS['Starter']:
        return 'Starter'
    elif pct >= TIER_THRESHOLDS['Role Player']:
        return 'Role Player'
    else:
        return 'Fringe'


def get_age_factor(age):
    """
    Age-based upside multiplier. Reflects career trajectory:
    young players have higher ceiling, veterans are in decline.
    """
    if pd.isna(age):
        return 1.0  # neutral default
    age = float(age)
    if age < 22:    return 1.40
    elif age < 24:  return 1.25
    elif age < 26:  return 1.10
    elif age < 29:  return 1.00  # peak years
    elif age < 31:  return 0.90
    elif age < 33:  return 0.75
    else:           return 0.60


def get_draft_factor(overall_pick):
    """
    Draft pedigree multiplier. High draft picks signal higher ceiling
    due to team evaluation of physical tools and skill ceiling.
    """
    if pd.isna(overall_pick) or int(overall_pick) >= 999:
        return 0.90   # undrafted
    pick = int(overall_pick)
    if pick <= 5:    return 1.15  # franchise-changing pick
    elif pick <= 14: return 1.10  # lottery
    elif pick <= 30: return 1.00  # first round
    elif pick <= 60: return 0.95  # second round
    else:            return 0.90  # edge case (historical large drafts)


def main():
    print("=" * 70)
    print("PLAYER RATING ENGINE")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    try:
        df = pd.read_csv('player_stats_by_season.csv')
    except FileNotFoundError:
        print("✗ player_stats_by_season.csv not found — run nba_player_data_collector.py first")
        return

    print(f"✓ Loaded {len(df)} rows from player_stats_by_season.csv")

    # Ensure TOTAL_MIN exists
    if 'TOTAL_MIN' not in df.columns:
        df['TOTAL_MIN'] = df['MIN'] * df['GP']

    # -----------------------------------------------------------------------
    # 2. Minutes filter — scale for shortened seasons
    # -----------------------------------------------------------------------
    season_max_gp = df.groupby('SEASON')['GP'].max()
    df['_season_max_gp'] = df['SEASON'].map(season_max_gp)
    df['_min_threshold'] = (500 * (df['_season_max_gp'] / 82)).clip(lower=250)

    df_q = df[df['TOTAL_MIN'] >= df['_min_threshold']].copy()
    print(f"✓ Qualified after minutes filter: {len(df_q)} rows "
          f"(removed {len(df) - len(df_q)} low-minute players)")

    # -----------------------------------------------------------------------
    # 2b. Join team net rating as winning-context signal
    # Team NRtg rewards players on high-performing teams and penalizes
    # stat accumulators on mediocre teams (e.g. Sabonis on Sacramento).
    # Primary source: nba_team_stats_1980_2025.csv (has TEAM_ID + NRtg).
    # Fallback for 2025-26: nba_team_stats_clean.csv via team name bridge.
    # -----------------------------------------------------------------------
    team_nrtg_by_id = {}    # (TEAM_ID, SEASON) → NRtg
    team_name_by_id = {}    # TEAM_ID → canonical name (for name bridge)
    nrtg_by_name = {}       # (Team, SEASON) → NRtg  (clean CSV, has 2025-26)

    try:
        raw_ts = pd.read_csv('nba_team_stats_1980_2025.csv',
                             usecols=['TEAM_ID', 'Team', 'SEASON', 'NRtg'])
        for _, row in raw_ts.iterrows():
            team_nrtg_by_id[(int(row['TEAM_ID']), row['SEASON'])] = row['NRtg']
            team_name_by_id[int(row['TEAM_ID'])] = row['Team']

        clean_ts = pd.read_csv('nba_team_stats_clean.csv',
                               usecols=['Team', 'SEASON', 'NRtg'])
        nrtg_by_name = clean_ts.set_index(['Team', 'SEASON'])['NRtg'].to_dict()

        def get_team_nrtg(row):
            tid = int(row['TEAM_ID'])
            season = row['SEASON']
            nrtg = team_nrtg_by_id.get((tid, season))
            if nrtg is not None and not pd.isna(nrtg):
                return float(nrtg)
            tname = team_name_by_id.get(tid)
            if tname:
                nrtg = nrtg_by_name.get((tname, season))
                if nrtg is not None and not pd.isna(nrtg):
                    return float(nrtg)
            return float('nan')

        df_q['team_nrtg'] = df_q.apply(get_team_nrtg, axis=1)
        n_mapped = df_q['team_nrtg'].notna().sum()
        print(f"✓ Team NRtg mapped: {n_mapped}/{len(df_q)} player-seasons "
              f"({n_mapped/len(df_q)*100:.0f}%)")
    except Exception as e:
        print(f"⚠ Could not load team NRtg ({e}) — using box+minutes only")
        df_q['team_nrtg'] = float('nan')

    # -----------------------------------------------------------------------
    # 3. Box score contribution per 36 minutes
    # -----------------------------------------------------------------------
    min_safe = df_q['MIN'].clip(lower=0.1)
    raw_box = (df_q['PTS']
               + 1.2 * df_q['REB']   # reduced from 1.5 — less big-man bias
               + 1.5 * df_q['AST']   # reduced from 2.0
               + 1.5 * df_q['STL']   # reduced from 2.0
               + 1.5 * df_q['BLK']   # reduced from 2.0 — blocks were inflating backup bigs
               - df_q['TOV'])
    df_q['box_per36'] = raw_box * (36.0 / min_safe)

    # -----------------------------------------------------------------------
    # 4. Composite score — computed within each season
    # -----------------------------------------------------------------------
    print("Computing composite scores (within-season normalization)...")
    composite_parts = []
    for season, group in df_q.groupby('SEASON'):
        comp = compute_composite_for_group(group)
        composite_parts.append(comp)

    df_q['composite_score'] = pd.concat(composite_parts)

    # -----------------------------------------------------------------------
    # 4b. Career-smoothed composite
    # Blends current-season composite with prior seasons to stabilize tiers.
    # - Prevents one-year breakout spikes from inflating tier
    #   (e.g. a bench player's career year ≠ 3 years of Superstar consistency)
    # - Prevents injury-year drops from demoting elite players
    #   (e.g. a 97-OVR player missing 30 games still rates as Superstar)
    # Weights: 55% current / 30% prior-season / 15% two-seasons-ago
    # Automatically renormalized when prior seasons are unavailable (rookies).
    # For traded players with multiple team stints, uses best composite as reference.
    # -----------------------------------------------------------------------
    print("Computing career-smoothed composites (blending up to 3 seasons)...")
    CAREER_WEIGHTS = [(0, 0.55), (-1, 0.30), (-2, 0.15)]

    comp_lookup = (df_q.groupby(['PLAYER_ID', 'SEASON_YEAR'])['composite_score']
                   .max()
                   .to_dict())

    def smooth_composite(row):
        pid = row['PLAYER_ID']
        yr  = int(row['SEASON_YEAR'])
        total_w, total_val = 0.0, 0.0
        for offset, w in CAREER_WEIGHTS:
            c = comp_lookup.get((pid, yr + offset))
            if c is not None and not np.isnan(float(c)):
                total_w += w
                total_val += w * float(c)
        return total_val / total_w if total_w > 0 else row['composite_score']

    df_q['smoothed_composite'] = df_q.apply(smooth_composite, axis=1)

    # -----------------------------------------------------------------------
    # 5. Within-season percentile and tier  (based on smoothed composite)
    # -----------------------------------------------------------------------
    def add_percentile(group):
        g = group.copy()
        g['within_season_pct'] = g['smoothed_composite'].rank(pct=True) * 100
        return g

    df_q = df_q.groupby('SEASON', group_keys=False).apply(add_percentile)
    df_q['tier'] = df_q['within_season_pct'].apply(assign_tier)

    # -----------------------------------------------------------------------
    # 6. Potential score = smoothed_composite × age_factor × draft_factor
    # -----------------------------------------------------------------------
    df_q['age_factor'] = df_q['AGE'].apply(get_age_factor)
    df_q['draft_factor'] = df_q['OVERALL_PICK'].apply(get_draft_factor)
    df_q['potential_score'] = (df_q['smoothed_composite']
                               * df_q['age_factor']
                               * df_q['draft_factor'])

    # -----------------------------------------------------------------------
    # 7. Select output columns and save
    # -----------------------------------------------------------------------
    out_cols = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION',
                'SEASON', 'SEASON_YEAR', 'AGE', 'birth_year', 'TOTAL_MIN',
                'composite_score', 'smoothed_composite', 'tier', 'within_season_pct',
                'potential_score', 'age_factor', 'draft_factor',
                'OVERALL_PICK', 'DRAFT_YEAR']

    # Include TEAM_NAME if it was captured during collection
    if 'TEAM_NAME' in df_q.columns:
        out_cols.insert(4, 'TEAM_NAME')

    out_df = df_q[[col for col in out_cols if col in df_q.columns]].copy()
    out_df.to_csv('player_ratings.csv', index=False)
    print(f"✓ Saved player_ratings.csv ({len(out_df)} rows, {len(out_df.columns)} columns)")

    # -----------------------------------------------------------------------
    # 8. Validation output
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TIER DISTRIBUTION (all seasons combined)")
    print("=" * 70)
    tier_order = ['Superstar', 'Star', 'Starter', 'Role Player', 'Fringe']
    tier_counts = out_df['tier'].value_counts().reindex(tier_order, fill_value=0)
    total = len(out_df)
    for tier, count in tier_counts.items():
        bar = '█' * int(count / total * 40)
        print(f"  {tier:<12} {count:>5}  {count/total*100:>5.1f}%  {bar}")

    print("\n" + "=" * 70)
    print("SAMPLE SUPERSTARS (top composite scores)")
    print("=" * 70)
    superstars = out_df[out_df['tier'] == 'Superstar'].nlargest(20, 'smoothed_composite')
    print(superstars[['PLAYER_NAME', 'SEASON', 'TEAM_ABBREVIATION',
                       'smoothed_composite', 'composite_score', 'potential_score', 'AGE']
                     ].to_string(index=False))

    print("\n" + "=" * 70)
    print("SUPERSTARS PER SEASON (recent 5 seasons)")
    print("=" * 70)
    recent = out_df[out_df['SEASON_YEAR'] >= out_df['SEASON_YEAR'].max() - 4]
    ss_per_season = (recent[recent['tier'] == 'Superstar']
                     .groupby('SEASON')['PLAYER_NAME']
                     .apply(list))
    for season, players in ss_per_season.items():
        print(f"  {season}: {', '.join(players)}")

    print("\n✓ Next step: run team_profile_builder.py")


if __name__ == "__main__":
    main()
