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
    'Elite Star':  95.0,   # top ~3.8%  → ~10-12 players per season
    #              Near-superstar tier: top-10-ish players in the league who anchor
    #              a team's playoff ceiling without yet reaching true Superstar status.
    #              Examples: Anthony Edwards (97.8%), Kawhi Leonard, Cade Cunningham.
    #              Distinguishes conference-finals-caliber aces from Star-depth players.
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

    PIE is available only in 2015-16 and 2025-26 (the two seasons where the
    Advanced stats endpoint returned it). For all other seasons the fallback
    is used unchanged from the original formula.

    When PIE is available:
        composite = 0.35×PIE + 0.18×rel_NET_RATING + 0.15×MPG + 0.12×USG_PCT
                  + 0.10×rel_DEF_RATING + 0.06×EFF + 0.04×box_per36

    rel_NET_RATING = player_NET_RATING − team_NRtg, stripping teammate quality.
    A +14 on OKC and +1 on Atlanta both reduce to ~+3 if those teams are
    +11 and 0 respectively — equal individual contributions.

    MPG (MIN per game) reflects how much the coaching staff trusts a player.
    Genuine starters (32-36 mpg) are rated above bench contributors (16-22 mpg)
    regardless of per-minute efficiency. Directly penalises T.J. McConnell,
    Robert Williams III and other efficient bench players who inflate per-36
    metrics but are not stars.

    rel_DEF_RATING = player_DEF_RATING − team_DRtg (inverted: lower = better
    defender). Captures two-way players like Scottie Barnes who anchor team
    defense beyond what steals/blocks show.

    USG_PCT (usage rate) penalises low-usage rim-protectors whose raw box
    numbers inflate per-36 stats, rewards high-usage guards/wings who carry
    offensive load.

    EFF_per36 approximation: NBA-style efficiency using TS% to estimate missed
    shots (FGA−FGM + FTA−FTM equivalent) without needing raw shot-count data.

    Fallback (no PIE — most seasons):
        composite = 0.50×box_per36 + 0.30×min_norm + 0.20×team_nrtg
        (original formula — unchanged to preserve cross-era calibration)

    Box score weights: REB 0.75 (reduced from 1.2 to dampen big-man inflation
    in the box component, which is only 4–20% of the total composite).
    """
    g = group.copy()

    box_norm = normalize_within_season(g['box_per36'].fillna(g['box_per36'].median()))
    min_norm  = normalize_within_season(g['TOTAL_MIN'])

    # MPG (minutes per game): coaching trust signal, rewards genuine starters
    mpg_norm = normalize_within_season(g['MIN'].fillna(g['MIN'].median()))

    # Use relative NET_RATING (player minus team baseline) to strip out teammate
    # quality. A +14 on OKC and a +1 on Atlanta can represent equal individual
    # contributions if both teams differ by ~13 points in team NRtg.
    if 'team_nrtg' in g.columns and g['team_nrtg'].notna().sum() > 0:
        rel_net = g['NET_RATING'] - g['team_nrtg']
    else:
        rel_net = g['NET_RATING']
    net_norm  = normalize_within_season(rel_net.fillna(rel_net.median()))

    pie = g['PIE']
    has_pie = pie.notna() & (pie != 0)

    composite = pd.Series(np.nan, index=g.index)

    if has_pie.sum() > 0:
        pie_norm = normalize_within_season(pie.fillna(pie.median()))

        # USG_PCT: penalises low-usage bigs, rewards high-usage initiators
        usg_norm = normalize_within_season(
            g['USG_PCT'].fillna(g['USG_PCT'].median()) if 'USG_PCT' in g.columns
            else pd.Series(0.5, index=g.index)
        )

        # EFF_per36 approximation via TS%:
        #   missed_per36 ≈ PTS × (1/(2×TS%) − 0.5) × (36/MIN)
        if 'TS_PCT' in g.columns:
            ts_safe   = g['TS_PCT'].fillna(g['TS_PCT'].median()).clip(lower=0.30)
            min_safe  = g['MIN'].clip(lower=0.1)
            raw_eff   = (g['PTS'] + g['REB'] + g['AST'] + g['STL'] + g['BLK'] - g['TOV'])
            missed36  = g['PTS'] * (1.0 / (2.0 * ts_safe) - 0.5) * (36.0 / min_safe)
            eff_per36 = raw_eff * (36.0 / min_safe) - missed36
            eff_norm  = normalize_within_season(eff_per36.fillna(eff_per36.median()))
        else:
            eff_norm = box_norm

        # rel_DEF_RATING: lower DEF_RATING = better defender relative to team.
        # Invert so that a player who defends better than their team baseline
        # scores higher. Only available in PIE-seasons (DEF_RATING from Advanced).
        if ('DEF_RATING' in g.columns and 'team_drtg' in g.columns
                and g['DEF_RATING'].notna().sum() > 0):
            rel_def = g['team_drtg'] - g['DEF_RATING']   # positive = better than team
            def_norm = normalize_within_season(rel_def.fillna(0.0))
        else:
            def_norm = pd.Series(0.5, index=g.index)

        composite[has_pie] = (0.35 * pie_norm[has_pie]
                              + 0.18 * net_norm[has_pie]
                              + 0.15 * mpg_norm[has_pie]
                              + 0.12 * usg_norm[has_pie]
                              + 0.10 * def_norm[has_pie]
                              + 0.06 * eff_norm[has_pie]
                              + 0.04 * box_norm[has_pie])

    # Fallback (no PIE): original formula — preserves historical calibration
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
    elif pct >= TIER_THRESHOLDS['Elite Star']:
        return 'Elite Star'
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


# Manual overrides for players whose draft position is missing from the API
# (DraftHistory API gaps for recent drafts or older players).
# Keyed by PLAYER_NAME (exact match from nba_api).
DRAFT_POSITION_OVERRIDES = {
    # 2025 Draft — not in DraftHistory API at time of data collection
    'Cooper Flagg':       1,
    'Kon Knueppel':       5,   # Charlotte Hornets ~#5 pick
    'Dylan Harper':       2,
    'Tre Johnson':        3,
    'VJ Edgecombe':       4,
    'Ace Bailey':         7,
    'Noa Essengue':       8,
    'Khaman Maluach':     9,
    'Kasparas Jakucionis': 10,
    'Collin Murray-Boyles': 11,
    'Egor Demin':         14,
    # 2020 Draft
    'Anthony Edwards':    1,    # 2020 pick #1 (Minnesota Timberwolves)
    'LaMelo Ball':        3,    # 2020 pick #3 (Charlotte Hornets)
    'Deni Avdija':        9,    # 2020 pick #9
    'Tyrese Haliburton':  12,   # 2020 pick #12
    'Desmond Bane':       30,   # 2020 pick #30
    # 2021 Draft
    'Evan Mobley':        3,    # 2021 pick #3 (Cleveland Cavaliers)
    'Jalen Suggs':        5,    # 2021 pick #5 (Orlando Magic)
    'Jonathan Kuminga':   7,    # 2021 pick #7 (Golden State Warriors)
    'Alperen Sengun':     16,   # 2021 pick #16 (Houston Rockets)
    'Jalen Johnson':      20,   # 2021 pick #20 (Atlanta Hawks)
    'Jalen Williams':     34,   # 2021 pick #34 (OKC Thunder — second round)
    'Herb Jones':         35,   # 2021 pick #35
    'Isaiah Jackson':     22,   # 2021 pick #22
    'Ayo Dosunmu':        38,   # 2021 pick #38
    # 2022 Draft
    'Paolo Banchero':     1,    # 2022 pick #1 (Orlando Magic)
    'Jalen Duren':        13,   # 2022 pick #13 (Detroit Pistons via Charlotte)
    'Ochai Agbaji':       14,   # 2022 pick #14
    'AJ Griffin':         16,   # 2022 pick #16
    'Mark Williams':      15,   # 2022 pick #15
    'Jaden Hardy':        37,   # 2022 second round
    'Max Christie':       35,   # 2022 pick #35
    # 2024 Draft — verify data coverage
    'Zaccharie Risacher': 1,
    'Alex Sarr':          2,
    'Reed Sheppard':      3,
    'Stephon Castle':     4,    # 2024 pick #4 (San Antonio Spurs)
    'Donovan Clingan':    5,    # 2024 pick #5 (Portland)
    'Tidjane Salaun':     6,
    'Matas Buzelis':      11,
    'Ron Holland II':     5,
    'Kel\'el Ware':       15,   # 2024 pick #15 (Miami Heat)
    'Rob Dillingham':     8,    # 2024 pick #8 (Minnesota)
    'Isaiah Collier':     9,    # 2024 pick #9 (Utah)
    'Cody Williams':      10,   # 2024 pick #10
    'Dalton Knecht':      17,   # 2024 pick #17 (LA Lakers)
    'Derik Queen':        14,   # 2024 pick (New Orleans area)
    'Ajay Mitchell':      38,   # 2024 second round (OKC)
    # Veterans with known data gaps
    'Jaylen Brown':       3,    # 2016 pick #3 (Boston Celtics)
    'Jayson Tatum':       3,    # 2017 pick #3 (Boston Celtics)
    'Bam Adebayo':        14,   # 2017 pick #14
    'Anfernee Simons':    24,   # 2018 pick #24
    'De\'Aaron Fox':      5,    # 2017 pick #5
    'Tyrese Maxey':       21,   # 2020 pick #21
    'Cade Cunningham':    1,    # 2021 pick #1
    'Jalen Green':        2,    # 2021 pick #2
    'Scottie Barnes':     4,    # 2021 pick #4
    'Franz Wagner':       8,    # 2021 pick #8
    'Josh Giddey':        6,    # 2021 pick #6
    'Chet Holmgren':      2,    # 2022 pick #2
    'Jabari Smith Jr.':   3,    # 2022 pick #3
    'Keegan Murray':      4,    # 2022 pick #4
    'Bennedict Mathurin': 6,    # 2022 pick #6
    'Shaedon Sharpe':     7,    # 2022 pick #7
    'Dyson Daniels':      8,    # 2022 pick #8
    'Victor Wembanyama':  1,    # 2023 pick #1
    'Brandon Miller':     2,    # 2023 pick #2
    'Scoot Henderson':    3,    # 2023 pick #3
    'Amen Thompson':      4,    # 2023 pick #4
    'Ausar Thompson':     5,    # 2023 pick #5
    'Jarace Walker':      6,    # 2023 pick #6
    'Anthony Black':      7,    # 2023 pick #7
    'Bilal Coulibaly':    8,    # 2023 pick #8
    'Gradey Dick':        13,   # 2023 pick #13
    'Cason Wallace':      10,   # 2023 pick #10
    'Keyonte George':     16,   # 2023 pick #16
    'Alexandre Sarr':     2,    # 2024 (duplicate check)
    'Zach Edey':          13,   # 2024 pick #13
    'Jared McCain':       16,   # 2024 pick #16
}


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

    print(f"OK: Loaded {len(df)} rows from player_stats_by_season.csv")

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
    print(f"OK: Qualified after minutes filter: {len(df_q)} rows "
          f"(removed {len(df) - len(df_q)} low-minute players)")

    # -----------------------------------------------------------------------
    # 2b. Join team net rating and defensive rating as context signals.
    # rel_NET = player_NET - team_NRtg strips teammate quality.
    # rel_DEF = player_DEF_RATING - team_DRtg measures individual defense
    #           vs team baseline (negative = better than team average).
    # Primary source: nba_team_stats_1980_2025.csv (has TEAM_ID + NRtg/DRtg).
    # Fallback for 2025-26: nba_team_stats_clean.csv via team name bridge.
    # -----------------------------------------------------------------------
    team_nrtg_by_id = {}    # (TEAM_ID, SEASON) → NRtg
    team_drtg_by_id = {}    # (TEAM_ID, SEASON) → DRtg
    team_name_by_id = {}    # TEAM_ID → canonical name (for name bridge)
    nrtg_by_name = {}       # (Team, SEASON) → NRtg  (clean CSV, has 2025-26)
    drtg_by_name = {}       # (Team, SEASON) → DRtg

    try:
        raw_ts = pd.read_csv('nba_team_stats_1980_2025.csv',
                             usecols=['TEAM_ID', 'Team', 'SEASON', 'NRtg', 'DRtg'])
        for _, row in raw_ts.iterrows():
            key = (int(row['TEAM_ID']), row['SEASON'])
            team_nrtg_by_id[key] = row['NRtg']
            team_drtg_by_id[key] = row['DRtg']
            team_name_by_id[int(row['TEAM_ID'])] = row['Team']

        clean_ts = pd.read_csv('nba_team_stats_clean.csv',
                               usecols=['Team', 'SEASON', 'NRtg', 'DRtg'])
        nrtg_by_name = clean_ts.set_index(['Team', 'SEASON'])['NRtg'].to_dict()
        drtg_by_name = clean_ts.set_index(['Team', 'SEASON'])['DRtg'].to_dict()

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

        def get_team_drtg(row):
            tid = int(row['TEAM_ID'])
            season = row['SEASON']
            drtg = team_drtg_by_id.get((tid, season))
            if drtg is not None and not pd.isna(drtg):
                return float(drtg)
            tname = team_name_by_id.get(tid)
            if tname:
                drtg = drtg_by_name.get((tname, season))
                if drtg is not None and not pd.isna(drtg):
                    return float(drtg)
            return float('nan')

        df_q['team_nrtg'] = df_q.apply(get_team_nrtg, axis=1)
        df_q['team_drtg'] = df_q.apply(get_team_drtg, axis=1)
        n_mapped = df_q['team_nrtg'].notna().sum()
        print(f"OK: Team NRtg mapped: {n_mapped}/{len(df_q)} player-seasons "
              f"({n_mapped/len(df_q)*100:.0f}%)")
    except Exception as e:
        print(f"⚠ Could not load team NRtg ({e}) — using box+minutes only")
        df_q['team_nrtg'] = float('nan')
        df_q['team_drtg'] = float('nan')

    # -----------------------------------------------------------------------
    # 3. Box score contribution per 36 minutes
    # -----------------------------------------------------------------------
    min_safe = df_q['MIN'].clip(lower=0.1)
    raw_box = (df_q['PTS']
               + 0.75 * df_q['REB']  # reduced from 1.2 — rebounds over-inflate bigs
               + 1.5  * df_q['AST']
               + 1.5  * df_q['STL']
               + 1.5  * df_q['BLK']  # blocks still valued but USG/EFF dampens backup-big inflation
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
    CAREER_WEIGHTS = [(0, 0.65), (-1, 0.25), (-2, 0.10)]

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
    # Apply manual overrides for players whose draft position is missing from the API
    def _corrected_pick(row):
        override = DRAFT_POSITION_OVERRIDES.get(row['PLAYER_NAME'])
        if override is not None:
            return override
        return row['OVERALL_PICK']
    df_q['OVERALL_PICK'] = df_q.apply(_corrected_pick, axis=1)
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
    print(f"OK: Saved player_ratings.csv ({len(out_df)} rows, {len(out_df.columns)} columns)")

    # -----------------------------------------------------------------------
    # 8. Validation output
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TIER DISTRIBUTION (all seasons combined)")
    print("=" * 70)
    tier_order = ['Superstar', 'Elite Star', 'Star', 'Starter', 'Role Player', 'Fringe']
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

    print("\nOK: Next step: run team_profile_builder.py")


if __name__ == "__main__":
    main()
