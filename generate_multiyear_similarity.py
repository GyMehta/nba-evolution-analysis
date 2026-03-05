"""
Multi-Year Similarity Engine  (v2 — with trajectory features)
Compares each current team's 3-year window (2023-24, 2024-25, 2025-26)
against every historical 3-year window in the dataset.

WHAT'S NEW IN V2:
  Level features  (25%/35%/40% recency-weighted averages) capture WHERE the
  team is — roster composition, performance level, star quality.

  Trajectory features  (y3 − y1 slope for each level feature) capture HOW
  the team got there — are they rising, peaking, or declining?

  Combined distance = level distance + trajectory distance.
  Trajectory weight ≈ 40% of total signal, matching the intuition that a
  team improving from lottery to title contender is different from a team
  that has always been a contender.

  CSV also exports:
    level_similarity   — match quality on level features only
    trend_similarity   — match quality on trajectory features only
    current/match trend deltas for win_pct, net_rating, n_superstars

Output: nba_multiyear_similarity.csv

Run: python -X utf8 generate_multiyear_similarity.py

Author: GY Mehta
Date: February 2026
"""

import argparse
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Season configuration  (override with --year, e.g.  --year 2025  for 2024-25)
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(description='Multi-year similarity engine')
_parser.add_argument('--year', type=int, default=2026,
                     help='End year of the target season (2026=2025-26, 2025=2024-25, ...)')
_args = _parser.parse_args()

def _seas(end_yr):
    """'2025-26' from end_yr=2026."""
    return f'{end_yr - 1}-{str(end_yr)[2:]}'

ANCHOR_END_YEAR  = _args.year
ANCHOR_SEASON    = _seas(ANCHOR_END_YEAR)       # e.g. '2025-26'
PREV_SEASON      = _seas(ANCHOR_END_YEAR - 1)   # e.g. '2024-25'
PREV2_SEASON     = _seas(ANCHOR_END_YEAR - 2)   # e.g. '2023-24'
PREV3_SEASON     = _seas(ANCHOR_END_YEAR - 3)   # e.g. '2022-23'

if ANCHOR_END_YEAR == 2026:
    SIM_CSV = 'nba_multiyear_similarity.csv'
else:
    slug    = ANCHOR_SEASON.replace('-', '_')
    SIM_CSV = f'nba_{slug}_similarity.csv'

# ---------------------------------------------------------------------------
# Level feature weights  (recency-averaged snapshot of the team)
# ---------------------------------------------------------------------------
FEATURE_WEIGHTS = {
    # 1. Net rating — overall team quality, most predictive single number
    'net_rating':          3.0,
    # 2. Roster quality cascade — composite_score of each player tier
    'top_player_score':    2.5,   # best player on the team
    'second_player_score': 2.0,   # second-best player
    'depth_score':         1.5,   # weighted sum of top-8 composite scores
    'n_superstars':        1.2,   # how many elite players (top ~5 in league)
    # 3. Roster upside
    'roster_potential':    1.2,
    # 4. Win percentage — actual results
    'win_pct':             0.8,
    # 5. Offensive / defensive efficiency rank (inverted so 1.0 = best in league)
    'ortg_rank_inv':       0.5,
    'drtg_rank_inv':       0.5,
    # Context: roster upheaval
    'stars_lost':          0.4,   # star/superstar players lost vs prior season
    'stars_gained':        0.4,   # star/superstar players gained vs prior season
}
FEATURE_COLUMNS = list(FEATURE_WEIGHTS.keys())

# ---------------------------------------------------------------------------
# Trajectory feature weights  (slope = y3 - y1 for each level feature)
# Kept at ~40% of total signal weight (level total = 14.0, trend total = ~6.5)
# ---------------------------------------------------------------------------
TREND_WEIGHTS = {
    'net_rating':          1.5,   # trajectory of team quality — most important arc
    'top_player_score':    1.2,   # is the star improving or declining?
    'win_pct':             1.0,   # winning trajectory
    'n_superstars':        0.8,   # gaining / losing a superstar
    'depth_score':         0.6,   # getting deeper or thinner?
    'second_player_score': 0.5,
    'ortg_rank_inv':       0.4,   # offensive rank trend
    'drtg_rank_inv':       0.4,   # defensive rank trend
    'roster_potential':    0.3,
}

FEATURE_LABELS = {
    'n_superstars':        'superstar presence',
    'top_player_score':    'top player quality',
    'depth_score':         'roster depth',
    'roster_potential':    'roster potential',
    'net_rating':          'net rating',
    'ortg_rank_inv':       'offensive rank',
    'drtg_rank_inv':       'defensive rank',
    'second_player_score': '2nd player quality',
    'win_pct':             'win percentage',
    'stars_lost':          'star players lost',
    'stars_gained':        'star players gained',
}
TREND_LABELS = {col: f'{lbl} trend' for col, lbl in FEATURE_LABELS.items()}

YEAR_WEIGHTS    = [0.25, 0.35, 0.40]          # recency weights for level average
CURRENT_SEASONS = [PREV2_SEASON, PREV_SEASON, ANCHOR_SEASON]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_features(df, feature_cols):
    """Z-score normalize in-place, return (normed_df, means, stds)."""
    means, stds = {}, {}
    df_norm = df.copy()
    for col in feature_cols:
        mean = df[col].mean()
        std  = df[col].std()
        means[col] = mean
        stds[col]  = std if std > 0 else 1.0
        df_norm[col] = (df[col] - means[col]) / stds[col]
    return df_norm, means, stds


def build_weighted_norm(rows_norm, weights_used):
    """Weighted average of normalized single-season rows."""
    w_total = sum(weights_used)
    result = {}
    for col in FEATURE_COLUMNS:
        result[col] = sum(r[col] * w for r, w in zip(rows_norm, weights_used)) / w_total
    return result


def compute_trend(rows_raw, weights_available):
    """
    Trajectory = weighted-endpoint slope: (newest_raw - oldest_raw) / 2.
    If only 2 years are available we still compute y1 - y0.
    Returns dict keyed by FEATURE_COLUMNS.
    """
    trend = {}
    y_first = rows_raw[0]
    y_last  = rows_raw[-1]
    span    = max(len(rows_raw) - 1, 1)
    for col in FEATURE_COLUMNS:
        try:
            trend[col] = (float(y_last[col]) - float(y_first[col])) / span
        except (TypeError, ValueError):
            trend[col] = 0.0
    return trend


# ---------------------------------------------------------------------------
# Tolerance band: features within this Z-score range are treated as identical.
# Only the excess beyond the band contributes to distance.
# 0.25 std ≈ ~1.25 NRtg pts — widened from 0.15 to account for the larger
# distances produced by per-year comparison (vs single averaged-vector).
# ---------------------------------------------------------------------------
FEATURE_TOLERANCE = 0.25


def vec_distance(q_dict, c_dict, weights):
    """
    Weighted Euclidean distance with per-feature tolerance band.
    Differences within FEATURE_TOLERANCE (in Z-score space) are treated as
    zero — only the excess beyond the band is penalised.
    """
    sq = 0.0
    for col in weights:
        diff = abs(q_dict[col] - c_dict[col])
        excess = max(0.0, diff - FEATURE_TOLERANCE)
        sq += (excess * weights[col]) ** 2
    return np.sqrt(sq)


def dist_to_sim(dist):
    """
    Convert banded distance to 0–100.
    Uses gentler denominator (÷0.40) calibrated for per-year comparison:
    dist=0 → 100,  dist=1 → 71,  dist=2 → 56,  dist=3 → 45.
    """
    return 100.0 / (1.0 + 0.40 * dist)


def get_reason(q_norm, m_norm, weights, labels):
    """Return the 2 dimensions where q and m are most similar (smallest gap)."""
    diffs = {col: abs(q_norm.get(col, 0) - m_norm.get(col, 0)) for col in weights}
    sorted_dims = sorted(diffs.items(), key=lambda x: x[1])
    top2 = [labels.get(col, col) for col, _ in sorted_dims[:2]]
    return f"Similar {top2[0]} + {top2[1]}"


def _safe_int(v):
    try:
        f = float(v)
        return int(f) if not np.isnan(f) else 0
    except (TypeError, ValueError):
        return 0


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
    print('3-YEAR SIMILARITY WITH TRAJECTORY FEATURES')
    print(f'Level weights: {YEAR_WEIGHTS[0]*100:.0f}% / {YEAR_WEIGHTS[1]*100:.0f}% / {YEAR_WEIGHTS[2]*100:.0f}%')
    print(f'Trajectory weight: ~40% of total signal')
    print('=' * 70)

    # -----------------------------------------------------------------------
    # 0. Load playoff data and build top-5 player lookup
    # -----------------------------------------------------------------------
    _PLAYOFF_NAME_ALIASES = {
        'Los Angeles Clippers': 'LA Clippers',
        'LA Clippers':          'Los Angeles Clippers',
    }
    playoff_lookup = {}   # (team, season) → playoff_round (0=missed, 5=champion)
    try:
        playoff_df = pd.read_csv('nba_playoff_history.csv')
        for _, row in playoff_df.iterrows():
            val = int(row['playoff_round'])
            playoff_lookup[(row['Team'], row['SEASON'])] = val
            alt = _PLAYOFF_NAME_ALIASES.get(row['Team'])
            if alt:
                playoff_lookup[(alt, row['SEASON'])] = val
        print(f'Loaded playoff data: {len(playoff_lookup)} team-season entries')
    except FileNotFoundError:
        print('nba_playoff_history.csv not found — run collect_playoff_data.py first')

    # -----------------------------------------------------------------------
    # 1. Load and prep team profiles
    # -----------------------------------------------------------------------
    try:
        profiles = pd.read_csv('team_profiles.csv')
    except FileNotFoundError:
        print('team_profiles.csv not found — run team_profile_builder.py first')
        return

    working = profiles[profiles['has_player_data'] == True].copy()

    # -----------------------------------------------------------------------
    # Compute within-season offensive / defensive rank features.
    # ortg_rank_inv: 1.0 = best offense (highest ORtg), 0.0 = worst
    # drtg_rank_inv: 1.0 = best defense (lowest DRtg),  0.0 = worst
    # -----------------------------------------------------------------------
    working['ortg_rank_inv'] = float('nan')
    working['drtg_rank_inv'] = float('nan')
    for season, grp in working.groupby('SEASON'):
        idx = grp.index
        n   = len(idx)
        if n < 2:
            continue
        if 'ORtg' in working.columns:
            o_rank = grp['ORtg'].rank(ascending=False, method='min', na_option='keep')
            working.loc[idx, 'ortg_rank_inv'] = (n - o_rank) / max(n - 1, 1)
        if 'DRtg' in working.columns:
            d_rank = grp['DRtg'].rank(ascending=True,  method='min', na_option='keep')
            working.loc[idx, 'drtg_rank_inv'] = (n - d_rank) / max(n - 1, 1)

    # -----------------------------------------------------------------------
    # Compute stars_lost / stars_gained: how much did the star roster change
    # vs the prior season?  1 = lost/gained one Star+, 2 = two, etc.
    # Uses player_ratings tier data; falls back to 0 if unavailable.
    # -----------------------------------------------------------------------
    working['stars_lost']   = 0.0
    working['stars_gained'] = 0.0
    try:
        _ratings = pd.read_csv('player_ratings.csv')
        _star_tiers = {'Superstar', 'Star'}

        # Build {(abbr, season): set_of_star_player_ids} lookup
        _star_lookup = {}
        for (_abbr, _sea), _grp in _ratings[_ratings['tier'].isin(_star_tiers)].groupby(
                ['TEAM_ABBREVIATION', 'SEASON']):
            _star_lookup[(_abbr, _sea)] = set(_grp['PLAYER_ID'].tolist())

        # Also build abbr_season_to_team (profile Team name ← abbr+season)
        _ast = {}
        for _, _trow in working.iterrows():
            _tp = _trow.get('top_player_name', '')
            if _tp:
                _rr = _ratings[(_ratings['SEASON'] == _trow['SEASON']) &
                               (_ratings['PLAYER_NAME'] == _tp)]
                if len(_rr) > 0:
                    _ast[(_rr.iloc[0]['TEAM_ABBREVIATION'], _trow['SEASON'])] = _trow['Team']

        # For each team-season, compare to prior season
        _all_seasons = sorted(working['SEASON'].unique(),
                              key=lambda s: int(s.split('-')[0]))
        _sea_idx = {s: i for i, s in enumerate(_all_seasons)}

        for idx, trow in working.iterrows():
            _sea  = trow['SEASON']
            _si   = _sea_idx.get(_sea, 0)
            if _si == 0:
                continue
            _prev = _all_seasons[_si - 1]
            # Only compare consecutive seasons (gap = 1 year)
            if int(_sea.split('-')[0]) - int(_prev.split('-')[0]) != 1:
                continue

            # Find the team's abbreviation in both seasons
            _tp = trow.get('top_player_name', '')
            if not _tp:
                continue
            _rr_curr = _ratings[(_ratings['SEASON'] == _sea) &
                                 (_ratings['PLAYER_NAME'] == _tp)]
            if len(_rr_curr) == 0:
                continue
            _abbr_curr = _rr_curr.iloc[0]['TEAM_ABBREVIATION']

            # Find canonical team name in prior season via abbreviation
            _stars_curr = _star_lookup.get((_abbr_curr, _sea), set())

            # For prior season, try the same abbr first (handles stable franchises)
            _stars_prev = _star_lookup.get((_abbr_curr, _prev), set())

            _lost   = len(_stars_prev - _stars_curr)
            _gained = len(_stars_curr - _stars_prev)
            working.loc[idx, 'stars_lost']   = float(_lost)
            working.loc[idx, 'stars_gained'] = float(_gained)

        print(f'Computed stars_lost/gained for {len(working)} team-season profiles')
    except Exception as _e:
        print(f'  Warning: could not compute stars_lost/gained: {_e}')

    for col in FEATURE_COLUMNS:
        if col not in working.columns:
            working[col] = 0.0
        working[col] = working[col].fillna(working[col].mean())

    # Normalize single-season profiles
    working_norm, _, _ = normalize_features(working, FEATURE_COLUMNS)

    print(f'Total single-season profiles: {len(working)}')

    # -----------------------------------------------------------------------
    # 1b. Team name alias map (handles LA / Los Angeles Clippers etc.)
    # -----------------------------------------------------------------------
    abbr_to_canonical = {}
    top5_lookup = {}  # (full_team_name, season) → [top-5 player names by minutes]
    try:
        ratings = pd.read_csv('player_ratings.csv')
        r2526   = ratings[ratings['SEASON'] == ANCHOR_SEASON]
        for abbr in r2526['TEAM_ABBREVIATION'].unique():
            for season in [PREV_SEASON, PREV2_SEASON, PREV3_SEASON]:
                r = ratings[ratings['SEASON'] == season]
                grp = r[r['TEAM_ABBREVIATION'] == abbr]
                if len(grp) == 0:
                    continue
                for tp in grp.nlargest(3, 'TOTAL_MIN')['PLAYER_NAME'].tolist():
                    m = working[(working['SEASON'] == season) &
                                (working['top_player_name'] == tp)]
                    if len(m) == 1:
                        abbr_to_canonical[abbr] = m.iloc[0]['Team']
                        break
                if abbr in abbr_to_canonical:
                    break
    except FileNotFoundError:
        ratings = pd.DataFrame()

    # Build top-5 players by minutes for every (full_team_name, season)
    # Requires cross-referencing player_ratings (TEAM_ABBREVIATION) with
    # team_profiles (Team full name) via the top_player_name bridge.
    if not ratings.empty:
        abbr_season_to_team = {}
        for _, trow in working.iterrows():
            team_full = trow['Team']
            season    = trow['SEASON']
            top_p     = trow.get('top_player_name', '')
            if top_p:
                rr = ratings[(ratings['SEASON'] == season) &
                             (ratings['PLAYER_NAME'] == top_p)]
                if len(rr) > 0:
                    abbr = rr.iloc[0]['TEAM_ABBREVIATION']
                    abbr_season_to_team[(abbr, season)] = team_full
        for (abbr, season), team_full in abbr_season_to_team.items():
            grp  = ratings[(ratings['SEASON'] == season) &
                           (ratings['TEAM_ABBREVIATION'] == abbr)]
            # Top-10 by minutes, then re-sort by composite_score so recognised
            # stars appear before high-minute fringe/role players
            top5 = (grp.nlargest(10, 'TOTAL_MIN')
                       .nlargest(5, 'composite_score')
                       ['PLAYER_NAME'].tolist())
            top5_lookup[(team_full, season)] = top5
        print(f'Built top-5 player lookup: {len(top5_lookup)} team-season entries')

    team_name_alias = {}
    current_2526_teams = working[working['SEASON'] == ANCHOR_SEASON]['Team'].unique()
    for t26 in current_2526_teams:
        r26 = working[(working['SEASON'] == ANCHOR_SEASON) & (working['Team'] == t26)]
        if len(r26) == 0:
            continue
        top_player = r26.iloc[0].get('top_player_name', '')
        if top_player and not ratings.empty:
            try:
                rr = ratings[(ratings['SEASON'] == ANCHOR_SEASON) &
                              (ratings['PLAYER_NAME'] == top_player)]
                if len(rr) > 0:
                    abbr = rr.iloc[0]['TEAM_ABBREVIATION']
                    if abbr in abbr_to_canonical:
                        canonical = abbr_to_canonical[abbr]
                        if canonical != t26:
                            team_name_alias[t26] = canonical
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # 2. Build current team windows
    # -----------------------------------------------------------------------
    current_windows = []

    for team in current_2526_teams:
        hist_name = team_name_alias.get(team, team)

        rows_raw  = []   # raw profile rows (or None)
        rows_norm = []   # normalized rows (or None)
        for season in CURRENT_SEASONS:
            lname = hist_name if season != ANCHOR_SEASON else team
            m_raw  = working[     (working['Team']      == lname) & (working['SEASON']      == season)]
            m_norm = working_norm[(working_norm['Team'] == lname) & (working_norm['SEASON'] == season)]
            rows_raw .append(m_raw .iloc[0] if len(m_raw)  == 1 else None)
            rows_norm.append(m_norm.iloc[0] if len(m_norm) == 1 else None)

        available_raw  = [r for r in rows_raw  if r is not None]
        available_norm = [r for r in rows_norm if r is not None]
        if len(available_raw) < 2:
            print(f'  Skipping {team}: fewer than 2 seasons of data')
            continue

        weights_used = [YEAR_WEIGHTS[i] for i, r in enumerate(rows_raw) if r is not None]

        # Level: weighted avg of normalized features (kept for reason text)
        level_norm = build_weighted_norm(available_norm, weights_used)

        # Per-year normalized dicts (for year-by-year matching)
        year_norms = [
            {col: float(r[col]) for col in FEATURE_COLUMNS} if r is not None else None
            for r in rows_norm
        ]

        # Trajectory: slope in raw units
        trend_raw = compute_trend(available_raw, weights_used)

        # Metadata for CSV
        newest = available_raw[-1]
        oldest = available_raw[0]

        current_windows.append({
            'team':              team,
            'hist_name':         hist_name,
            'seasons_in_window': [CURRENT_SEASONS[i] for i, r in enumerate(rows_raw) if r is not None],
            'level_norm':        level_norm,
            'year_norms':        year_norms,
            'trend_raw':         trend_raw,
            'trend_norm':        {},          # filled after normalizing across all windows
            'weighted_profile':  {col: sum(r[col]*w for r, w in zip(available_raw, weights_used))
                                   / sum(weights_used) for col in FEATURE_COLUMNS},
            'newest_raw':        newest,
            'oldest_raw':        oldest,
            # Key trend deltas for CSV output
            'win_pct_trend':          trend_raw['win_pct'],
            'net_rating_trend':        trend_raw['net_rating'],
            'n_superstars_trend':      trend_raw['n_superstars'],
        })

    print(f'Current team windows built: {len(current_windows)}')

    # -----------------------------------------------------------------------
    # 3. Build historical 3-year windows
    # -----------------------------------------------------------------------
    team_season_map = {}
    season_years    = {}
    for _, row in working.iterrows():
        team_season_map[(row['Team'], row['SEASON'])] = row
        season_years[row['SEASON']] = int(row.get('SEASON_YEAR', row.get('YEAR', 0)))

    # (team, end_year) → season string — used to look up post-window seasons
    team_year_to_season = {}
    for (team, season) in team_season_map:
        yr = season_years.get(season, 0)
        if yr > 0:
            team_year_to_season[(team, yr)] = season

    all_seasons   = sorted(working['SEASON'].unique(), key=lambda s: season_years.get(s, 0))
    query_seasons = set(CURRENT_SEASONS)
    hist_windows  = []

    for team in working['Team'].unique():
        team_seasons = sorted(
            [s for s in all_seasons if (team, s) in team_season_map],
            key=lambda s: season_years.get(s, 0)
        )
        for i in range(len(team_seasons) - 2):
            s0, s1, s2 = team_seasons[i], team_seasons[i+1], team_seasons[i+2]
            if s0 in query_seasons or s1 in query_seasons or s2 in query_seasons:
                continue
            y0, y1, y2 = (season_years.get(s, 0) for s in (s0, s1, s2))
            if y1 - y0 != 1 or y2 - y1 != 1:
                continue

            r0 = team_season_map[(team, s0)]
            r1 = team_season_map[(team, s1)]
            r2 = team_season_map[(team, s2)]

            # Normalized rows
            def _norm(t, s):
                m = working_norm[(working_norm['Team'] == t) & (working_norm['SEASON'] == s)]
                return m.iloc[0] if len(m) == 1 else None

            n0, n1, n2 = _norm(team, s0), _norm(team, s1), _norm(team, s2)
            if n0 is None or n1 is None or n2 is None:
                continue

            level_norm = build_weighted_norm([n0, n1, n2], YEAR_WEIGHTS)
            year_norms_hist = [
                {col: float(n0[col]) for col in FEATURE_COLUMNS},
                {col: float(n1[col]) for col in FEATURE_COLUMNS},
                {col: float(n2[col]) for col in FEATURE_COLUMNS},
            ]
            trend_raw  = compute_trend([r0, r1, r2], YEAR_WEIGHTS)

            total_w = _safe_int(r0.get('W', 0)) + _safe_int(r1.get('W', 0)) + _safe_int(r2.get('W', 0))
            total_l = _safe_int(r0.get('L', 0)) + _safe_int(r1.get('L', 0)) + _safe_int(r2.get('L', 0))

            avg_win_pct    = (YEAR_WEIGHTS[0]*float(r0.get('win_pct', 0) or 0) +
                              YEAR_WEIGHTS[1]*float(r1.get('win_pct', 0) or 0) +
                              YEAR_WEIGHTS[2]*float(r2.get('win_pct', 0) or 0))
            avg_net_rating = (YEAR_WEIGHTS[0]*float(r0.get('net_rating', 0) or 0) +
                              YEAR_WEIGHTS[1]*float(r1.get('net_rating', 0) or 0) +
                              YEAR_WEIGHTS[2]*float(r2.get('net_rating', 0) or 0))

            # Post-window performance: what did this team do the next 1-2 seasons?
            def _next_wp(delta):
                s_next = team_year_to_season.get((team, y2 + delta))
                if s_next:
                    r_next = team_season_map.get((team, s_next))
                    if r_next is not None:
                        return float(r_next.get('win_pct', 0) or 0)
                return float('nan')

            next_1yr_wp = _next_wp(1)
            next_2yr_wp = _next_wp(2)
            next_vals   = [v for v in [next_1yr_wp, next_2yr_wp] if not np.isnan(v)]
            next_avg_wp = float(np.mean(next_vals)) if next_vals else float('nan')

            # Post-window playoff rounds
            def _next_pr(delta):
                s_next = team_year_to_season.get((team, y2 + delta))
                if s_next:
                    return playoff_lookup.get((team, s_next), 0)
                return None

            next_1yr_pr = _next_pr(1)
            next_2yr_pr = _next_pr(2)

            # Success score for optimistic/pessimistic selection.
            # Uses Year 3 (anchor year) playoff round + Year 4 (next season) outcome.
            # Year 3 playoff depth already tells us if this was a peaking team;
            # Year 4 tells us if they sustained or declined immediately after.
            yr3_pr = playoff_lookup.get((team, s2), 0)
            if not np.isnan(next_1yr_wp) and next_1yr_pr is not None:
                next_avg_success = float(yr3_pr) * 0.5 + float(next_1yr_pr) * 0.5 + float(next_1yr_wp)
            elif not np.isnan(next_1yr_wp):
                next_avg_success = float(yr3_pr) * 0.5 + float(next_1yr_wp)
            else:
                # No Year 4 data — fall back to Year 3 playoff round only
                next_avg_success = float(yr3_pr) * 0.5 if yr3_pr else float('nan')

            hist_windows.append({
                'team':            team,
                'season_0':        s0,
                'season_1':        s1,
                'season_2':        s2,
                'year_2':          y2,
                'level_norm':      level_norm,
                'year_norms':      year_norms_hist,
                'trend_raw':       trend_raw,
                'trend_norm':      {},         # filled below
                'total_W':         total_w,
                'total_L':         total_l,
                'avg_win_pct':     avg_win_pct,
                'avg_net_rating':  avg_net_rating,
                'top_player':      r2.get('top_player_name', ''),
                'second_player':   r2.get('second_player_name', ''),
                # Key trend deltas for CSV output
                'win_pct_trend':       trend_raw['win_pct'],
                'net_rating_trend':    trend_raw['net_rating'],
                'n_superstars_trend':  trend_raw['n_superstars'],
                # Post-window performance (basis for optimistic/pessimistic selection)
                'next_1yr_win_pct':     next_1yr_wp,
                'next_2yr_win_pct':     next_2yr_wp,
                'next_avg_win_pct':     next_avg_wp,
                # Post-window playoff rounds
                'next_1yr_playoff_round': next_1yr_pr,
                'next_2yr_playoff_round': next_2yr_pr,
                # Combined success score (playoff-weighted; used for opt/pes selection)
                'next_avg_success':     next_avg_success,
            })

    print(f'Historical 3-year windows built: {len(hist_windows)}')

    # -----------------------------------------------------------------------
    # 4. Normalize trajectory features across historical distribution
    #    (historical windows only → reference distribution)
    # -----------------------------------------------------------------------
    hist_trend_df = pd.DataFrame([h['trend_raw'] for h in hist_windows])[FEATURE_COLUMNS]
    trend_means   = hist_trend_df.mean()
    trend_stds    = hist_trend_df.std().replace(0, 1.0)

    def normalize_trend(trend_raw_dict):
        return {col: (trend_raw_dict[col] - trend_means[col]) / trend_stds[col]
                for col in FEATURE_COLUMNS}

    for h in hist_windows:
        h['trend_norm'] = normalize_trend(h['trend_raw'])
    for c in current_windows:
        c['trend_norm'] = normalize_trend(c['trend_raw'])

    print('Trajectory features normalized.')

    # -----------------------------------------------------------------------
    # 5. Run similarity for each current team window
    # -----------------------------------------------------------------------
    sim_rows = []

    for cw in current_windows:
        team_name  = cw['team']
        q_level    = cw['level_norm']   # kept for reason text
        q_ynorms   = cw['year_norms']   # per-year norms for matching
        q_trend    = cw['trend_norm']
        newest_raw = cw['newest_raw']

        # Compute all distances at once
        level_sims = []
        trend_sims = []
        combined_sims = []

        for h in hist_windows:
            # Per-year level distance: Year1↔Year1, Year2↔Year2, Year3↔Year3
            ld_sq, total_w = 0.0, 0.0
            for i in range(3):
                qn = q_ynorms[i]
                if qn is not None:
                    d = vec_distance(qn, h['year_norms'][i], FEATURE_WEIGHTS)
                    ld_sq   += YEAR_WEIGHTS[i] * d ** 2
                    total_w += YEAR_WEIGHTS[i]
            ld = np.sqrt(ld_sq / total_w) if total_w > 0 else float('inf')

            td = vec_distance(q_trend, h['trend_norm'], TREND_WEIGHTS)
            cd = np.sqrt(ld**2 + td**2)       # combined Euclidean in augmented space
            level_sims   .append(dist_to_sim(ld))
            trend_sims   .append(dist_to_sim(td))
            combined_sims.append(dist_to_sim(cd))

        level_sims    = np.array(level_sims)
        trend_sims    = np.array(trend_sims)
        combined_sims = np.array(combined_sims)

        # Absolute similarity score: 100 = perfect match, real matches typically 30–65
        combined_norm = combined_sims

        # Rank by combined score, take top 10 candidates
        top10_idx = np.argsort(combined_norm)[::-1][:10]

        # Top 3 by similarity
        top3_idx = top10_idx[:3]

        # Optimistic / pessimistic: ranked by playoff-weighted success score
        # (next_avg_success = playoff_round*0.5 + win_pct for 1-2 seasons after window).
        # This ensures championship > deep playoff run > good regular season > decline.
        # Falls back to win% if future data is unavailable.
        # Limit optimistic/pessimistic to the 3 most-similar matches so the
        # "optimistic" comp is always at least as similar as the third-best match.
        # Using a wider pool risks labelling a low-similarity comp as "optimistic"
        # because one lucky playoff run inflates its success score.
        top3_future = [(hi, hist_windows[hi]['next_avg_success']) for hi in top3_idx
                       if not np.isnan(hist_windows[hi]['next_avg_success'])]
        if top3_future:
            optimistic_h_idx  = max(top3_future, key=lambda x: x[1])[0]
            pessimistic_h_idx = min(top3_future, key=lambda x: x[1])[0]
        else:
            # fallback (e.g. windows near dataset edge with no future data)
            top3_win_pcts = [(hi, hist_windows[hi]['avg_win_pct']) for hi in top3_idx]
            optimistic_h_idx  = max(top3_win_pcts, key=lambda x: x[1])[0]
            pessimistic_h_idx = min(top3_win_pcts, key=lambda x: x[1])[0]

        # Build ordered output: top 3 by similarity, then optimistic, then pessimistic
        output_entries = []
        for rank_i, h_idx in enumerate(top3_idx):
            output_entries.append((rank_i + 1, f'top_{rank_i + 1}', h_idx))
        output_entries.append((4, 'optimistic',  optimistic_h_idx))
        output_entries.append((5, 'pessimistic', pessimistic_h_idx))

        for match_rank, match_type, h_idx in output_entries:
            h = hist_windows[h_idx]

            # Reason: pick the 2 dimensions (level) where q and h are closest
            level_reason = get_reason(q_level, h['level_norm'], FEATURE_WEIGHTS, FEATURE_LABELS)
            # Trend reason: which trajectory dimensions are closest
            trend_reason = get_reason(q_trend, h['trend_norm'], TREND_WEIGHTS, TREND_LABELS)

            sim_rows.append({
                # --- Current team ---
                'team':                   team_name,
                'window':                 ' / '.join(CURRENT_SEASONS),
                'seasons_used':           ' / '.join(cw['seasons_in_window']),
                'current_win_pct':        round(float(cw['weighted_profile'].get('win_pct', 0) or 0), 3),
                'current_net_rating':     round(float(cw['weighted_profile'].get('net_rating', 0) or 0), 1),
                'current_n_superstars':   round(float(cw['weighted_profile'].get('n_superstars', 0) or 0), 2),
                'current_n_stars':        int(float(newest_raw.get('n_stars', 0) or 0)),
                'current_top_player':     newest_raw.get('top_player_name', ''),
                'current_2nd_player':     newest_raw.get('second_player_name', ''),
                'current_W':              _safe_int(newest_raw.get('W', 0)),
                'current_L':              _safe_int(newest_raw.get('L', 0)),
                # Trajectory deltas for current team
                'current_win_pct_trend':      round(cw['win_pct_trend'], 3),
                'current_net_rating_trend':   round(cw['net_rating_trend'], 1),
                'current_n_superstars_trend': round(cw['n_superstars_trend'], 2),
                # --- Match ---
                'match_rank':             match_rank,
                'match_type':             match_type,
                'match_team':             h['team'],
                'match_seasons':          f"{h['season_0']} / {h['season_1']} / {h['season_2']}",
                'match_anchor_season':    h['season_2'],
                'match_3yr_W':            h['total_W'],
                'match_3yr_L':            h['total_L'],
                'match_avg_win_pct':      round(h['avg_win_pct'], 3),
                'match_avg_net_rating':   round(h['avg_net_rating'], 1),
                'match_top_player':       h['top_player'],
                'match_2nd_player':       h['second_player'],
                'match_top_5_players':    ', '.join(top5_lookup.get((h['team'], h['season_2']), [])),
                # Trajectory deltas for match team
                'match_win_pct_trend':      round(h['win_pct_trend'], 3),
                'match_net_rating_trend':   round(h['net_rating_trend'], 1),
                'match_n_superstars_trend': round(h['n_superstars_trend'], 2),
                # Post-window outcomes of the matched team (1-2 seasons after window)
                'match_next_1yr_win_pct': (round(h['next_1yr_win_pct'], 3)
                                           if not np.isnan(h['next_1yr_win_pct']) else None),
                'match_next_2yr_win_pct': (round(h['next_2yr_win_pct'], 3)
                                           if not np.isnan(h['next_2yr_win_pct']) else None),
                'match_next_avg_win_pct': (round(h['next_avg_win_pct'], 3)
                                           if not np.isnan(h['next_avg_win_pct']) else None),
                'match_next_1yr_playoff_round': h.get('next_1yr_playoff_round'),
                'match_next_2yr_playoff_round': h.get('next_2yr_playoff_round'),
                # --- Scores ---
                'similarity_score':       round(float(combined_norm[h_idx]), 1),
                'level_similarity':       round(float(level_sims[h_idx]), 1),
                'trend_similarity':       round(float(trend_sims[h_idx]), 1),
                'level_reason':           level_reason,
                'trend_reason':           trend_reason,
            })

    sim_df = pd.DataFrame(sim_rows)
    sim_df = sim_df.sort_values(['team', 'match_rank']).reset_index(drop=True)
    sim_df.to_csv(SIM_CSV, index=False)
    print(f'\nSaved {SIM_CSV} ({len(sim_df)} rows, {len(sim_df.columns)} columns)')

    # -----------------------------------------------------------------------
    # 6. Summary table
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('TOP MATCH PER TEAM  (combined level + trajectory similarity)')
    print('=' * 70)
    top1 = sim_df[sim_df['match_rank'] == 1].sort_values('team')

    arrows = {True: ' rising', False: ' falling', None: ''}
    col_w  = [26, 26, 22, 8, 8, 8]
    hdrs   = ['Current Team', 'Best Match', 'Seasons', 'Comb', 'Level', 'Trend']
    print('  '.join(h.ljust(w) for h, w in zip(hdrs, col_w)))
    print('-' * 100)
    for _, row in top1.iterrows():
        wpt = row['current_win_pct_trend']
        dir_arrow = ('^' if wpt > 0.03 else ('v' if wpt < -0.03 else '-'))
        vals = [
            f"{str(row['team'])[:24]} {dir_arrow}",
            str(row['match_team'])[:25],
            str(row['match_seasons'])[:21],
            f"{row['similarity_score']:.1f}",
            f"{row['level_similarity']:.1f}",
            f"{row['trend_similarity']:.1f}",
        ]
        print('  '.join(v.ljust(w) for v, w in zip(vals, col_w)))

    print('\nDone.')
    print(f'  {SIM_CSV} — 30 teams x 5 rows = 150 rows')
    print('  match_type: top_1/top_2/top_3 (best similarity) + optimistic/pessimistic (from top 10 by win%).')
    print('  Columns include level_similarity + trend_similarity for each match.')


if __name__ == '__main__':
    main()
