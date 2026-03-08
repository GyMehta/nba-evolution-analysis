"""
ESPN Team Stats Fetcher  (fallback when stats.nba.com is unreachable)
Fetches 2025-26 team standings + per-game scoring stats from ESPN's public API
and updates nba_team_stats_clean.csv with fresh W/L/NRtg/ORtg/DRtg.

ESPN PPG / OPP-PPG are used as ORtg / DRtg proxies — the rank-inversion
features in the similarity engine only care about within-season relative rank,
which is preserved perfectly by raw per-game scoring differentials.

Run: python -X utf8 fetch_espn_team_stats.py
"""

import requests, pandas as pd, time, warnings
warnings.filterwarnings('ignore')

SEASON      = '2025-26'
SEASON_YEAR = 2026
ERA         = '2020s'
OUT_CSV     = 'nba_team_stats_clean.csv'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://www.espn.com/nba/',
    'Accept': 'application/json',
}


def get_json(url, label=''):
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f'  ERROR ({label}): {e}')
        return None


# ---------------------------------------------------------------------------
# Step 1 — team list  (displayName → id)
# ---------------------------------------------------------------------------
def fetch_team_list():
    data = get_json('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams',
                    'team list')
    if data is None:
        return {}
    teams = {}
    for entry in data['sports'][0]['leagues'][0]['teams']:
        t = entry['team']
        teams[t['displayName']] = t['id']
    print(f'  Team list: {len(teams)} teams')
    return teams   # displayName → id


# ---------------------------------------------------------------------------
# Step 2 — standings (all 30 teams in one call)
# ---------------------------------------------------------------------------
def fetch_standings():
    data = get_json('https://site.api.espn.com/apis/v2/sports/basketball/nba/standings',
                    'standings')
    if data is None:
        return {}

    records = {}
    for conf in data.get('children', []):
        for entry in conf.get('standings', {}).get('entries', []):
            name  = entry['team']['displayName']
            stats = {s['name']: s.get('value') for s in entry.get('stats', [])}
            w  = int(stats.get('wins',   0) or 0)
            l  = int(stats.get('losses', 0) or 0)
            gp = w + l
            records[name] = {
                'Team':    name,
                'GP':      gp,
                'W':       w,
                'L':       l,
                'WIN_PCT': round(w / gp, 4) if gp > 0 else 0.0,
                # PPG / OPP-PPG from standings — same as from per-team stats
                'ORtg':    round(float(stats['avgPointsFor']),   1) if stats.get('avgPointsFor')   else None,
                'DRtg':    round(float(stats['avgPointsAgainst']),1) if stats.get('avgPointsAgainst') else None,
                'NRtg':    round(float(stats['differential']),   1) if stats.get('differential')   else None,
            }
    print(f'  Standings: {len(records)} teams')
    return records


# ---------------------------------------------------------------------------
# Step 3 — per-team stats (30 calls, ~3 seconds total)
# ---------------------------------------------------------------------------
def parse_team_stats(data):
    """Extract per-game stats from ESPN team statistics response."""
    row = {}
    cats = data.get('results', {}).get('stats', {}).get('categories', [])
    for cat in cats:
        for s in cat.get('stats', []):
            name = s.get('name', '')
            val  = s.get('value')
            if val is None:
                continue
            val = float(val)
            if   name == 'gamesPlayed':                    row['GP']   = int(val)
            elif name == 'avgPoints':                      row['PTS']  = round(val, 1)
            elif name == 'fieldGoalPct':                   row['FG%']  = round(val / 100, 4)
            elif name == 'threePointPct':                  row['3P%']  = round(val / 100, 4)
            elif name == 'threePointFieldGoalPct':         row['3P%']  = round(val / 100, 4)
            elif name == 'avgThreePointFieldGoalsMade':    row['FG3M'] = round(val, 1)
            elif name == 'avgThreePointFieldGoalsAttempted': row['3PA']= round(val, 1)
            elif name == 'freeThrowPct':                   row['FT%']  = round(val / 100, 4)
            elif name == 'avgAssists':                     row['AST']  = round(val, 1)
            elif name == 'assistTurnoverRatio':            row['AST_TO'] = round(val, 2)
            elif name == 'avgOffensiveRebounds':           row['OREB'] = round(val, 1)
            elif name == 'avgDefensiveRebounds':           row['DREB'] = round(val, 1)
            elif name == 'avgRebounds':                    row['REB']  = round(val, 1)
    return row


def fetch_all_team_stats(team_ids):
    """Returns dict: displayName → per-game stats dict."""
    results = {}
    for name, tid in sorted(team_ids.items()):
        url  = (f'https://site.api.espn.com/apis/site/v2/sports/basketball/'
                f'nba/teams/{tid}/statistics')
        data = get_json(url, name)
        if data:
            results[name] = parse_team_stats(data)
        else:
            results[name] = {}
        time.sleep(0.08)   # gentle pacing — ESPN can handle much more
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print('=' * 65)
    print('ESPN NBA TEAM STATS FETCHER')
    print(f'Season: {SEASON}')
    print('=' * 65)

    team_ids  = fetch_team_list()          # displayName → ESPN id
    standings = fetch_standings()          # displayName → {W, L, WIN_PCT, ORtg, DRtg, NRtg}

    if not standings:
        print('ERROR: could not fetch standings.')
        return

    print(f'\nFetching per-team stats ({len(team_ids)} teams)...')
    per_team = fetch_all_team_stats(team_ids)

    # -----------------------------------------------------------------------
    # Build rows
    # -----------------------------------------------------------------------
    all_rows = []
    for name in sorted(standings):
        rec   = standings[name]
        stats = per_team.get(name, {})

        pts     = stats.get('PTS', rec.get('ORtg'))
        opp_pts = rec.get('DRtg')                    # from standings differential
        nrtg    = rec.get('NRtg')

        row = {
            'Team':    rec['Team'],
            'SEASON':  SEASON,
            'YEAR':    SEASON_YEAR,
            'ERA':     ERA,
            'GP':      stats.get('GP', rec['GP']),
            'W':       rec['W'],
            'L':       rec['L'],
            'WIN_PCT': rec['WIN_PCT'],
            'ORtg':    pts,
            'DRtg':    opp_pts,
            'NRtg':    nrtg,
            'PTS':     pts,
            'FG%':     stats.get('FG%'),
            '3PA':     stats.get('3PA'),
            '3P%':     stats.get('3P%'),
            'FG3M':    stats.get('FG3M'),
            'FT%':     stats.get('FT%'),
            'AST_TO':  stats.get('AST_TO'),
            # Columns not available from ESPN — leave as NaN
            'eFG%': None, 'TS%': None, 'Pace': None,
            'AST%': None, 'AST_RATIO': None, 'TOV%': None,
            'ORB%': None, 'DRB%': None, 'REB_PCT': None,
            'FTA_RATE': None, 'PIE': None,
            'OPP_OPP_PTS':    opp_pts,
            'OPP_OPP_FG_PCT': None,
        }
        all_rows.append(row)
        nrtg_str = f'{nrtg:+.1f}' if nrtg is not None else 'N/A'
        print(f'  {rec["Team"]:30s}  {rec["W"]}-{rec["L"]}  NRtg={nrtg_str}')

    new_df = pd.DataFrame(all_rows)

    # -----------------------------------------------------------------------
    # Update CSV — replace 2025-26 rows
    # -----------------------------------------------------------------------
    try:
        existing = pd.read_csv(OUT_CSV)
    except FileNotFoundError:
        print(f'ERROR: {OUT_CSV} not found.')
        return

    clean_cols = list(existing.columns)
    for c in clean_cols:
        if c not in new_df.columns:
            new_df[c] = None
    new_df = new_df[[c for c in clean_cols if c in new_df.columns]]

    before  = len(existing[existing['SEASON'] == SEASON])
    updated = pd.concat([existing[existing['SEASON'] != SEASON], new_df],
                        ignore_index=True)
    updated = updated.sort_values(['SEASON', 'Team']).reset_index(drop=True)
    updated.to_csv(OUT_CSV, index=False)

    print(f'\n✓ {OUT_CSV} updated  '
          f'(replaced {before} rows → {len(new_df)} fresh rows; '
          f'total {len(updated)})')

    print('\nTop 8 by NRtg:')
    top8 = new_df.dropna(subset=['NRtg']).nlargest(8, 'NRtg')[['Team', 'W', 'L', 'NRtg', 'PTS']]
    for _, r in top8.iterrows():
        print(f'  {r["Team"]:30s}  {int(r["W"])}-{int(r["L"])}  '
              f'NRtg={r["NRtg"]:+.1f}  PPG={r["PTS"]:.1f}')

    print('\nDone. Next:')
    print('  python -X utf8 player_rating_engine.py')
    print('  python -X utf8 team_profile_builder.py')
    print('  python -X utf8 generate_multiyear_similarity.py')
    print('  python -X utf8 generate_multiyear_report.py')


if __name__ == '__main__':
    main()
