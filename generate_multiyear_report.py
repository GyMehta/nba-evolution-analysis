"""
Multi-Year Similarity Report Generator  (v3 — compact card layout)
Reads nba_multiyear_similarity.csv and team_profiles.csv to produce a
self-contained HTML report.

Layout per team:
  - Compact header: record, trend, key players
  - 3-season arc in a single row
  - 3 match cards (best historical comparisons)
  - Optimistic / Pessimistic scenario tags embedded on existing cards
    when they overlap with top-3; shown as separate cards otherwise

Output: nba_multiyear_report.html
Run:    python -X utf8 generate_multiyear_report.py

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
_parser = argparse.ArgumentParser(description='Multi-year similarity report generator')
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
    SIM_CSV  = 'nba_multiyear_similarity.csv'
    HTML_OUT = 'nba_multiyear_report.html'
else:
    slug     = ANCHOR_SEASON.replace('-', '_')
    SIM_CSV  = f'nba_{slug}_similarity.csv'
    HTML_OUT = f'nba_{slug}_report.html'

CURRENT_SEASONS = [PREV2_SEASON, PREV_SEASON, ANCHOR_SEASON]

ROUND_LABELS = {
    0: 'Missed playoffs',
    1: 'First round exit',
    2: 'Conf. Semis',
    3: 'Conf. Finals',
    4: 'Lost in Finals',
    5: 'Won Championship',
}

# Key events by (team_name, season) that explain trajectory divergences.
# Checked for all 3 window seasons + the 2 post-window seasons.
MAJOR_EVENTS = {
    # Warriors dynasty
    ('Golden State Warriors', '2014-15'): 'Curry won MVP — first of 3 titles in 4 years',
    ('Golden State Warriors', '2015-16'): '73-win record season; Kevin Durant signed in free agency',
    ('Golden State Warriors', '2016-17'): 'KD joined — dynasty cemented with 3rd title',
    ('Golden State Warriors', '2017-18'): 'Repeated as champions; KD/Curry/Klay/Draymond core intact',
    ('Golden State Warriors', '2018-19'): 'KD ruptured Achilles in Finals; Klay tore ACL — dynasty ended',
    ('Golden State Warriors', '2019-20'): 'Klay missed 2 full seasons (ACL + Achilles) — bottomed out',
    # LeBron / Cleveland
    ('Cleveland Cavaliers', '2014-15'): 'LeBron returned from Miami; Kyrie Irving core formed',
    ('Cleveland Cavaliers', '2015-16'): 'Won Championship — historic 3-1 comeback against Warriors',
    ('Cleveland Cavaliers', '2016-17'): 'Reached Finals again; Kyrie trade request that summer',
    ('Cleveland Cavaliers', '2017-18'): 'LeBron James departed for LA; franchise collapsed to lottery',
    # Oklahoma City
    ('Oklahoma City Thunder', '2011-12'): 'KD + Westbrook reached Finals; Harden traded away afterward',
    ('Oklahoma City Thunder', '2015-16'): 'Best team in West by record; Kevin Durant left for Golden State',
    ('Oklahoma City Thunder', '2018-19'): 'Paul George + Westbrook both traded; full reset began',
    ('Oklahoma City Thunder', '2022-23'): 'SGA MVP candidate; Chet Holmgren debuted after missed year',
    ('Oklahoma City Thunder', '2023-24'): 'SGA scoring title; young core (Chet, Jalen Williams) — won 2025 title',
    # San Antonio Spurs
    ('San Antonio Spurs', '2002-03'): 'Tim Duncan dynasty: 4 titles in 6 years (1999–2005)',
    ('San Antonio Spurs', '2012-13'): 'Lost 2013 Finals in heartbreaking collapse vs Heat (Game 6)',
    ('San Antonio Spurs', '2013-14'): 'Won Championship — peak Kawhi Leonard era',
    ('San Antonio Spurs', '2015-16'): '67 wins; aging core — Kawhi traded to Toronto 2018',
    ('San Antonio Spurs', '2017-18'): 'Kawhi Leonard injured; trade to Toronto ended dynasty era',
    ('San Antonio Spurs', '2022-23'): 'Victor Wembanyama selected #1 — generational rebuild',
    # Miami Heat
    ('Miami Heat', '2010-11'): 'LeBron + Wade + Bosh superteam formed; lost 2011 Finals',
    ('Miami Heat', '2012-13'): 'Won Championship; 27-game win streak — dynasty peak',
    ('Miami Heat', '2013-14'): 'Reached Finals again; LeBron departed for Cleveland that summer',
    ('Miami Heat', '2019-20'): 'Jimmy Butler led surprising Bubble Finals run — lost to Lakers',
    ('Miami Heat', '2022-23'): '8-seed Finals run; Bam + Butler; declined in following seasons',
    # Chicago Bulls
    ('Chicago Bulls', '2010-11'): 'Derrick Rose won MVP at 22 — youngest ever',
    ('Chicago Bulls', '2011-12'): 'Rose tore ACL in first-round playoffs; never returned to that level',
    # Los Angeles Lakers
    ('Los Angeles Lakers', '2019-20'): 'LeBron + Anthony Davis — Won Championship in the Bubble',
    ('Los Angeles Lakers', '2022-23'): 'LeBron first to 38,000 pts; AD injuries; AD departed 2024',
    # Boston Celtics
    ('Boston Celtics', '2007-08'): 'KG + Pierce + Ray Allen Big Three — Won Championship',
    ('Boston Celtics', '2009-10'): 'Reached Finals; aging core began declining',
    ('Boston Celtics', '2017-18'): 'Hayward ACL game 1; Kyrie injuries — underachieved badly',
    ('Boston Celtics', '2021-22'): 'Reached Finals; coach Ime Udoka suspended that offseason',
    ('Boston Celtics', '2022-23'): 'Added Kristaps Porzingis; Won 2024 Championship',
    # Philadelphia 76ers
    ('Philadelphia 76ers', '2015-16'): '"The Process" bottomed out — drafted Embiid (2014), Simmons (2016)',
    ('Philadelphia 76ers', '2017-18'): 'Embiid + Simmons emerged; rose from 3 wins to 52 wins in 2 years',
    ('Philadelphia 76ers', '2021-22'): 'Ben Simmons refused to play; traded for James Harden',
    ('Philadelphia 76ers', '2022-23'): 'Embiid MVP; Harden trade request; window effectively closed',
    # Toronto Raptors
    ('Toronto Raptors', '2018-19'): 'Won Championship — Kawhi Leonard left for LA Clippers',
    ('Toronto Raptors', '2019-20'): 'Lost Kawhi; Siakam + OG Anunoby carried team respectably',
    # Denver Nuggets
    ('Denver Nuggets', '2019-20'): 'Jokic + Murray Bubble run; Murray tore ACL in April 2021',
    ('Denver Nuggets', '2021-22'): 'Murray + Porter returned from injury; Jokic back-to-back MVP',
    ('Denver Nuggets', '2022-23'): 'Won first Championship — Jokic 3-time MVP; historic achievement',
    # Milwaukee Bucks
    ('Milwaukee Bucks', '2019-20'): 'Giannis won back-to-back MVPs (2019-20)',
    ('Milwaukee Bucks', '2020-21'): 'Won Championship — Giannis dominant clutch performance vs Suns',
    # Phoenix Suns
    ('Phoenix Suns', '2020-21'): 'Chris Paul acquisition — surprise Finals run; declined sharply after',
    ('Phoenix Suns', '2022-23'): 'Added Kevin Durant AND Bradley Beal; all-in experiment failed',
    # Dallas Mavericks
    ('Dallas Mavericks', '2022-23'): 'Luka + Kyrie Irving acquired; reached 2024 Finals',
    ('Dallas Mavericks', '2023-24'): 'Reached Finals; Kyrie departed; Luka traded to LA Lakers 2025',
    # Indiana Pacers
    ('Indiana Pacers', '2023-24'): 'Tyrese Haliburton + Pascal Siakam acquisition — reached ECF',
    # Houston Rockets
    ('Houston Rockets', '2017-18'): 'Harden MVP; CP3 hamstring in G6 vs Warriors — crushing exit',
    ('Houston Rockets', '2018-19'): 'Harden 36 PPG; CP3 departed; window closed',
    # Minnesota Timberwolves
    ('Minnesota Timberwolves', '2023-24'): 'Anthony Edwards breakout; Rudy Gobert trade vindicated — Conf Finals',
    # Sacramento Kings
    ('Sacramento Kings', '2022-23'): "De'Aaron Fox led Kings to playoffs — ended 16-year drought",
    # Brooklyn Nets
    ('Brooklyn Nets', '2020-21'): 'KD + Kyrie + Harden Big Three — Harden traded mid-season',
    ('Brooklyn Nets', '2021-22'): 'KD trade request; Ben Simmons acquired; full roster reset',
    # New York Knicks
    ('New York Knicks', '2022-23'): 'Jalen Brunson signed; reached 2nd round — first deep run in years',
    ('New York Knicks', '2023-24'): 'Added OG Anunoby + Karl-Anthony Towns; Conference Finals run',
    # ── 2025-26 current season ──────────────────────────────────────────────
    # Departures / injuries (detected from player data)
    ('Indiana Pacers', '2025-26'):     'Tyrese Haliburton missed extended time — season derailed by injury',
    ('Milwaukee Bucks', '2025-26'):    'Damian Lillard out for season (Achilles/extended absence)',
    ('Atlanta Hawks', '2025-26'):      'Trae Young departed; Jalen Johnson emerged as new franchise cornerstone',
    ('Boston Celtics', '2025-26'):     'Jayson Tatum missed extended time with injury',
    ('Dallas Mavericks', '2025-26'):   'Anthony Davis departed to Washington; roster in transition after Luka trade',
    ('Sacramento Kings', '2025-26'):   "De'Aaron Fox + Sabonis both declined sharply; franchise reset",
    ('Memphis Grizzlies', '2025-26'):  'Ja Morant continued to miss significant time',
    # Arrivals / breakouts
    ('Cleveland Cavaliers', '2025-26'): 'James Harden acquired from LA Clippers; new Big Three formed',
    ('Houston Rockets', '2025-26'):    'Kevin Durant acquired from Phoenix Suns — contender window opened',
    ('Los Angeles Lakers', '2025-26'): 'Austin Reaves breakout season as legitimate Star alongside LeBron',
    ('Toronto Raptors', '2025-26'):    'Scottie Barnes breakout — emerged as true franchise Star',
    ('Detroit Pistons', '2025-26'):    'Jalen Duren + Cade Cunningham breakout — surprise contender',
    ('San Antonio Spurs', '2025-26'):  "Victor Wembanyama Year 2 leap — franchise ascent accelerating",
    ('Golden State Warriors', '2025-26'): 'Jimmy Butler arrived; new identity around Curry + Butler core',
    ('Oklahoma City Thunder', '2025-26'): 'Chet Holmgren + Isaiah Hartenstein both reached Star tier — deepest OKC core yet',
    ('Philadelphia 76ers', '2025-26'): 'Joel Embiid healthy; Tyrese Maxey reached Star tier — both stars contributing',
    ('Los Angeles Clippers', '2025-26'): 'Kawhi Leonard returned healthy; Darius Garland acquired',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def fmt_nrtg(v):
    try:
        return f'{float(v):+.1f}'
    except Exception:
        return 'N/A'


def fmt_record(w, l):
    try:
        return f'{int(float(w))}-{int(float(l))}'
    except Exception:
        return 'N/A'


def fmt_pct(v):
    """Win% as e.g. '43.5%'"""
    try:
        return f'{float(v)*100:.1f}%'
    except Exception:
        return '—'


def trend_arrow_html(delta, threshold=0.03):
    if delta > threshold:
        return '<span class="arr up">▲</span>'
    elif delta < -threshold:
        return '<span class="arr dn">▼</span>'
    else:
        return '<span class="arr fl">→</span>'


def outcome_color(next_avg, match_avg, pr1=None, pr2=None):
    """CSS class for outcome, playoff-aware. Championship always green."""
    try:
        pr1 = int(pr1) if pr1 is not None and pd.notna(pr1) else None
        pr2 = int(pr2) if pr2 is not None and pd.notna(pr2) else None
        best_pr = max(r for r in [pr1, pr2] if r is not None) if (pr1 is not None or pr2 is not None) else None
        if best_pr is not None:
            if best_pr >= 5: return 'out-champ'   # championship
            if best_pr >= 4: return 'out-up'      # finals appearance = strong outcome
            if best_pr >= 3: return 'out-up'      # conf finals = improvement
            if best_pr >= 2: return 'out-fl'      # conf semis = neutral
            if best_pr == 1: return 'out-fl'      # first round = flat
            if best_pr == 0: return 'out-dn'      # missed = decline
        diff = float(next_avg) - float(match_avg)
        if diff > 0.08:  return 'out-up'
        if diff < -0.08: return 'out-dn'
        return 'out-fl'
    except Exception:
        return 'out-fl'


def playoff_outcome_html(pr1, pr2):
    """Return playoff round label HTML for the best post-window playoff result."""
    try:
        pr1 = int(pr1) if pr1 is not None and pd.notna(pr1) else None
        pr2 = int(pr2) if pr2 is not None and pd.notna(pr2) else None
        rounds = [r for r in [pr1, pr2] if r is not None]
    except Exception:
        rounds = []
    if not rounds:
        return ''
    best = max(rounds)
    label = ROUND_LABELS.get(best, '')
    if not label or best == 0:
        return ''
    star = ' ★' if best == 5 else ''
    css = 'po-champ' if best == 5 else ('po-good' if best >= 3 else 'po-flat')
    return f'<div class="card-playoff {css}">{label}{star}</div>'


def era_str(seasons_str):
    """'1997-98 / 1998-99 / 1999-00' → '1997-98 to 1999-00'"""
    parts = [s.strip() for s in seasons_str.split('/')]
    if len(parts) <= 1 or parts[0] == parts[-1]:
        return parts[0]
    return f'{parts[0]} to {parts[-1]}'


def get_card_events_split(m_team, anchor_season):
    """Return (yr3_events, yr4_events) for a match card.

    yr3_events: notable events from the anchor season itself (injuries, trades mid-season)
    yr4_events: notable offseason/year-4 events (roster moves before the following season)
    """
    yr3_evt = MAJOR_EVENTS.get((m_team, anchor_season))
    yr4_sea = ''
    try:
        start_yr = int(anchor_season.split('-')[0]) + 1
        yr4_sea  = f'{start_yr}-{str(start_yr + 1)[-2:].zfill(2)}'
    except Exception:
        pass
    yr4_evt = MAJOR_EVENTS.get((m_team, yr4_sea)) if yr4_sea else None
    return ([yr3_evt] if yr3_evt else []), ([yr4_evt] if yr4_evt else [])


def annotate_player_names(names, seasons, player_tier_lkp):
    """Return list of player name HTML strings with inline tier markers."""
    result = []
    for name in names:
        tier = None
        for s in seasons:
            t = player_tier_lkp.get((name, s))
            if t in ('Superstar', 'Star'):
                tier = t
                break
        if tier == 'Superstar':
            result.append(f'{name}&nbsp;<span class="ptier-sup">Superstar</span>')
        elif tier == 'Star':
            result.append(f'{name}&nbsp;<span class="ptier-star">Star</span>')
        else:
            result.append(name)
    return result


def build_radar_vals(profile_rows, seasons_list, season_rank_lkp, radar_norm,
                     override_win_pct=None):
    """Return list of 7 normalized [0-100] values for a radar chart dataset.

    profile_rows:     list of working-profile Series for each season in the window
    seasons_list:     season strings matching profile_rows
    season_rank_lkp:  {(team, season): {ortg, drtg}}
    radar_norm:       {field: (min, max)} for global normalization
    override_win_pct: if provided, use this value for Win% axis instead of the
                      window average (used to show a match team's FUTURE win rate)
    """
    if not profile_rows:
        return [50] * 7

    def _avg(field):
        vals = [_safe_float(r.get(field)) for r in profile_rows if r is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def _norm(field, v):
        mn, mx = radar_norm.get(field, (0, 1))
        if mx == mn:
            return 50.0
        return round(max(0.0, min(100.0, (v - mn) / (mx - mn) * 100)), 1)

    def _rank_norm(ranks):
        valid = [r for r in ranks if r is not None]
        if not valid:
            return 50.0
        avg_r = sum(valid) / len(valid)
        return round(max(0.0, min(100.0, (30 - avg_r) / 29 * 100)), 1)

    team_name  = profile_rows[0].get('Team', '') if profile_rows else ''
    ortg_ranks = [season_rank_lkp.get((team_name, s), {}).get('ortg') for s in seasons_list]
    drtg_ranks = [season_rank_lkp.get((team_name, s), {}).get('drtg') for s in seasons_list]

    win_pct_val = override_win_pct if override_win_pct is not None else _avg('win_pct')

    return [
        _norm('win_pct',          win_pct_val),
        _norm('net_rating',       _avg('net_rating')),
        _rank_norm(ortg_ranks),                          # Offense rank -> higher = better
        _rank_norm(drtg_ranks),                          # Defense rank -> higher = better
        _norm('top_player_score', _avg('top_player_score')),
        _norm('depth_score',      _avg('depth_score')),
        _norm('roster_potential', _avg('roster_potential')),
    ]


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0a0a0a;
    color: #e2e2e2;
    padding: 24px 20px;
    max-width: 1180px;
    margin: 0 auto;
}
h1 { text-align:center; font-size:1.65rem; color:#c084fc; margin-bottom:4px; letter-spacing:1px; }
.subtitle { text-align:center; color:#4b5563; font-size:0.82rem; margin-bottom:30px; }

/* ── Team card shell ── */
.team-card {
    background: #121212;
    border: 1px solid #242424;
    border-radius: 12px;
    margin-bottom: 24px;
    overflow: hidden;
}

/* ── Header ── */
.team-hdr {
    background: linear-gradient(135deg,#1a1040 0%,#221660 50%,#1a1040 100%);
    padding: 14px 20px 12px;
    display: flex; justify-content: space-between; align-items: center;
}
.team-hdr-left { flex: 1; }
.team-name { font-size:1.3rem; font-weight:700; color:#e0e7ff; }
.team-record {
    margin-top: 5px;
    display: flex; gap: 14px; align-items: center; flex-wrap: wrap;
    font-size: 0.82rem;
}
.rec-item { color:#94a3b8; }
.rec-val  { color:#d1d5db; font-weight:600; margin-left:4px; }
.rec-players { color:#818cf8; font-size:0.78rem; }
.rec-season-lbl { font-size:0.62rem; font-weight:700; text-transform:uppercase;
                  letter-spacing:.6px; color:#4b5563; margin-right:8px; }
.team-dir { font-size:2rem; font-weight:700; line-height:1; padding-left:12px; }
.team-dir.up { color:#4ade80; } .team-dir.dn { color:#f87171; } .team-dir.fl { color:#6b7280; }

/* ── 3-year arc row ── */
.arc-row {
    background: #0e0e0e;
    border-bottom: 1px solid #1e1e1e;
    padding: 9px 20px;
    display: flex; gap: 6px; align-items: center; flex-wrap: wrap;
    font-size: 0.8rem;
}
.arc-lbl  { color:#4b5563; text-transform:uppercase; letter-spacing:.5px; font-size:0.68rem; margin-right:4px; }
.arc-cell { color:#d1d5db; white-space:nowrap; }
.arc-sep  { color:#334155; }
.arc-curr { color:#a5b4fc; font-weight:600; }

/* ── Section headers ── */
.section-hdr {
    padding: 10px 20px 6px;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: .7px;
    color: #4b5563;
    background: #0e0e0e;
    border-top: 1px solid #1a1a1a;
}

/* ── Card grid ── */
.card-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0;
    border-top: 1px solid #1a1a1a;
}

/* ── Individual match card ── */
.match-card {
    padding: 14px 16px;
    border-right: 1px solid #1a1a1a;
    border-bottom: 1px solid #1a1a1a;
    position: relative;
}
.match-card:last-child { border-right: none; }

.card-top { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:2px; }
.card-rank { font-size:0.75rem; font-weight:700; }
.card-rank.r1 { color:#facc15; } .card-rank.r2 { color:#94a3b8; } .card-rank.r3 { color:#cd7c2f; }
.card-sim  { font-size:0.68rem; color:#4b5563; background:#1a1a2e; padding:2px 6px; border-radius:10px; }

.card-team { font-size:0.98rem; font-weight:700; color:#c7d2fe; margin-bottom:1px; }
.card-era  { font-size:0.72rem; color:#4b5563; margin-bottom:4px; }
.card-nrtg { font-size:0.70rem; font-weight:600; margin-left:6px; }
.card-nrtg-pos { color:#4ade80; }
.card-nrtg-neg { color:#f87171; }
.card-nrtg-neu { color:#94a3b8; }
.card-player { font-size:0.76rem; color:#6b7280; margin-bottom:2px; }
.card-players { font-size:0.71rem; color:#6b7280; margin-bottom:6px; line-height:1.5; }
.card-winpct { font-size:0.76rem; color:#94a3b8; margin-bottom:9px; }

.card-divider { height:1px; background:#1e1e1e; margin-bottom:8px; }

.card-out-lbl { font-size:0.66rem; color:#4b5563; text-transform:uppercase; letter-spacing:.5px; margin-bottom:5px; }
.card-out { font-size:0.9rem; font-weight:700; }
.out-up  { color:#4ade80; }
.out-dn  { color:#f87171; }
.out-fl  { color:#94a3b8; }
.card-out-years { font-size:0.72rem; color:#6b7280; margin-top:2px; }

/* Scenario tags on cards */
.scenario-tags { display:flex; gap:5px; flex-wrap:wrap; margin-top:8px; }
.tag-opt { font-size:0.62rem; font-weight:700; padding:2px 7px; border-radius:10px;
           background:#0d2b0d; color:#4ade80; border:1px solid #1e4a1e; }
.tag-pes { font-size:0.62rem; font-weight:700; padding:2px 7px; border-radius:10px;
           background:#2b1200; color:#fb923c; border:1px solid #4a2000; }

/* Standalone scenario card (when opt/pes is a unique team) */
.scenario-hdr-opt {
    padding: 8px 20px 5px;
    font-size: 0.68rem; text-transform:uppercase; letter-spacing:.7px;
    color: #4ade80; background: #081208; border-top:1px solid #1a1a1a;
}
.scenario-hdr-pes {
    padding: 8px 20px 5px;
    font-size: 0.68rem; text-transform:uppercase; letter-spacing:.7px;
    color: #fb923c; background: #120800; border-top:1px solid #1a1a1a;
}
.card-grid-single {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0;
    border-top: 1px solid #1a1a1a;
}
.match-card.opt-card { border-left: 3px solid #4ade80; background:#0b150b; }
.match-card.pes-card { border-left: 3px solid #fb923c; background:#150800; }

/* Playoff outcome on cards */
.card-playoff { font-size:0.72rem; font-weight:600; margin-top:4px; }
.po-champ { color:#facc15; }
.po-good  { color:#4ade80; }
.po-flat  { color:#94a3b8; }

/* Outcome colors */
.out-champ { color:#facc15; }

/* Radar / comparison chart */
.chart-wrap {
    background: #0e0e0e;
    border-bottom: 1px solid #1e1e1e;
    padding: 10px 20px 8px;
    display: flex; gap: 16px; align-items: flex-start;
}
.chart-canvas-area { flex: 0 0 280px; }
.chart-legend-area {
    flex: 1; display: flex; flex-direction: column; justify-content: center;
    padding: 8px 0;
}
.chart-lbl {
    font-size: 0.62rem; color: #4b5563; text-transform: uppercase;
    letter-spacing: .5px; margin-bottom: 10px;
}
.chart-legend-row {
    display: flex; align-items: center; gap: 6px;
    font-size: 0.71rem; color: #6b7280; margin-bottom: 6px;
}
.chart-legend-dot {
    width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
}
.chart-axis-key {
    margin-top: 12px; font-size: 0.65rem; color: #374151; line-height: 1.7;
}
.chart-axis-key span { display: block; }

/* Arrow spans */
.arr { font-weight:700; }
.arr.up { color:#4ade80; } .arr.dn { color:#f87171; } .arr.fl { color:#6b7280; }

/* Inline player tier badges */
.ptier-sup  {
    color:#78350f; background:#facc15;
    font-size:0.58em; font-weight:700; letter-spacing:.3px;
    padding:1px 5px; border-radius:4px;
    vertical-align:middle; margin-left:3px;
}
.ptier-star {
    color:#0f172a; background:#94a3b8;
    font-size:0.58em; font-weight:700; letter-spacing:.3px;
    padding:1px 5px; border-radius:4px;
    vertical-align:middle; margin-left:3px;
}

/* Key event note on match cards */
.card-event {
    font-size: 0.68rem; color: #6b7280; font-style: italic;
    margin-top: 5px; line-height: 1.4;
}
.card-event-icon { color: #4b5563; font-style:normal; }

/* Year-by-year outcome rows */
.card-yr-section { margin-top: 6px; }
.card-yr-row {
    display: flex; align-items: baseline; gap: 6px;
    margin-top: 5px; font-size: 0.72rem;
}
.yr-sea-lbl {
    font-size: 0.68rem; font-weight: 600; color: #6b7280;
    flex-shrink: 0; width: 52px; font-variant-numeric: tabular-nums;
}
.yr-anchor .yr-sea-lbl { color: #9ca3af; }
.yr-wp  { color: #9ca3af; flex-shrink: 0; }
.yr-po  { font-weight: 600; }
.yr-sep { color: #374151; }
.yr-rank { font-size: 0.62rem; color: #4b5563; margin-left: 4px; }
.card-offseason-lbl {
    font-size: 0.62rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: .4px; color: #4b5563; margin-top: 7px; margin-bottom: 2px;
}
"""


# ---------------------------------------------------------------------------
# Match card renderer
# ---------------------------------------------------------------------------

def _playoff_css(pr):
    """CSS class for a playoff round integer."""
    if pr is None:  return 'out-fl'
    if pr >= 5:     return 'out-champ'
    if pr >= 3:     return 'out-up'
    if pr >= 1:     return 'out-fl'
    return 'out-dn'


def render_match_card(m, rank_label, rank_cls, card_cls='',
                      opt_tag=False, pes_tag=False,
                      player_tier_lkp=None,
                      ortg_rank=None, drtg_rank=None,
                      window_arc=None,
                      yr4_season=None, yr4_win_pct=None, yr4_net_rating=None,
                      yr4_playoff_round=None, yr4_playoff_label=None,
                      yr3_events=None, yr4_events=None):
    """Return HTML for a single compact match card.

    window_arc: list of 3 dicts, one per window season:
        {season, win_pct, playoff_round, playoff_label}
    yr4_season: the season string immediately after the window (e.g. '2014-15')
    """
    m_team   = m['match_team']
    m_seas   = m['match_seasons']
    anchor   = str(m.get('match_anchor_season', '')).strip()
    comb_sim = _safe_float(m.get('similarity_score', 0))
    arc      = window_arc or []

    # Era label from anchor season (Year 3)
    era = era_str(anchor) if anchor else era_str(m_seas)

    # Top-5 players (from match_top_5_players; fall back to top_player)
    top5_raw = m.get('match_top_5_players', '') or ''
    if top5_raw and str(top5_raw) != 'nan':
        players_list = [p.strip() for p in str(top5_raw).split(',') if p.strip()]
    else:
        fallback = m.get('match_top_player', '') or ''
        players_list = [fallback] if fallback else []

    if players_list:
        if player_tier_lkp:
            seas_for_lkp = [s.strip() for s in m_seas.split('/')]
            annotated = annotate_player_names(players_list, seas_for_lkp, player_tier_lkp)
        else:
            annotated = players_list
        players_html = f'<div class="card-players">{" &middot; ".join(annotated)}</div>'
    else:
        players_html = ''

    # ── 3-year arc rows (season string as label) ──
    arc_rows_html = ''
    for i, yr in enumerate(arc):
        sea    = yr.get('season', '')
        wp     = yr.get('win_pct')
        pr     = yr.get('playoff_round')
        pl     = yr.get('playoff_label', '—')
        is_anc = (sea == anchor)   # anchor = Year 3 (last window season)

        wp_str = fmt_pct(wp) if wp is not None else '—'
        po_css = _playoff_css(pr)
        po_lbl = pl or ('Missed playoffs' if pr == 0 else '—')

        # O/D rank shown inline with the anchor (Year 3) row
        rank_extra = ''
        if is_anc and ortg_rank is not None and drtg_rank is not None:
            rank_extra = (f'&nbsp;<span class="yr-rank">'
                          f'Off&nbsp;#{int(round(ortg_rank))}'
                          f'&nbsp;&middot;&nbsp;Def&nbsp;#{int(round(drtg_rank))}</span>')

        nr = yr.get('net_rating')
        if nr is not None and not (nr != nr):
            sign = '+' if nr >= 0 else ''
            nr_cls = 'card-nrtg-pos' if nr > 0.5 else ('card-nrtg-neg' if nr < -0.5 else 'card-nrtg-neu')
            nrtg_row_html = f'<span class="card-nrtg {nr_cls}">{sign}{nr:.1f}</span>'
        else:
            nrtg_row_html = ''

        anchor_cls = ' yr-anchor' if is_anc else ''
        arc_rows_html += (
            f'<div class="card-yr-row{anchor_cls}">'
            f'<span class="yr-sea-lbl">{sea}</span>'
            f'<span class="yr-wp">{wp_str}</span>'
            f'{nrtg_row_html}'
            f'<span class="yr-sep">&middot;</span>'
            f'<span class="yr-po {po_css}">{po_lbl}</span>'
            f'{rank_extra}'
            f'</div>'
        )

    # ── Year-3 in-season events ──
    yr3_evts_html = ''.join(
        f'<div class="card-event"><span class="card-event-icon">&#9889;</span> {e}</div>'
        for e in (yr3_events or [])
    )

    # ── Offseason notes ──
    yr4_evts_html = ''
    if yr4_events:
        yr4_evts_html = '<div class="card-offseason-lbl">Offseason</div>'
        yr4_evts_html += ''.join(
            f'<div class="card-event"><span class="card-event-icon">&#8594;</span> {e}</div>'
            for e in yr4_events
        )

    # ── Year-4 outcome row ──
    has_yr4 = yr4_win_pct is not None and pd.notna(yr4_win_pct)
    yr4_sea_lbl = yr4_season or 'Next year'
    if has_yr4:
        yr4_wp_str = fmt_pct(yr4_win_pct)
        yr4_po_lbl = yr4_playoff_label or ROUND_LABELS.get(yr4_playoff_round, '—')
        yr4_po_css = _playoff_css(yr4_playoff_round)
        if yr4_net_rating is not None and not (yr4_net_rating != yr4_net_rating):
            sign = '+' if yr4_net_rating >= 0 else ''
            nr4_cls = 'card-nrtg-pos' if yr4_net_rating > 0.5 else ('card-nrtg-neg' if yr4_net_rating < -0.5 else 'card-nrtg-neu')
            yr4_nrtg_html = f'<span class="card-nrtg {nr4_cls}">{sign}{yr4_net_rating:.1f}</span>'
        else:
            yr4_nrtg_html = ''
        yr4_row = (
            f'<div class="card-yr-row">'
            f'<span class="yr-sea-lbl">{yr4_sea_lbl}</span>'
            f'<span class="yr-wp">{yr4_wp_str}</span>'
            f'{yr4_nrtg_html}'
            f'<span class="yr-sep">&middot;</span>'
            f'<span class="yr-po {yr4_po_css}">{yr4_po_lbl}</span>'
            f'</div>'
        )
    else:
        yr4_row = (f'<div class="card-yr-row">'
                   f'<span class="yr-sea-lbl">{yr4_sea_lbl}</span>'
                   f'<span class="yr-wp" style="color:#374151;">No data</span>'
                   f'</div>')

    # ── Scenario tags ──
    tags_html = ''
    if opt_tag or pes_tag:
        tags = []
        if opt_tag:
            tags.append('<span class="tag-opt">Optimistic Path</span>')
        if pes_tag:
            tags.append('<span class="tag-pes">Pessimistic Path</span>')
        tags_html = f'<div class="scenario-tags">{"".join(tags)}</div>'

    sim_badge = f'Similarity: {comb_sim:.0f}/100'

    return f'''
  <div class="match-card {card_cls}">
    <div class="card-top">
      <span class="card-rank {rank_cls}">{rank_label}</span>
      <span class="card-sim">{sim_badge}</span>
    </div>
    <div class="card-team">{m_team}</div>
    <div class="card-era">{era}</div>
    {players_html}
    <div class="card-divider"></div>
    <div class="card-out-lbl">3-year arc</div>
    <div class="card-yr-section">
      {arc_rows_html}
      {yr3_evts_html}
    </div>
    <div class="card-out-lbl" style="margin-top:8px;">What followed</div>
    <div class="card-yr-section">
      {yr4_evts_html}
      {yr4_row}
    </div>
    {tags_html}
  </div>'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('Loading data...')
    try:
        profiles = pd.read_csv('team_profiles.csv')
        sim_df   = pd.read_csv(SIM_CSV)
    except FileNotFoundError as e:
        print(f'Missing file: {e}')
        return

    # Playoff lookup: (Team, SEASON) -> (playoff_round_int, round_label_str)
    # Dual-entry for known name changes so either variant resolves correctly.
    _PLAYOFF_NAME_ALIASES = {
        'Los Angeles Clippers': 'LA Clippers',
        'LA Clippers':          'Los Angeles Clippers',
    }
    playoff_lkp = {}
    try:
        po_df = pd.read_csv('nba_playoff_history.csv')
        for _, row in po_df.iterrows():
            val = (int(row['playoff_round']), str(row['round_label']))
            playoff_lkp[(row['Team'], row['SEASON'])] = val
            alt = _PLAYOFF_NAME_ALIASES.get(row['Team'])
            if alt:
                playoff_lkp[(alt, row['SEASON'])] = val
    except FileNotFoundError:
        pass

    working = profiles[profiles['has_player_data'] == True].copy()
    profile_idx = {(r['Team'], r['SEASON']): idx for idx, r in working.iterrows()}

    # Team name alias (handles franchise relocations / name changes between eras)
    try:
        ratings = pd.read_csv('player_ratings.csv')
    except FileNotFoundError:
        ratings = pd.DataFrame()

    abbr_to_canonical = {}
    if not ratings.empty:
        for abbr in ratings[ratings['SEASON'] == ANCHOR_SEASON]['TEAM_ABBREVIATION'].unique():
            for season in [PREV_SEASON, PREV2_SEASON, PREV3_SEASON]:
                grp = ratings[(ratings['SEASON'] == season) &
                              (ratings['TEAM_ABBREVIATION'] == abbr)]
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

    team_name_alias = {}
    for t26 in working[working['SEASON'] == ANCHOR_SEASON]['Team'].unique():
        r26 = working[(working['SEASON'] == ANCHOR_SEASON) & (working['Team'] == t26)]
        if len(r26) == 0:
            continue
        tp = r26.iloc[0].get('top_player_name', '')
        if tp and not ratings.empty:
            try:
                rr = ratings[(ratings['SEASON'] == ANCHOR_SEASON) &
                             (ratings['PLAYER_NAME'] == tp)]
                if len(rr) > 0:
                    abbr = rr.iloc[0]['TEAM_ABBREVIATION']
                    if abbr in abbr_to_canonical:
                        c = abbr_to_canonical[abbr]
                        if c != t26:
                            team_name_alias[t26] = c
            except Exception:
                pass

    # Build top-5 current players (anchor season) for each team
    current_top5_lookup = {}   # team_name → [player_names]
    if not ratings.empty:
        for t26 in working[working['SEASON'] == ANCHOR_SEASON]['Team'].unique():
            r26 = working[(working['SEASON'] == ANCHOR_SEASON) & (working['Team'] == t26)]
            if len(r26) == 0:
                continue
            top_p = r26.iloc[0].get('top_player_name', '')
            if top_p:
                rr = ratings[(ratings['SEASON'] == ANCHOR_SEASON) &
                             (ratings['PLAYER_NAME'] == top_p)]
                if len(rr) > 0:
                    abbr = rr.iloc[0]['TEAM_ABBREVIATION']
                    grp  = ratings[(ratings['SEASON'] == ANCHOR_SEASON) &
                                   (ratings['TEAM_ABBREVIATION'] == abbr)]
                    # Display order (within top-10 by minutes):
                    #  1. Superstars / Stars  (≥ 800 min)  — by minutes desc
                    #  2. Starters            (≥ 1000 min) — by minutes desc
                    #     1000-min floor removes part-time Starters (e.g. Mitchell Robinson
                    #     873 min) while keeping full contributors (OG Anunoby 1584 min).
                    #  3. Role Players / Fringe (≥ 800 min) — by minutes desc
                    #     Keeps high-minute role players like Dillon Brooks (1530 min).
                    top10 = grp.nlargest(10, 'TOTAL_MIN')
                    stars_df    = top10[(top10['tier'].isin({'Superstar', 'Star'}))    & (top10['TOTAL_MIN'] >= 800)].sort_values('TOTAL_MIN', ascending=False)
                    starters_df = top10[(top10['tier'] == 'Starter')                  & (top10['TOTAL_MIN'] >= 1000)].sort_values('TOTAL_MIN', ascending=False)
                    others_df   = top10[(~top10['tier'].isin({'Superstar', 'Star', 'Starter'})) & (top10['TOTAL_MIN'] >= 800)].sort_values('TOTAL_MIN', ascending=False)
                    ordered     = pd.concat([stars_df, starters_df, others_df])
                    current_top5_lookup[t26] = ordered.head(5)['PLAYER_NAME'].tolist()

    # Build auto-detected roster change events per team (ANCHOR_SEASON vs PREV_SEASON)
    # Detects Star/Superstar players who departed or arrived between seasons.
    # "Departed" = was a Star/Superstar last season AND is no longer on the team at all.
    # "Arrived"  = is a Star/Superstar this season AND was not on the team at all last season.
    roster_changes_lookup = {}   # team_name → {'departed': [names], 'arrived': [names]}
    if not ratings.empty and 'tier' in ratings.columns:
        _star_tiers = {'Superstar', 'Star'}
        _star_name_lkp = {}   # (abbr, season) → {player_name: tier}  — Stars/Superstars only
        _all_player_lkp = {}  # (abbr, season) → set of all player names on that team
        # Minimum minutes for a player to count as a meaningful star arrival/departure.
        # Filters out injured players or those with a small sample (e.g. Robert Williams).
        _MIN_STAR_MINUTES = 800
        for (_abbr, _sea), _grp in ratings.groupby(['TEAM_ABBREVIATION', 'SEASON']):
            _all_player_lkp[(_abbr, _sea)] = set(_grp['PLAYER_NAME'].tolist())
            stars = _grp[_grp['tier'].isin(_star_tiers) & (_grp['TOTAL_MIN'] >= _MIN_STAR_MINUTES)]
            if len(stars):
                _star_name_lkp[(_abbr, _sea)] = {
                    row['PLAYER_NAME']: row['tier']
                    for _, row in stars.iterrows()
                }

        for t26 in working[working['SEASON'] == ANCHOR_SEASON]['Team'].unique():
            r26 = working[(working['SEASON'] == ANCHOR_SEASON) & (working['Team'] == t26)]
            if len(r26) == 0:
                continue
            top_p = r26.iloc[0].get('top_player_name', '')
            abbr = None
            if top_p:
                rr = ratings[(ratings['SEASON'] == ANCHOR_SEASON) &
                             (ratings['PLAYER_NAME'] == top_p)]
                if len(rr) > 0:
                    abbr = rr.iloc[0]['TEAM_ABBREVIATION']
            if abbr is None:
                continue

            curr_stars   = _star_name_lkp.get((abbr, ANCHOR_SEASON), {})
            prev_stars   = _star_name_lkp.get((abbr, PREV_SEASON), {})
            curr_players = _all_player_lkp.get((abbr, ANCHOR_SEASON), set())
            prev_players = _all_player_lkp.get((abbr, PREV_SEASON), set())

            # Only flag as departed if the player is genuinely gone from the roster
            departed = [n for n in prev_stars if n not in curr_players]
            # Only flag as arrived if the player wasn't on this team last season at all
            arrived  = [n for n in curr_stars  if n not in prev_players]

            if departed or arrived:
                roster_changes_lookup[t26] = {
                    'departed': departed,
                    'arrived':  arrived,
                }

    # Win% by (team, season) — kept for potential future use
    win_pct_by_season = {}
    for _, row in working.iterrows():
        wp = _safe_float(row.get('win_pct'))
        win_pct_by_season[(row['Team'], row['SEASON'])] = wp

    # Profile row by (team, season) — for radar chart data lookup
    profile_by_key = {}
    for _, row in working.iterrows():
        profile_by_key[(row['Team'], row['SEASON'])] = row

    # Within-season ORtg/DRtg ranks for all team-seasons in `working`
    # ortg rank 1 = best offense; drtg rank 1 = best defense (lowest DRtg)
    season_rank_lkp = {}  # (team, season) → {'ortg': rank, 'drtg': rank}
    for season, grp in working[working['has_player_data'] == True].groupby('SEASON'):
        grp_o = grp.dropna(subset=['ORtg']).sort_values('ORtg', ascending=False).reset_index(drop=True)
        grp_d = grp.dropna(subset=['DRtg']).sort_values('DRtg', ascending=True).reset_index(drop=True)
        for i, row in grp_o.iterrows():
            season_rank_lkp[(row['Team'], season)] = {'ortg': i + 1, 'drtg': None}
        for i, row in grp_d.iterrows():
            key = (row['Team'], season)
            if key in season_rank_lkp:
                season_rank_lkp[key]['drtg'] = i + 1
            else:
                season_rank_lkp[key] = {'ortg': None, 'drtg': i + 1}

    # Global normalization bounds for radar axes
    radar_fields = ['win_pct', 'net_rating', 'top_player_score', 'depth_score', 'roster_potential']
    radar_norm = {}
    for f in radar_fields:
        col = working[f].dropna()
        mn, mx = float(col.min()), float(col.max())
        radar_norm[f] = (mn, mx if mx > mn else mn + 1)

    # Player tier lookup: (player_name, season) → tier
    # Used for inline tier markers on player names in cards and headers
    player_season_tier = {}
    if not ratings.empty and 'tier' in ratings.columns:
        for _, row in ratings.iterrows():
            player_season_tier[(row['PLAYER_NAME'], row['SEASON'])] = row['tier']

    current_teams = sorted(sim_df['team'].unique())
    print(f'Building report for {len(current_teams)} teams...')

    sections = []

    for team in current_teams:
        hist_name    = team_name_alias.get(team, team)
        team_matches = sim_df[sim_df['team'] == team].sort_values('match_rank')
        if team_matches.empty:
            continue

        # ── Fetch 3-year arc rows ──
        arc_raw = {}
        for season in CURRENT_SEASONS:
            lname = hist_name if season != ANCHOR_SEASON else team
            key   = (lname, season)
            arc_raw[season] = working.loc[profile_idx[key]] if key in profile_idx else None

        latest = arc_raw.get(ANCHOR_SEASON) if arc_raw.get(ANCHOR_SEASON) is not None else arc_raw.get(PREV_SEASON)

        # ── Trend metrics ──
        first_row   = team_matches.iloc[0]
        win_trend   = _safe_float(first_row.get('current_win_pct_trend', 0))
        nrtg_trend  = _safe_float(first_row.get('current_net_rating_trend', 0))
        overall_dir = ('up' if win_trend > 0.03 else ('dn' if win_trend < -0.03 else 'fl'))
        dir_char    = {'up':'▲','dn':'▼','fl':'→'}[overall_dir]

        # ── Header ──
        # Current-season stats come from the anchor (or prev) profile row
        curr_season_row = arc_raw.get(ANCHOR_SEASON) if arc_raw.get(ANCHOR_SEASON) is not None else arc_raw.get(PREV_SEASON)
        curr_W   = int(first_row.get('current_W', 0))
        curr_L   = int(first_row.get('current_L', 0))
        curr_season_wp = _safe_float(curr_season_row.get('win_pct', 0)) if curr_season_row is not None else 0.0
        curr_season_nr = _safe_float(curr_season_row.get('net_rating', 0)) if curr_season_row is not None else 0.0
        n_sup    = int(_safe_float(first_row.get('current_n_superstars', 0)))
        n_star   = int(_safe_float(first_row.get('current_n_stars', 0)))

        # Top-5 players from current roster lookup; fall back to top-2 from similarity CSV
        top5_curr = current_top5_lookup.get(team, [])
        if not top5_curr:
            tp = first_row.get('current_top_player', '') or ''
            sp = first_row.get('current_2nd_player', '') or ''
            top5_curr = [p for p in [tp, sp] if p and p != '—']

        # Annotate current players with inline tier markers (★ for superstar, ◆ for star)
        annotated_curr = annotate_player_names(top5_curr, [ANCHOR_SEASON], player_season_tier)
        players_str = ' &middot; '.join(annotated_curr) if annotated_curr else '—'
        tier_part = ''   # tier info now shown inline per player

        # Auto-detected roster change row (departed/arrived Star+ players vs prior season)
        rc = roster_changes_lookup.get(team, {})
        rc_parts = []
        if rc.get('departed'):
            d_names = ', '.join(rc['departed'])
            rc_parts.append(f'<span style="color:#f87171;">&#8595; Out: {d_names}</span>')
        if rc.get('arrived'):
            a_names = ', '.join(rc['arrived'])
            rc_parts.append(f'<span style="color:#4ade80;">&#8593; In: {a_names}</span>')
        roster_changes_html = (
            f'<div class="rec-players" style="margin-top:3px;font-size:0.74rem;">'
            f'{"&nbsp;&nbsp;·&nbsp;&nbsp;".join(rc_parts)}</div>'
        ) if rc_parts else ''

        # Current-season event note (from MAJOR_EVENTS)
        curr_event = MAJOR_EVENTS.get((team, ANCHOR_SEASON))
        curr_event_html = (
            f'<div class="card-event" style="margin-top:4px;">'
            f'<span class="card-event-icon">&#9889;</span> {curr_event}</div>'
        ) if curr_event else ''

        # O/D rank for current team (anchor season or most recent season)
        curr_ortg_rank = season_rank_lkp.get((team, ANCHOR_SEASON), {}).get('ortg')
        curr_drtg_rank = season_rank_lkp.get((team, ANCHOR_SEASON), {}).get('drtg')
        rank_display = ''
        if curr_ortg_rank is not None and curr_drtg_rank is not None:
            rank_display = (f'<span><span class="rec-item">Off</span>'
                            f'<span class="rec-val">#{curr_ortg_rank}</span></span>'
                            f'<span><span class="rec-item">Def</span>'
                            f'<span class="rec-val">#{curr_drtg_rank}</span></span>')

        html = f'''
<div class="team-card">
  <div class="team-hdr">
    <div class="team-hdr-left">
      <div class="team-name">{team}</div>
      <div class="team-record">
        <span class="rec-season-lbl">{ANCHOR_SEASON}</span>
        <span><span class="rec-item">Record</span><span class="rec-val">{curr_W}-{curr_L}</span></span>
        <span><span class="rec-item">Win rate</span><span class="rec-val">{fmt_pct(curr_season_wp)}</span></span>
        <span><span class="rec-item">Net Rtg</span><span class="rec-val">{fmt_nrtg(curr_season_nr)}</span></span>
        {rank_display}
      </div>
      <div class="team-record" style="margin-top:4px;">
        <span class="rec-season-lbl">3-yr trend</span>
        <span class="rec-item">Win%</span><span class="rec-val" style="margin-left:4px;">{trend_arrow_html(win_trend)}</span>
        <span class="rec-item" style="margin-left:10px;">Net Rtg</span><span class="rec-val" style="margin-left:4px;">{trend_arrow_html(nrtg_trend, 1.0)}</span>
      </div>
      <div class="rec-players" style="margin-top:4px;">{players_str}{tier_part}</div>
      {roster_changes_html}
      {curr_event_html}
    </div>
    <div class="team-dir {overall_dir}">{dir_char}</div>
  </div>
'''

        # ── 3-year arc (compact single row) ──
        arc_parts = []
        for si, season in enumerate(CURRENT_SEASONS):
            r = arc_raw.get(season)
            if r is not None:
                rec     = fmt_record(r.get('W'), r.get('L'))
                wp      = fmt_pct(r.get('win_pct'))
                is_curr = (season == ANCHOR_SEASON)

                # Playoff result for this season
                hist_team = team_name_alias.get(team, team) if season != ANCHOR_SEASON else team
                po_data = playoff_lkp.get((hist_team, season))
                if po_data:
                    pr, pl = po_data
                    po_css = _playoff_css(pr)
                    po_str = f'&nbsp;<span class="{po_css}" style="font-weight:600;font-size:0.65rem;">{pl}</span>'
                elif is_curr:
                    po_str = ''   # season in progress — no playoff result yet
                else:
                    po_str = '&nbsp;<span style="color:#4b5563;font-size:0.65rem;">Missed playoffs</span>'

                nr = _safe_float(r.get('net_rating'))
                if nr is not None and not (nr != nr):
                    sign = '+' if nr >= 0 else ''
                    nr_cls = 'card-nrtg-pos' if nr > 0.5 else ('card-nrtg-neg' if nr < -0.5 else 'card-nrtg-neu')
                    nrtg_str = f'&nbsp;<span class="card-nrtg {nr_cls}">{sign}{nr:.1f}</span>'
                else:
                    nrtg_str = ''

                cell = (f'<span class="arc-cell arc-curr">{season}: {rec} ({wp}){nrtg_str}{po_str}</span>'
                        if is_curr else
                        f'<span class="arc-cell">{season}: {rec} ({wp}){nrtg_str}{po_str}</span>')
            else:
                cell = f'<span class="arc-cell" style="color:#2d2d2d;">{season}: —</span>'
            arc_parts.append(cell)

        def fmt_delta(v, unit=''):
            sign = '+' if v >= 0 else ''
            return f'{sign}{v:.2f}{unit}'

        arc_inner = ' <span class="arc-sep">→</span> '.join(arc_parts)
        html += f'''
  <div class="arc-row">
    <span class="arc-lbl">3-yr arc</span>
    {arc_inner}
    <span class="arc-sep" style="margin-left:8px;">·</span>
    <span style="font-size:0.72rem;color:#4b5563;margin-left:4px;">
      Win% {trend_arrow_html(win_trend)} {fmt_delta(win_trend)}
      &nbsp; NRtg {trend_arrow_html(nrtg_trend,1.0)} {fmt_delta(nrtg_trend,' pts')}
    </span>
  </div>
'''

        # ── Profile radar chart ──
        # Shows the 7-dimension team profile of current team vs historical matches.
        # Axes: Win%, Net Rtg, Offense rank, Defense rank, Top Player, Depth, Potential.
        # All values normalized 0-100 where higher = better.

        def js_arr(vals):
            return '[' + ','.join(str(round(v, 1)) for v in vals) + ']'

        def get_match_future_win_pct(mt_name, mt_seas):
            """Average Win% for the 1-2 seasons after the match window."""
            if not mt_seas:
                return None
            last_sea = mt_seas[-1].strip()
            try:
                end_yr = int(last_sea.split('-')[0]) + 1   # '2019-20' -> 2020
            except (ValueError, IndexError):
                return None
            future_wps = []
            for delta in (1, 2):
                yr  = end_yr + delta
                yr2 = str(yr + 1)[-2:]
                s   = f'{yr}-{yr2}'
                row = profile_by_key.get((mt_name, s))
                if row is not None:
                    v = _safe_float(row.get('win_pct'))
                    if v and v > 0:
                        future_wps.append(v)
            return sum(future_wps) / len(future_wps) if future_wps else None

        def get_match_radar(m_row):
            mt_name  = m_row['match_team']
            mt_seas  = [s.strip() for s in m_row['match_seasons'].split('/')]
            # Use only the LAST season of the window (most recent snapshot)
            last_sea = mt_seas[-1] if mt_seas else None
            last_row = profile_by_key.get((mt_name, last_sea)) if last_sea else None
            rows     = [last_row] if last_row is not None else []
            seas     = [last_sea] if last_row is not None else []
            # Win% axis shows future performance (where this team went after the window)
            future_wp = get_match_future_win_pct(mt_name, mt_seas)
            return build_radar_vals(rows, seas, season_rank_lkp, radar_norm,
                                    override_win_pct=future_wp)

        # Current team: only current season (anchor) — shows where they are RIGHT NOW
        curr_row_only = arc_raw.get(ANCHOR_SEASON)
        if curr_row_only is None:
            curr_row_only = arc_raw.get(PREV_SEASON)
        curr_rows  = [curr_row_only] if curr_row_only is not None else []
        curr_seas  = [ANCHOR_SEASON] if arc_raw.get(ANCHOR_SEASON) is not None else [PREV_SEASON]
        curr_rvals = build_radar_vals(curr_rows, curr_seas, season_rank_lkp, radar_norm)

        team_id = ''.join(c if c.isalnum() else '-' for c in team.lower())

        # Build match datasets: top1 (gray), optimistic (green), pessimistic (red)
        radar_datasets = []
        added_keys     = set()
        legend_html    = ''

        RADAR_MATCHES = [
            ('top_1',       '#facc15',              'rgba(250,204,21,0.07)',  False, 'Best match'),
            ('optimistic',  '#fb923c',              'rgba(251,146,60,0.07)',  True,  'Optimistic path'),
            ('pessimistic', '#f87171',              'rgba(248,113,113,0.07)', True,  'Pessimistic path'),
        ]

        # Collect match rows and detect overlaps before building datasets
        match_info = []   # list of (mt, bc, bg, dashed, lbl_pfx, m_row)
        key_first_role = {}   # key -> (bc, lbl_pfx) of the first role that claimed it
        for mt, bc, bg, dashed, lbl_pfx in RADAR_MATCHES:
            mt_row = team_matches[team_matches['match_type'] == mt]
            if mt_row.empty:
                continue
            m   = mt_row.iloc[0]
            key = (m['match_team'], m['match_seasons'])
            match_info.append((mt, bc, bg, dashed, lbl_pfx, m, key))
            if key not in key_first_role:
                key_first_role[key] = (bc, lbl_pfx)

        for mt, bc, bg, dashed, lbl_pfx, m, key in match_info:
            era_lbl  = era_str(m['match_seasons'])
            is_dupe  = key in added_keys   # already plotted under a different role
            first_bc, first_lbl = key_first_role[key]

            if is_dupe:
                # Don't add another dataset line — just note the overlap in the legend
                legend_html += (
                    f'<div class="chart-legend-row">'
                    f'<div class="chart-legend-dot" style="background:{bc};'
                    f'border:1px dashed {bc};"></div>'
                    f'<span>{lbl_pfx} <span style="color:#9ca3af;font-size:0.8em;">'
                    f'(same as {first_lbl})</span></span></div>'
                )
                continue

            added_keys.add(key)
            vals  = get_match_radar(m)
            lbl   = f"{lbl_pfx}: {m['match_team']} ({era_lbl})".replace('"', "'")
            dash_s = '[5,3]' if dashed else '[]'
            radar_datasets.append(
                f'{{"label":"{lbl}","data":{js_arr(vals)},'
                f'"borderColor":"{bc}","backgroundColor":"{bg}",'
                f'"borderDash":{dash_s},"borderWidth":1.8,"pointRadius":2}}'
            )
            legend_html += (
                f'<div class="chart-legend-row">'
                f'<div class="chart-legend-dot" style="background:{bc};'
                f'{"border:1px dashed "+bc+";" if dashed else ""}"></div>'
                f'<span>{lbl_pfx}</span></div>'
            )

        # Current team last (rendered on top)
        curr_lbl = team.replace('"', "'")
        radar_datasets.append(
            f'{{"label":"{curr_lbl}","data":{js_arr(curr_rvals)},'
            f'"borderColor":"#ffffff","backgroundColor":"rgba(255,255,255,0.09)",'
            f'"borderWidth":2.5,"pointRadius":3}}'
        )
        legend_html = (
            f'<div class="chart-legend-row">'
            f'<div class="chart-legend-dot" style="background:#ffffff;"></div>'
            f'<span style="color:#e2e2e2;font-weight:600;">{team}</span></div>'
        ) + legend_html

        chart_js = f"""
<div class="chart-wrap">
  <div class="chart-canvas-area" style="position:relative;height:240px;">
    <canvas id="ch-{team_id}"></canvas>
  </div>
  <div class="chart-legend-area">
    <div class="chart-lbl">Team profile — all dimensions normalized 0&ndash;100</div>
    {legend_html}
    <div class="chart-axis-key">
      <span>&ndash; <b>Win %</b> current team: this season &middot; matches: avg of next 1&ndash;2 seasons after window</span>
      <span>&ndash; <b>Net Rtg</b> point diff per 100 possessions</span>
      <span>&ndash; <b>Offense</b> &amp; <b>Defense</b> rank among 30 teams (1 = best)</span>
      <span>&ndash; <b>Top Player</b> best player composite score</span>
      <span>&ndash; <b>Depth</b> weighted top-8 roster quality</span>
      <span>&ndash; <b>Potential</b> age &amp; draft-adjusted upside</span>
    </div>
  </div>
</div>
<script>
(function(){{
  new Chart(document.getElementById('ch-{team_id}'),{{
    type:'radar',
    data:{{
      labels:['Win %','Net Rtg','Offense','Defense','Top Player','Depth','Potential'],
      datasets:[{','.join(radar_datasets)}]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{
        legend:{{display:false}},
        tooltip:{{callbacks:{{label:function(c){{return c.dataset.label+': '+c.parsed.r.toFixed(0)+'/100'}}}}}}
      }},
      scales:{{
        r:{{
          min:0,max:100,
          grid:{{color:'rgba(255,255,255,0.05)'}},
          angleLines:{{color:'rgba(255,255,255,0.07)'}},
          pointLabels:{{color:'#6b7280',font:{{size:9}}}},
          ticks:{{display:false,backdropColor:'transparent'}}
        }}
      }}
    }}
  }});
}})();
</script>
"""
        html += chart_js

        # ── Deduplication: find which match_type maps to which (team, seasons) ──
        key_of = {}   # match_type → (team, seasons) unique key
        for _, m in team_matches.iterrows():
            mt  = m.get('match_type', '')
            key = (m['match_team'], m['match_seasons'])
            key_of[mt] = key

        top3_keys = {key_of.get(f'top_{i}') for i in range(1, 4) if key_of.get(f'top_{i}')}
        opt_key   = key_of.get('optimistic')
        pes_key   = key_of.get('pessimistic')

        opt_in_top3 = (opt_key in top3_keys)
        pes_in_top3 = (pes_key in top3_keys)

        # Which top-3 ranks get scenario tags?
        card_opt_ranks = set()
        card_pes_ranks = set()
        if opt_in_top3:
            for i in range(1, 4):
                if key_of.get(f'top_{i}') == opt_key:
                    card_opt_ranks.add(i)
        if pes_in_top3:
            for i in range(1, 4):
                if key_of.get(f'top_{i}') == pes_key:
                    card_pes_ranks.add(i)

        # ── Top 3 match cards ──
        html += '<div class="section-hdr">Most similar teams in NBA history — 3-year profile + trajectory</div>'
        html += '<div class="card-grid">'

        rank_labels = {1: ('#1 Best Match', 'r1'),
                       2: ('#2 Best Match', 'r2'),
                       3: ('#3 Best Match', 'r3')}

        def _card_data(m_row):
            """Pre-compute 3-year arc + Year-4 data for a match card row."""
            mt    = m_row['match_team']
            anc   = str(m_row.get('match_anchor_season', '')).strip()
            seas  = [s.strip() for s in str(m_row.get('match_seasons', '')).split('/')]

            # ── 3-year arc: win% and playoff for each window season ──
            window_arc = []
            for sea in seas:
                prof = profile_by_key.get((mt, sea))
                wp   = _safe_float(prof.get('win_pct')) if prof is not None else None
                if (mt, sea) in playoff_lkp:
                    pr, pl = playoff_lkp[(mt, sea)]
                else:
                    pr, pl = 0, 'Missed playoffs'
                nr = _safe_float(prof.get('net_rating')) if prof is not None else None
                window_arc.append({'season': sea, 'win_pct': wp, 'net_rating': nr,
                                   'playoff_round': pr, 'playoff_label': pl})

            # O/D rank from anchor (Year 3) season
            yr3_o = season_rank_lkp.get((mt, anc), {}).get('ortg')
            yr3_d = season_rank_lkp.get((mt, anc), {}).get('drtg')

            # ── Year 4: season string, win%, playoff ──
            yr4_sea, yr4_wp, yr4_pr, yr4_pl = '', None, None, None
            try:
                anc_start = int(anc.split('-')[0]) + 1
                yr4_sea   = f'{anc_start}-{str(anc_start + 1)[-2:].zfill(2)}'
            except Exception:
                pass

            yr4_wp_raw = m_row.get('match_next_1yr_win_pct')
            yr4_wp = _safe_float(yr4_wp_raw) if yr4_wp_raw is not None and pd.notna(yr4_wp_raw) else None

            if yr4_sea and (mt, yr4_sea) in playoff_lkp:
                yr4_pr, yr4_pl = playoff_lkp[(mt, yr4_sea)]
            else:
                pr_raw = m_row.get('match_next_1yr_playoff_round')
                if pr_raw is not None and pd.notna(pr_raw):
                    yr4_pr = int(pr_raw)
                    yr4_pl = ROUND_LABELS.get(yr4_pr, '—')
                elif yr4_wp is not None:
                    yr4_pr, yr4_pl = 0, 'Missed playoffs'

            # Year 4 net rating from profile
            yr4_prof = profile_by_key.get((mt, yr4_sea)) if yr4_sea else None
            yr4_nr = _safe_float(yr4_prof.get('net_rating')) if yr4_prof is not None else None

            # ── Events split by year ──
            yr3_evts, yr4_evts = get_card_events_split(mt, anc)

            return dict(
                window_arc=window_arc,
                ortg_rank=yr3_o, drtg_rank=yr3_d,
                yr4_season=yr4_sea,
                yr4_win_pct=yr4_wp,
                yr4_net_rating=yr4_nr,
                yr4_playoff_round=yr4_pr, yr4_playoff_label=yr4_pl,
                yr3_events=yr3_evts, yr4_events=yr4_evts,
            )

        for _, m in team_matches[team_matches['match_type'].isin(
                ['top_1', 'top_2', 'top_3'])].iterrows():
            rank    = int(m['match_rank'])
            rl, rc  = rank_labels.get(rank, (f'#{rank}', ''))
            opt_tag = rank in card_opt_ranks
            pes_tag = rank in card_pes_ranks
            cd      = _card_data(m)
            html += render_match_card(m, rank_label=rl, rank_cls=rc,
                                      opt_tag=opt_tag, pes_tag=pes_tag,
                                      player_tier_lkp=player_season_tier, **cd)

        html += '</div>'  # card-grid

        # ── Standalone optimistic card (if not already in top 3) ──
        if not opt_in_top3:
            opt_row = team_matches[team_matches['match_type'] == 'optimistic']
            if not opt_row.empty:
                html += '<div class="scenario-hdr-opt">&#8593; Optimistic Scenario — the similar team that went on to improve the most</div>'
                html += '<div class="card-grid-single">'
                om = opt_row.iloc[0]
                cd = _card_data(om)
                html += render_match_card(om, rank_label='&#8593; Optimistic',
                                          rank_cls='', card_cls='opt-card',
                                          player_tier_lkp=player_season_tier, **cd)
                html += '</div>'

        # ── Standalone pessimistic card (if not already in top 3) ──
        if not pes_in_top3:
            pes_row = team_matches[team_matches['match_type'] == 'pessimistic']
            if not pes_row.empty:
                html += '<div class="scenario-hdr-pes">&#8595; Pessimistic Scenario — the similar team that declined the most afterward</div>'
                html += '<div class="card-grid-single">'
                pm = pes_row.iloc[0]
                cd = _card_data(pm)
                html += render_match_card(pm, rank_label='&#8595; Pessimistic',
                                          rank_cls='', card_cls='pes-card',
                                          player_tier_lkp=player_season_tier, **cd)
                html += '</div>'

        html += '</div>'  # team-card
        sections.append(html)

    # ── Assemble HTML doc ──
    n_teams = len(working[working['SEASON'] == ANCHOR_SEASON])
    body    = '\n'.join(sections)

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>NBA 3-Year Trajectory Similarity — {ANCHOR_SEASON}</title>
<style>{CSS}</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>
<h1>NBA Franchise Trajectory Report — {ANCHOR_SEASON}</h1>
<p class="subtitle">
  Each team's 3-year profile (roster quality, win rate, net rating + trajectory) matched against every historical 3-year window in NBA history &nbsp;·&nbsp;
  {n_teams} teams vs 748 historical windows (1995–2023)
</p>
<p class="subtitle" style="margin-top:-18px; color:#374151;">
  <strong style="color:#6b7280;">Similarity score</strong> &nbsp;(0–100): how closely a historical team's 3-year profile and trajectory matched the current team —
  100 = best match found in the dataset across 7 dimensions: superstar presence, top player quality, roster depth, win rate, net rating, roster potential, and trend direction
</p>
{body}
</body>
</html>"""

    with open(HTML_OUT, 'w', encoding='utf-8') as f:
        f.write(doc)

    print(f'Saved {HTML_OUT}  ({len(sections)} team sections)')
    print('Open in any browser.')


if __name__ == '__main__':
    main()
