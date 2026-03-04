"""
Prediction Accuracy Report — 2025-26 Season Check
How well did the 2024-25 3-year-similarity predictions hold up
against actual 2025-26 team performance?

Reads:  nba_2024_25_similarity.csv   (predictions made from 2024-25 window)
        team_profiles.csv             (actual 2025-26 data)
        nba_playoff_history.csv       (2024-25 playoff results)

Output: nba_prediction_accuracy_2025_26.html

Author: GY Mehta
Date: March 2026
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SIM_CSV   = 'nba_2024_25_similarity.csv'
HTML_OUT  = 'nba_prediction_accuracy_2025_26.html'

# Team name aliases (sim 2024-25 naming → team_profiles 2025-26 naming)
TEAM_ALIAS = {'LA Clippers': 'Los Angeles Clippers'}

ROUND_LABELS = {
    0: 'Missed playoffs', 1: 'First round exit', 2: 'Conf. Semis',
    3: 'Conf. Finals', 4: 'Lost in Finals', 5: 'Won Championship ★',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v, default=np.nan):
    try:
        f = float(v)
        return f if not np.isnan(f) else default
    except (TypeError, ValueError):
        return default


def fmt_pct(v, default='—'):
    try:
        return f'{float(v)*100:.1f}%'
    except Exception:
        return default


def fmt_record(w, l):
    try:
        return f'{int(float(w))}-{int(float(l))}'
    except Exception:
        return '—'


def grade_prediction(actual_wp, top1_wp, pes_wp, opt_wp):
    """Return (letter, short_label, css_cls).

    Grades are based on how close the best-match prediction was,
    AND whether the actual result fell inside the predicted range.
    """
    if np.isnan(actual_wp) or np.isnan(top1_wp):
        return 'N/A', 'No data', 'grade-na'

    lo = min(_safe_float(pes_wp, top1_wp), top1_wp)
    hi = max(_safe_float(opt_wp, top1_wp), top1_wp)
    in_range  = lo <= actual_wp <= hi
    abs_err   = abs(actual_wp - top1_wp)

    if in_range and abs_err < 0.07:
        return 'A', 'On Target', 'grade-a'
    elif in_range and abs_err < 0.13:
        return 'B', 'In Range', 'grade-b'
    elif abs_err < 0.08:
        return 'B', 'Near Best', 'grade-b'
    elif in_range or abs_err < 0.16:
        return 'C', 'Reasonable', 'grade-c'
    elif abs_err < 0.23:
        return 'D', 'Off', 'grade-d'
    else:
        return 'F', 'Way Off', 'grade-f'


def scenario_label(actual_wp, top1_wp, pes_wp, opt_wp, top1_pr):
    """Which scenario does the actual result look like?"""
    if np.isnan(actual_wp):
        return None, None
    mid  = _safe_float(top1_wp, actual_wp)
    lo   = _safe_float(pes_wp, mid)
    hi   = _safe_float(opt_wp, mid)
    span = hi - lo if hi > lo else 1.0
    pos  = (actual_wp - lo) / span   # 0 = pessimistic end, 1 = optimistic end

    if pos >= 0.7:
        return 'Optimistic path', 'scen-opt'
    elif pos <= 0.3:
        return 'Pessimistic path', 'scen-pes'
    else:
        return 'On expected path', 'scen-mid'


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0a0a0a; color: #e2e2e2;
    padding: 24px 20px; max-width: 1180px; margin: 0 auto;
}
h1 { text-align:center; font-size:1.65rem; color:#c084fc; margin-bottom:4px; letter-spacing:1px; }
.subtitle { text-align:center; color:#4b5563; font-size:0.82rem; margin-bottom:20px; }
.note { text-align:center; color:#374151; font-size:0.75rem; margin-bottom:28px; font-style:italic; }

/* ── Summary stats bar ── */
.summary-bar {
    display: flex; gap: 16px; justify-content: center; flex-wrap: wrap;
    margin-bottom: 28px;
}
.stat-box {
    background: #121212; border: 1px solid #242424; border-radius: 10px;
    padding: 14px 20px; text-align: center; min-width: 130px;
}
.stat-val { font-size: 1.8rem; font-weight: 700; line-height: 1; }
.stat-lbl { font-size: 0.7rem; color: #4b5563; text-transform: uppercase;
            letter-spacing: .6px; margin-top: 4px; }
.stat-green { color: #4ade80; }
.stat-yellow { color: #facc15; }
.stat-orange { color: #fb923c; }
.stat-red { color: #f87171; }
.stat-gray { color: #6b7280; }

/* ── Grade distribution bar ── */
.grade-dist {
    background: #121212; border: 1px solid #242424; border-radius: 10px;
    padding: 14px 20px; margin-bottom: 28px;
}
.grade-dist-title { font-size: 0.7rem; color: #4b5563; text-transform: uppercase;
                    letter-spacing: .6px; margin-bottom: 10px; }
.grade-bar-row { display: flex; height: 28px; border-radius: 6px; overflow: hidden; gap: 2px; }
.grade-seg { display: flex; align-items: center; justify-content: center;
             font-size: 0.7rem; font-weight: 700; color: #000;
             transition: flex .3s; }
.grade-seg.g-a  { background: #4ade80; }
.grade-seg.g-b  { background: #86efac; }
.grade-seg.g-c  { background: #facc15; }
.grade-seg.g-d  { background: #fb923c; }
.grade-seg.g-f  { background: #f87171; }
.grade-labels { display: flex; gap: 14px; margin-top: 10px; flex-wrap: wrap; }
.grade-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; margin-right: 5px; }

/* ── Section header ── */
.section-hdr {
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: .7px;
    color: #4b5563; margin: 24px 0 10px; padding-bottom: 6px;
    border-bottom: 1px solid #1a1a1a;
}

/* ── Team prediction table ── */
.pred-table { width: 100%; border-collapse: collapse; }
.pred-table th {
    font-size: 0.62rem; text-transform: uppercase; letter-spacing: .5px;
    color: #374151; padding: 6px 10px; text-align: left;
    border-bottom: 1px solid #1a1a1a;
}
.pred-table tr:hover td { background: #121212; }
.pred-table td {
    padding: 8px 10px; border-bottom: 1px solid #111;
    font-size: 0.8rem; vertical-align: middle;
}
.td-team { color: #c7d2fe; font-weight: 600; white-space: nowrap; }
.td-rec { color: #6b7280; font-size: 0.72rem; }
.td-wp { color: #d1d5db; font-weight: 600; font-variant-numeric: tabular-nums; }
.td-pred { color: #94a3b8; font-variant-numeric: tabular-nums; }
.td-err { font-variant-numeric: tabular-nums; font-weight: 600; }
.err-pos { color: #4ade80; }
.err-neg { color: #f87171; }
.err-neu { color: #6b7280; }

/* Grade badges */
.grade-a { color: #4ade80; font-weight: 700; }
.grade-b { color: #86efac; font-weight: 700; }
.grade-c { color: #facc15; font-weight: 700; }
.grade-d { color: #fb923c; font-weight: 700; }
.grade-f { color: #f87171; font-weight: 700; }
.grade-na { color: #374151; }

/* Scenario labels */
.scen-opt { color: #4ade80; font-size: 0.65rem; }
.scen-pes { color: #f87171; font-size: 0.65rem; }
.scen-mid { color: #94a3b8; font-size: 0.65rem; }

/* ── Range bar ── */
.range-wrap { position: relative; height: 20px; min-width: 160px; }
.range-track {
    position: absolute; top: 8px; left: 0; right: 0;
    height: 4px; background: #1e1e1e; border-radius: 2px;
}
.range-fill {
    position: absolute; top: 0; height: 4px;
    background: #312e81; border-radius: 2px;
}
.range-dot {
    position: absolute; top: -4px; width: 12px; height: 12px;
    border-radius: 50%; transform: translateX(-50%);
}
.dot-top1   { background: #facc15; z-index: 2; }
.dot-actual { background: #ffffff; border: 2px solid #fff; z-index: 3; }
.dot-pes    { background: #374151; z-index: 1; }
.dot-opt    { background: #374151; z-index: 1; }
.range-lbl-lo { position: absolute; left: 0; top: 14px;
                font-size: 0.55rem; color: #374151; }
.range-lbl-hi { position: absolute; right: 0; top: 14px;
                font-size: 0.55rem; color: #374151; text-align: right; }

/* ── Detailed team cards (expandable) ── */
.detail-section { margin-top: 32px; }
.team-detail {
    background: #121212; border: 1px solid #1e1e1e; border-radius: 10px;
    margin-bottom: 16px; overflow: hidden;
}
.team-detail-hdr {
    padding: 12px 18px; display: flex; justify-content: space-between;
    align-items: center; cursor: pointer;
    border-bottom: 1px solid #1a1a1a;
}
.detail-team-name { font-size: 1.0rem; font-weight: 700; color: #e0e7ff; }
.detail-grade-badge {
    font-size: 0.85rem; font-weight: 700;
    padding: 2px 10px; border-radius: 6px; background: #1a1a2e;
}
.detail-body { padding: 14px 18px; }
.detail-row {
    display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 12px;
    font-size: 0.78rem;
}
.detail-lbl { color: #4b5563; font-size: 0.65rem; text-transform: uppercase;
              letter-spacing: .5px; margin-bottom: 3px; }
.detail-val { color: #d1d5db; font-weight: 600; }
.detail-sub { color: #6b7280; font-size: 0.72rem; }

/* ── Best match card (inside detail) ── */
.match-mini {
    background: #0e0e0e; border: 1px solid #1a1a1a; border-radius: 8px;
    padding: 10px 14px; margin-top: 10px; font-size: 0.75rem;
}
.match-mini-top { color: #818cf8; font-weight: 600; margin-bottom: 6px; }
.match-mini-row { display: flex; gap: 14px; flex-wrap: wrap; }
.mrow-lbl { color: #4b5563; font-size: 0.62rem; }
.mrow-val { color: #9ca3af; font-weight: 600; }

/* ── Arrows ── */
.arr { font-weight:700; }
.arr.up { color:#4ade80; } .arr.dn { color:#f87171; } .arr.fl { color:#6b7280; }
"""


# ---------------------------------------------------------------------------
# Range bar renderer
# ---------------------------------------------------------------------------

def render_range_bar(actual_wp, top1_wp, pes_wp, opt_wp, global_lo=0.1, global_hi=0.9):
    """Horizontal range bar: pessimistic → [top1 dot] → optimistic, with actual marker."""

    def _pct(v, lo=global_lo, hi=global_hi):
        span = hi - lo
        if span <= 0:
            return 50.0
        return max(0.0, min(100.0, (v - lo) / span * 100))

    top1_pos   = _pct(_safe_float(top1_wp, 0.5))
    actual_pos = _pct(_safe_float(actual_wp, 0.5))
    pes_pos    = _pct(_safe_float(pes_wp,  global_lo))
    opt_pos    = _pct(_safe_float(opt_wp,  global_hi))

    fill_left  = min(pes_pos, opt_pos)
    fill_width = abs(opt_pos - pes_pos)

    actual_color = '#4ade80' if actual_wp >= _safe_float(top1_wp, 0) else '#f87171'
    if abs(_safe_float(actual_wp, 0) - _safe_float(top1_wp, 0)) < 0.05:
        actual_color = '#ffffff'

    return f'''<div class="range-wrap">
  <div class="range-track">
    <div class="range-fill" style="left:{fill_left:.1f}%;width:{fill_width:.1f}%;"></div>
    <div class="range-dot dot-pes" style="left:{pes_pos:.1f}%;"></div>
    <div class="range-dot dot-opt" style="left:{opt_pos:.1f}%;"></div>
    <div class="range-dot dot-top1" style="left:{top1_pos:.1f}%;"></div>
    <div class="range-dot dot-actual" style="left:{actual_pos:.1f}%;background:{actual_color};border-color:{actual_color};"></div>
  </div>
  <div class="range-lbl-lo">{fmt_pct(pes_wp)}</div>
  <div class="range-lbl-hi">{fmt_pct(opt_wp)}</div>
</div>'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('Loading data...')

    try:
        sim_df   = pd.read_csv(SIM_CSV)
        profiles = pd.read_csv('team_profiles.csv')
    except FileNotFoundError as e:
        print(f'Missing file: {e}')
        return

    # Playoff lookup
    playoff_lkp = {}
    try:
        po_df = pd.read_csv('nba_playoff_history.csv')
        for _, row in po_df.iterrows():
            playoff_lkp[(row['Team'], row['SEASON'])] = (
                int(row['playoff_round']), str(row['round_label'])
            )
    except FileNotFoundError:
        pass

    # Actual 2025-26 data (team_profiles)
    actual_26 = profiles[profiles['SEASON'] == '2025-26'].copy()

    # Build lookup: sim team name → actual profile row
    # Handle LA Clippers → Los Angeles Clippers
    actual_26_lkp = {}
    for _, row in actual_26.iterrows():
        actual_26_lkp[row['Team']] = row

    # Also add reverse aliases
    for sim_name, prof_name in TEAM_ALIAS.items():
        if prof_name in actual_26_lkp:
            actual_26_lkp[sim_name] = actual_26_lkp[prof_name]

    # Current season GP (for "X games played" note)
    gp_vals = actual_26['GP'].dropna()
    avg_gp = int(gp_vals.mean()) if len(gp_vals) > 0 else 0

    # -----------------------------------------------------------------------
    # Build per-team prediction vs actual comparison
    # -----------------------------------------------------------------------
    current_teams = sorted(sim_df['team'].unique())
    rows = []

    for team in current_teams:
        tm = sim_df[sim_df['team'] == team]
        def _mt(match_type):
            r = tm[tm['match_type'] == match_type]
            return r.iloc[0] if not r.empty else None

        r_top1 = _mt('top_1')
        r_top2 = _mt('top_2')
        r_top3 = _mt('top_3')
        r_opt  = _mt('optimistic')
        r_pes  = _mt('pessimistic')

        if r_top1 is None:
            continue

        top1_wp  = _safe_float(r_top1.get('match_next_1yr_win_pct'))
        top2_wp  = _safe_float(r_top2.get('match_next_1yr_win_pct') if r_top2 is not None else np.nan)
        top3_wp  = _safe_float(r_top3.get('match_next_1yr_win_pct') if r_top3 is not None else np.nan)
        opt_wp   = _safe_float(r_opt.get('match_next_1yr_win_pct')  if r_opt  is not None else np.nan)
        pes_wp   = _safe_float(r_pes.get('match_next_1yr_win_pct')  if r_pes  is not None else np.nan)

        top1_pr  = _safe_float(r_top1.get('match_next_1yr_playoff_round'))
        opt_pr   = _safe_float(r_opt.get('match_next_1yr_playoff_round')  if r_opt  is not None else np.nan)
        pes_pr   = _safe_float(r_pes.get('match_next_1yr_playoff_round')  if r_pes  is not None else np.nan)

        # Ensemble prediction (50/30/20 weighted)
        top_wps  = [w for w in [top1_wp, top2_wp, top3_wp] if not np.isnan(w)]
        wts      = [0.5, 0.3, 0.2][:len(top_wps)]
        ens_wp   = sum(w*v for w, v in zip(wts, top_wps)) / sum(wts) if top_wps else top1_wp

        # 2024-25 data from sim
        curr_wp  = _safe_float(r_top1.get('current_win_pct'))
        curr_W   = int(r_top1.get('current_W', 0))
        curr_L   = int(r_top1.get('current_L', 0))
        win_pct_trend = _safe_float(r_top1.get('current_win_pct_trend', 0))

        # Actual 2025-26
        act_row  = actual_26_lkp.get(team)
        actual_wp = _safe_float(act_row.get('win_pct')) if act_row is not None else np.nan
        actual_W  = int(_safe_float(act_row.get('W', 0), 0)) if act_row is not None else 0
        actual_L  = int(_safe_float(act_row.get('L', 0), 0)) if act_row is not None else 0

        # Grade
        letter, grade_lbl, grade_cls = grade_prediction(actual_wp, top1_wp, pes_wp, opt_wp)
        scen_lbl, scen_cls = scenario_label(actual_wp, top1_wp, pes_wp, opt_wp, top1_pr)

        # Error vs top-1 prediction
        err = actual_wp - top1_wp if not np.isnan(actual_wp) and not np.isnan(top1_wp) else np.nan

        # 2025-26 win% vs their 2024-25 win% — direction check
        direction_correct = None
        if not np.isnan(actual_wp) and not np.isnan(curr_wp) and not np.isnan(top1_wp):
            predicted_dir = top1_wp - curr_wp   # what best match's year 4 was vs their year 3
            actual_dir    = actual_wp - curr_wp
            direction_correct = (predicted_dir >= 0) == (actual_dir >= 0)

        rows.append({
            'team':            team,
            'curr_wp_2425':    curr_wp,
            'curr_W_2425':     curr_W,
            'curr_L_2425':     curr_L,
            'win_pct_trend':   win_pct_trend,
            'top1_predicted':  top1_wp,
            'ens_predicted':   ens_wp,
            'opt_predicted':   opt_wp,
            'pes_predicted':   pes_wp,
            'top1_pr':         top1_pr,
            'opt_pr':          opt_pr,
            'pes_pr':          pes_pr,
            'actual_wp':       actual_wp,
            'actual_W':        actual_W,
            'actual_L':        actual_L,
            'err':             err,
            'abs_err':         abs(err) if not np.isnan(err) else np.nan,
            'letter':          letter,
            'grade_lbl':       grade_lbl,
            'grade_cls':       grade_cls,
            'scen_lbl':        scen_lbl,
            'scen_cls':        scen_cls,
            'direction_correct': direction_correct,
            # Match details
            'top1_team':       r_top1.get('match_team', ''),
            'top1_seasons':    r_top1.get('match_seasons', ''),
            'top1_sim':        _safe_float(r_top1.get('similarity_score')),
            'opt_team':        r_opt.get('match_team', '') if r_opt is not None else '',
            'opt_seasons':     r_opt.get('match_seasons', '') if r_opt is not None else '',
            'pes_team':        r_pes.get('match_team', '') if r_pes is not None else '',
            'pes_seasons':     r_pes.get('match_seasons', '') if r_pes is not None else '',
        })

    df = pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Summary stats
    # -----------------------------------------------------------------------
    valid = df.dropna(subset=['actual_wp', 'top1_predicted'])
    n_valid  = len(valid)
    avg_err  = valid['abs_err'].mean()
    grade_counts = df['letter'].value_counts().to_dict()

    lo = min(df['pes_predicted'].min(), df['top1_predicted'].min(), df['actual_wp'].min())
    hi = max(df['opt_predicted'].max(), df['top1_predicted'].max(), df['actual_wp'].max())
    global_lo = max(0.0, lo - 0.05)
    global_hi = min(1.0, hi + 0.05)

    in_range_n = 0
    for _, r in valid.iterrows():
        lo_r = min(_safe_float(r['pes_predicted'], r['top1_predicted']), r['top1_predicted'])
        hi_r = max(_safe_float(r['opt_predicted'], r['top1_predicted']), r['top1_predicted'])
        if lo_r <= r['actual_wp'] <= hi_r:
            in_range_n += 1

    dir_correct_n = int(valid['direction_correct'].sum())

    # -----------------------------------------------------------------------
    # Grade distribution bar
    # -----------------------------------------------------------------------
    grade_order = [('A', 'g-a'), ('B', 'g-b'), ('C', 'g-c'), ('D', 'g-d'), ('F', 'g-f')]
    grade_segs = ''
    for g, cls in grade_order:
        cnt = grade_counts.get(g, 0)
        if cnt == 0:
            continue
        pct = cnt / n_valid * 100 if n_valid else 0
        grade_segs += (f'<div class="grade-seg {cls}" style="flex:{pct:.0f}">'
                       f'{g} ({cnt})</div>')

    grade_labels_html = ''
    color_map = {'A':'#4ade80','B':'#86efac','C':'#facc15','D':'#fb923c','F':'#f87171'}
    desc_map  = {'A':'On Target','B':'In Range / Near Best','C':'Reasonable',
                 'D':'Off','F':'Way Off'}
    for g, cls in grade_order:
        cnt = grade_counts.get(g, 0)
        if cnt == 0:
            continue
        c = color_map[g]
        grade_labels_html += (f'<span style="font-size:0.7rem;color:#6b7280;">'
                              f'<span class="grade-dot" style="background:{c};"></span>'
                              f'<b style="color:{c};">{g}</b> {desc_map[g]} ({cnt})'
                              f'</span>')

    # -----------------------------------------------------------------------
    # Table — all teams sorted alphabetically, with range bar
    # -----------------------------------------------------------------------
    table_rows = ''
    for _, r in df.sort_values('team').iterrows():
        team      = r['team']
        actual_wp = r['actual_wp']
        top1_wp   = r['top1_predicted']
        err       = r['err']

        # Win% change vs 2024-25
        if not np.isnan(actual_wp) and not np.isnan(r['curr_wp_2425']):
            delta = actual_wp - r['curr_wp_2425']
            d_str = (f'<span class="arr up">▲</span> {abs(delta)*100:.1f}pp'
                     if delta > 0.01 else
                     (f'<span class="arr dn">▼</span> {abs(delta)*100:.1f}pp'
                      if delta < -0.01 else
                      f'<span class="arr fl">→</span> flat'))
        else:
            d_str = '—'

        # Error display
        if not np.isnan(err):
            sign   = '+' if err > 0 else ''
            e_cls  = 'err-pos' if err > 0.02 else ('err-neg' if err < -0.02 else 'err-neu')
            err_str = f'<span class="{e_cls}">{sign}{err*100:.1f}pp</span>'
        else:
            err_str = '—'

        bar = render_range_bar(actual_wp, top1_wp, r['pes_predicted'], r['opt_predicted'],
                               global_lo, global_hi)

        # Scenario label
        scen_html = f'<div class="{r["scen_cls"]}">{r["scen_lbl"]}</div>' if r['scen_lbl'] else ''

        grade_html = f'<span class="{r["grade_cls"]}">{r["letter"]}</span>'
        record_26  = fmt_record(r['actual_W'], r['actual_L'])
        record_25  = fmt_record(r['curr_W_2425'], r['curr_L_2425'])

        table_rows += f'''<tr>
  <td class="td-team">{team}<br><span class="td-rec">2024-25: {record_25}</span></td>
  <td class="td-wp">{fmt_pct(actual_wp)}<br><span class="td-rec">{record_26}</span></td>
  <td>{d_str}</td>
  <td class="td-pred">{fmt_pct(top1_wp)}</td>
  <td>{bar}</td>
  <td>{err_str}</td>
  <td>{grade_html}<br><span style="font-size:0.65rem;color:#4b5563;">{r["grade_lbl"]}</span></td>
  <td>{scen_html}</td>
</tr>'''

    # -----------------------------------------------------------------------
    # Detailed team cards  (sorted by abs error descending — most interesting first)
    # -----------------------------------------------------------------------
    detail_cards = ''
    for _, r in df.sort_values('abs_err', ascending=False, na_position='last').iterrows():
        team     = r['team']
        g_cls    = r['grade_cls']
        letter   = r['letter']
        actual_wp= r['actual_wp']
        top1_wp  = r['top1_predicted']
        err      = r['err']

        if not np.isnan(actual_wp):
            delta_from_25 = actual_wp - r['curr_wp_2425']
            chg_str = (f'+{delta_from_25*100:.1f}pp vs 2024-25'
                       if delta_from_25 > 0.01 else
                       (f'{delta_from_25*100:.1f}pp vs 2024-25'
                        if delta_from_25 < -0.01 else '≈ flat vs 2024-25'))
            chg_color = '#4ade80' if delta_from_25 > 0.01 else ('#f87171' if delta_from_25 < -0.01 else '#6b7280')
        else:
            chg_str, chg_color = 'N/A', '#4b5563'

        err_str_d = (f'{err*100:+.1f}pp' if not np.isnan(err) else '—')
        err_color_d = '#4ade80' if (not np.isnan(err) and err > 0.02) else (
                      '#f87171' if (not np.isnan(err) and err < -0.02) else '#6b7280')

        # Best match detail
        top1_pr_lbl = ROUND_LABELS.get(int(r['top1_pr']), '—') if not np.isnan(r['top1_pr']) else '—'
        opt_pr_lbl  = ROUND_LABELS.get(int(r['opt_pr']), '—')  if not np.isnan(r['opt_pr'])  else '—'
        pes_pr_lbl  = ROUND_LABELS.get(int(r['pes_pr']), '—')  if not np.isnan(r['pes_pr'])  else '—'

        # 2024-25 playoff result (actual)
        po_2425 = playoff_lkp.get((team, '2024-25'))
        po_str  = po_2425[1] if po_2425 else 'Missed playoffs'

        # Was direction correct?
        dir_html = ''
        if r['direction_correct'] is True:
            dir_html = '<span style="color:#4ade80;font-size:0.7rem;">✓ Direction correct</span>'
        elif r['direction_correct'] is False:
            dir_html = '<span style="color:#f87171;font-size:0.7rem;">✗ Direction wrong</span>'

        # Show opt vs pes only if different from top1
        top1_match_str = f'{r["top1_team"]} ({r["top1_seasons"]})'
        opt_match_str  = f'{r["opt_team"]} ({r["opt_seasons"]})' if r['opt_team'] != r['top1_team'] else 'Same as best match'
        pes_match_str  = f'{r["pes_team"]} ({r["pes_seasons"]})' if r['pes_team'] != r['top1_team'] else 'Same as best match'

        detail_cards += f'''<div class="team-detail">
  <div class="team-detail-hdr">
    <div>
      <span class="detail-team-name">{team}</span>
      <span style="color:#4b5563;font-size:0.75rem;margin-left:10px;">
        2025-26: <b style="color:#d1d5db;">{fmt_pct(actual_wp)}</b>
        ({fmt_record(r["actual_W"], r["actual_L"])})
        &nbsp;·&nbsp; <span style="color:{chg_color};">{chg_str}</span>
      </span>
    </div>
    <div>
      <span class="detail-grade-badge {g_cls}" style="font-size:1.1rem;">{letter}</span>
      <span style="font-size:0.7rem;color:#4b5563;margin-left:8px;">{r["grade_lbl"]}</span>
      &nbsp; {dir_html}
    </div>
  </div>
  <div class="detail-body">
    <div class="detail-row">
      <div>
        <div class="detail-lbl">2024-25 Win%</div>
        <div class="detail-val">{fmt_pct(r["curr_wp_2425"])}</div>
        <div class="detail-sub">{fmt_record(r["curr_W_2425"], r["curr_L_2425"])} · {po_str}</div>
      </div>
      <div>
        <div class="detail-lbl">Predicted 2025-26</div>
        <div class="detail-val">{fmt_pct(top1_wp)}</div>
        <div class="detail-sub">Best match: {top1_pr_lbl}</div>
      </div>
      <div>
        <div class="detail-lbl">Actual 2025-26</div>
        <div class="detail-val">{fmt_pct(actual_wp)}</div>
        <div class="detail-sub">{fmt_record(r["actual_W"], r["actual_L"])} · ~{avg_gp} GP</div>
      </div>
      <div>
        <div class="detail-lbl">Error vs Prediction</div>
        <div class="detail-val" style="color:{err_color_d};">{err_str_d}</div>
        <div class="detail-sub">{'Overperforming' if not np.isnan(err) and err > 0.02 else ('Underperforming' if not np.isnan(err) and err < -0.02 else 'On track')}</div>
      </div>
      <div>
        <div class="detail-lbl">Predicted Range</div>
        <div class="detail-val">{fmt_pct(r["pes_predicted"])} – {fmt_pct(r["opt_predicted"])}</div>
        <div class="detail-sub">Pes path: {pes_pr_lbl} &nbsp;·&nbsp; Opt path: {opt_pr_lbl}</div>
      </div>
    </div>
    <div class="match-mini">
      <div class="match-mini-top">&#9670; Best historical match: {r["top1_team"]} ({r["top1_seasons"]}) &nbsp; Similarity {r["top1_sim"]:.0f}/100</div>
      <div class="match-mini-row">
        <div><div class="mrow-lbl">Their Year 4 win%</div><div class="mrow-val">{fmt_pct(top1_wp)}</div></div>
        <div><div class="mrow-lbl">Their playoff result</div><div class="mrow-val">{top1_pr_lbl}</div></div>
        <div><div class="mrow-lbl">Optimistic path</div><div class="mrow-val">{opt_match_str} → {fmt_pct(r["opt_predicted"])}</div></div>
        <div><div class="mrow-lbl">Pessimistic path</div><div class="mrow-val">{pes_match_str} → {fmt_pct(r["pes_predicted"])}</div></div>
      </div>
    </div>
  </div>
</div>'''

    # -----------------------------------------------------------------------
    # Assemble doc
    # -----------------------------------------------------------------------
    body = f'''
<h1>NBA Prediction Accuracy Report — 2025-26</h1>
<p class="subtitle">
  How well did the 2024-25 3-year similarity model predict this season's team performance?
</p>
<p class="note">
  Based on ~{avg_gp} games played per team (season is approximately {avg_gp*100//82}% complete) —
  final accuracy may shift as the season finishes. &nbsp;·&nbsp;
  Yellow dot = best-match predicted win% &nbsp;·&nbsp; White dot = actual 2025-26 win% &nbsp;·&nbsp;
  Dark bar = pessimistic–optimistic predicted range
</p>

<div class="summary-bar">
  <div class="stat-box">
    <div class="stat-val stat-green">{in_range_n}/{n_valid}</div>
    <div class="stat-lbl">In Predicted Range</div>
  </div>
  <div class="stat-box">
    <div class="stat-val stat-yellow">{grade_counts.get("A",0)+grade_counts.get("B",0)}/{n_valid}</div>
    <div class="stat-lbl">Grade A or B</div>
  </div>
  <div class="stat-box">
    <div class="stat-val stat-orange">{avg_err*100:.1f}pp</div>
    <div class="stat-lbl">Avg Error (Best Match)</div>
  </div>
  <div class="stat-box">
    <div class="stat-val stat-gray">{dir_correct_n}/{n_valid}</div>
    <div class="stat-lbl">Direction Correct</div>
  </div>
  <div class="stat-box">
    <div class="stat-val stat-gray">~{avg_gp}</div>
    <div class="stat-lbl">Avg Games Played</div>
  </div>
</div>

<div class="grade-dist">
  <div class="grade-dist-title">Grade distribution — {n_valid} teams</div>
  <div class="grade-bar-row">{grade_segs}</div>
  <div class="grade-labels">{grade_labels_html}</div>
</div>

<div class="section-hdr">Prediction vs Actual — All 30 Teams</div>
<table class="pred-table">
  <thead>
    <tr>
      <th>Team (2024-25)</th>
      <th>2025-26 Actual</th>
      <th>vs Prior Season</th>
      <th>Predicted (Best Match)</th>
      <th style="min-width:180px;">Range: Pes ─── Best ─── Opt &nbsp;⬤ Actual</th>
      <th>Error</th>
      <th>Grade</th>
      <th>Scenario</th>
    </tr>
  </thead>
  <tbody>
{table_rows}
  </tbody>
</table>

<div class="section-hdr detail-section">Detailed Breakdown — Sorted by Biggest Surprise</div>
{detail_cards}
'''

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>NBA Prediction Accuracy — 2025-26</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>"""

    with open(HTML_OUT, 'w', encoding='utf-8') as f:
        f.write(doc)

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    print(f'\n{"="*65}')
    print(f'PREDICTION ACCURACY — 2024-25 model vs 2025-26 actual')
    print(f'{"="*65}')
    print(f'Teams with data:    {n_valid}/30')
    print(f'In predicted range: {in_range_n}/{n_valid}  ({in_range_n/n_valid*100:.0f}%)')
    print(f'Avg abs error:      {avg_err*100:.1f} percentage points')
    print(f'Direction correct:  {dir_correct_n}/{n_valid}')
    print(f'Grade breakdown:    ', end='')
    for g, _ in grade_order:
        cnt = grade_counts.get(g, 0)
        if cnt:
            print(f'{g}:{cnt}', end='  ')
    print()
    print()

    # Best and worst predictions
    sorted_df = df.dropna(subset=['abs_err']).sort_values('abs_err')
    print('MOST ACCURATE PREDICTIONS (closest to best-match):')
    for _, r in sorted_df.head(5).iterrows():
        print(f'  {r["team"]:<28} predicted {fmt_pct(r["top1_predicted"])} → actual {fmt_pct(r["actual_wp"])} '
              f'  err {r["err"]*100:+.1f}pp  [{r["letter"]}]')
    print()
    print('BIGGEST MISSES (furthest from best-match):')
    for _, r in sorted_df.tail(5).iterrows():
        print(f'  {r["team"]:<28} predicted {fmt_pct(r["top1_predicted"])} → actual {fmt_pct(r["actual_wp"])} '
              f'  err {r["err"]*100:+.1f}pp  [{r["letter"]}]')

    print(f'\nSaved {HTML_OUT}')


if __name__ == '__main__':
    main()
