"""
Generate Toronto_Raptors_recharts.html
Self-contained React + Recharts interactive page built from existing CSVs.
"""

import json
import math
import pandas as pd

# ── Era definitions (mirrors franchise_evolution_animation.py) ─────────────
FRANCHISE_ERAS = [
    {'name': 'Expansion',       'start': 1995, 'end': 1998, 'color': '#C8C8C8'},
    {'name': 'Vince Carter',    'start': 1998, 'end': 2004, 'color': '#CE1141'},
    {'name': 'Rebuild',         'start': 2004, 'end': 2012, 'color': '#A5ACAF'},
    {'name': 'Lowry/DeRozan',   'start': 2012, 'end': 2018, 'color': '#000000'},
    {'name': 'Championship',    'start': 2018, 'end': 2019, 'color': '#FFD700'},
    {'name': 'Post-Kawhi',      'start': 2019, 'end': 2025, 'color': '#CE1141'},
]

PLAYOFF_HEIGHTS = {0: 0.05, 1: 0.20, 2: 0.35, 3: 0.50, 4: 0.70, 5: 0.95}


def season_year(season_str):
    """'2018-19' -> 2019 (same logic as animation script)"""
    return int(season_str.split('-')[0]) + 1


def build_data():
    print("Loading CSVs...")
    df_all   = pd.read_csv('nba_team_stats_clean.csv')
    hist_df  = pd.read_csv('toronto_raptors_historical_data.csv')

    team_df = df_all[df_all['Team'] == 'Toronto Raptors'].copy()
    team_df = team_df.merge(
        hist_df[['SEASON', 'Head_Coach', 'General_Manager', 'Playoff_Result', 'Playoff_Depth']],
        on='SEASON', how='left'
    )
    team_df = team_df.sort_values('SEASON').reset_index(drop=True)
    print(f"  Raptors seasons found: {len(team_df)}")

    # ── Compute normalised league ranks ───────────────────────────────────
    print("Calculating league ranks...")
    ortg_norm, drtg_norm, nrtg_norm = [], [], []
    for _, row in team_df.iterrows():
        season_all = df_all[df_all['SEASON'] == row['SEASON']]
        n = len(season_all)
        if n == 0 or pd.isna(row.get('ORtg')):
            ortg_norm.append(None); drtg_norm.append(None); nrtg_norm.append(None)
            continue
        ortg_norm.append(round(1 - ((season_all['ORtg'] > row['ORtg']).sum()) / n, 4))
        drtg_norm.append(round(1 - ((season_all['DRtg'] < row['DRtg']).sum()) / n, 4))
        nrtg_v = row.get('NRtg')
        if pd.notna(nrtg_v):
            nrtg_norm.append(round(1 - ((season_all['NRtg'] > nrtg_v).sum()) / n, 4))
        else:
            nrtg_norm.append(None)

    team_df['ORtg_Rank_Norm'] = ortg_norm
    team_df['DRtg_Rank_Norm'] = drtg_norm
    team_df['NRtg_Rank_Norm'] = nrtg_norm

    # ── Map era ───────────────────────────────────────────────────────────
    def get_era(season_str):
        sy = season_year(season_str)
        for era in FRANCHISE_ERAS:
            if era['start'] <= sy <= era['end']:
                return era['name'], era['color']
        return 'Unknown', '#999999'

    team_df['era_name']  = team_df['SEASON'].apply(lambda s: get_era(s)[0])
    team_df['era_color'] = team_df['SEASON'].apply(lambda s: get_era(s)[1])

    # ── Playoff bar height ────────────────────────────────────────────────
    team_df['playoffHeight'] = team_df['Playoff_Depth'].apply(
        lambda d: PLAYOFF_HEIGHTS.get(int(d) if pd.notna(d) else 0, 0.05)
    )

    # ── Serialise to list of dicts ────────────────────────────────────────
    keep = ['SEASON', 'YEAR', 'W', 'L', 'WIN_PCT', 'ORtg', 'DRtg', 'NRtg',
            'Pace', '3PA', 'TS%', 'eFG%', 'PIE',
            'ORtg_Rank_Norm', 'DRtg_Rank_Norm', 'NRtg_Rank_Norm',
            'Head_Coach', 'General_Manager', 'Playoff_Result', 'Playoff_Depth',
            'era_name', 'era_color', 'playoffHeight']

    records = []
    for _, row in team_df.iterrows():
        rec = {}
        for col in keep:
            if col not in row:
                continue
            v = row[col]
            if isinstance(v, float) and math.isnan(v):
                rec[col] = None
            elif hasattr(v, 'item'):          # numpy scalar
                rec[col] = v.item()
            else:
                rec[col] = v
        records.append(rec)

    return records


def build_eras(data):
    """Build ERAS array: [{name, color, x1, x2}, ...] for ReferenceArea."""
    eras = []
    for era_def in FRANCHISE_ERAS:
        era_seasons = [d['SEASON'] for d in data
                       if era_def['start'] <= season_year(d['SEASON']) <= era_def['end']]
        if era_seasons:
            eras.append({
                'name':  era_def['name'],
                'color': era_def['color'],
                'x1':    era_seasons[0],
                'x2':    era_seasons[-1],
            })
    return eras


# ── HTML template ──────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Toronto Raptors – Franchise Evolution</title>
  <script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
  <script src="https://unpkg.com/recharts@2/umd/Recharts.js" crossorigin></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; background: #f5f5f5; font-family: 'Courier New', monospace; }
    h1 { text-align: center; font-size: 22px; margin: 16px 0 4px; }
    #root { max-width: 1400px; margin: 0 auto; padding: 16px; }
    button { cursor: pointer; font-family: 'Courier New', monospace; }
    input[type=range] { cursor: pointer; }
    .controls {
      display: flex; align-items: center; gap: 10px;
      background: #e8e8e8; border-radius: 8px; padding: 10px 16px; margin: 8px 0;
      flex-wrap: wrap;
    }
    .btn {
      padding: 6px 14px; border-radius: 6px; border: 1px solid #bbb;
      background: white; font-size: 14px;
    }
    .btn-play { background: #CE1141; color: white; border-color: #CE1141; }
    .season-label { font-weight: bold; min-width: 72px; font-size: 15px; }
    .season-count { color: #777; font-size: 12px; }
    .info-grid {
      display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 10px;
    }
    .card { padding: 16px; border-radius: 8px; font-size: 13px; line-height: 1.7; }
    .card-blue   { background: #ADD8E6; }
    .card-yellow { background: #FFFACD; }
    .card-title  { font-weight: bold; font-size: 14px; margin-bottom: 6px; }
    .legend-bar {
      display: flex; flex-wrap: wrap; gap: 12px;
      padding: 10px 14px; background: #efefef; border-radius: 8px;
      margin-top: 10px; font-size: 12px; align-items: center;
    }
    .legend-item { display: flex; align-items: center; gap: 5px; }
    .legend-box  { width: 15px; height: 15px; border-radius: 3px; opacity: 0.8; }
    .legend-line { width: 28px; height: 3px; display: inline-block; }
  </style>
</head>
<body>
<div id="root"></div>

<script>
  const DATA = __DATA__;
  const ERAS = __ERAS__;
</script>

<script type="text/babel">
const { useState, useEffect, useRef, useCallback } = React;
const {
  ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceArea, ReferenceLine, Cell
} = Recharts;

const PLAYOFF_COLORS  = { 0:'#E8E8E8', 1:'#FFA500', 2:'#FF6B6B', 3:'#9B59B6', 4:'#3498DB', 5:'#FFD700' };
const PLAYOFF_LABELS  = { 0:'No Playoffs', 1:'First Round', 2:'Second Round', 3:'Conf Finals', 4:'Finals', 5:'Champions' };

function rank(normVal) {
  if (normVal == null) return null;
  return Math.round((1 - normVal) * 30 + 1);
}

function fmt1(v) { return v != null ? v.toFixed(1) : 'N/A'; }
function fmt3(v) { return v != null ? v.toFixed(3) : 'N/A'; }

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  return (
    <div style={{background:'white', padding:'10px 14px', border:'1px solid #ccc',
                 borderRadius:'8px', fontSize:'12px', lineHeight:'1.6',
                 boxShadow:'0 2px 8px rgba(0,0,0,0.15)'}}>
      <div style={{fontWeight:'bold', marginBottom:'4px'}}>{label}</div>
      <div>Record: {d.W}-{d.L} ({fmt3(d.WIN_PCT)})</div>
      <div>Playoffs: {d.Playoff_Result || 'N/A'}</div>
      <div>Coach: {d.Head_Coach || 'N/A'}</div>
      <div style={{marginTop:'4px'}}>
        NRtg: {fmt1(d.NRtg)} &nbsp; ORtg: {fmt1(d.ORtg)} &nbsp; DRtg: {fmt1(d.DRtg)}
      </div>
      <div style={{color:'#888', marginTop:'2px'}}>Era: {d.era_name}</div>
    </div>
  );
};

function App() {
  const [frame, setFrame]   = useState(DATA.length - 1);
  const [playing, setPlaying] = useState(false);
  const timerRef = useRef(null);

  useEffect(() => {
    if (playing) {
      timerRef.current = setInterval(() => {
        setFrame(f => {
          if (f >= DATA.length - 1) { setPlaying(false); return f; }
          return f + 1;
        });
      }, 1000);
    }
    return () => clearInterval(timerRef.current);
  }, [playing]);

  const handlePlay = () => {
    if (frame >= DATA.length - 1) setFrame(0);
    setPlaying(true);
  };
  const handlePrev = () => { setPlaying(false); setFrame(f => Math.max(0, f - 1)); };
  const handleNext = () => { setPlaying(false); setFrame(f => Math.min(DATA.length - 1, f + 1)); };

  const current  = DATA[frame];
  const cumData  = DATA.slice(0, frame + 1);
  const cumWins  = cumData.reduce((s, d) => s + (d.W  || 0), 0);
  const cumLosses= cumData.reduce((s, d) => s + (d.L  || 0), 0);
  const cumPO    = cumData.filter(d => (d.Playoff_Depth || 0) > 0).length;
  const cumChamp = cumData.filter(d => d.Playoff_Depth === 5).length;
  const cumWinPct= cumWins + cumLosses > 0 ? cumWins / (cumWins + cumLosses) : 0;

  const ortgRank = rank(current.ORtg_Rank_Norm);
  const drtgRank = rank(current.DRtg_Rank_Norm);
  const nrtgRank = rank(current.NRtg_Rank_Norm);

  const customDot = useCallback((props) => {
    const { cx, cy, payload } = props;
    if (!cx || !cy) return null;
    if (payload.SEASON === current.SEASON) {
      return <circle key={payload.SEASON} cx={cx} cy={cy} r={12}
               fill="yellow" stroke="#CE1141" strokeWidth={3} />;
    }
    return <circle key={payload.SEASON} cx={cx} cy={cy} r={3.5}
             fill="#CE1141" stroke="white" strokeWidth={1.5} />;
  }, [current.SEASON]);

  return (
    <div>
      <h1>Toronto Raptors – Franchise Evolution</h1>

      <ResponsiveContainer width="100%" height={420}>
        <ComposedChart data={DATA} margin={{top:30, right:90, left:50, bottom:70}}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />

          {ERAS.map(era => (
            <ReferenceArea key={era.name} x1={era.x1} x2={era.x2}
              fill={era.color} fillOpacity={0.15} ifOverflow="extendDomain"
              label={{ value: era.name, position:'insideTop', fill: era.color,
                       fontWeight:'bold', fontSize:10 }} />
          ))}

          <ReferenceLine yAxisId="win" y={0.5}
            stroke="#888" strokeDasharray="5 5" strokeWidth={1.5} />
          <ReferenceLine yAxisId="win" x={current.SEASON}
            stroke="gold" strokeWidth={2} strokeDasharray="4 4" />

          <XAxis dataKey="SEASON" angle={-45} textAnchor="end"
            tick={{fontSize:10}} interval={0} height={65} />
          <YAxis yAxisId="win" domain={[0, 1]}
            tickFormatter={v => (v * 100).toFixed(0) + '%'}
            label={{value:'Win %', angle:-90, position:'insideLeft', offset:10, fontSize:12}} />
          <YAxis yAxisId="rank" orientation="right" domain={[0, 1]}
            tickFormatter={v => '#' + Math.round((1 - v) * 30 + 1)}
            label={{value:'League Rank', angle:90, position:'insideRight', offset:15, fontSize:12}}
            tick={{fontSize:9, fill:'#888'}} />

          <Tooltip content={<CustomTooltip />} />

          <Bar yAxisId="win" dataKey="playoffHeight" opacity={0.6} isAnimationActive={false}>
            {DATA.map((entry, i) => (
              <Cell key={i} fill={PLAYOFF_COLORS[entry.Playoff_Depth] ?? '#E8E8E8'} />
            ))}
          </Bar>

          <Line yAxisId="win"  dataKey="WIN_PCT"        stroke="#CE1141"
            strokeWidth={3}   dot={customDot}           activeDot={false} isAnimationActive={false} />
          <Line yAxisId="rank" dataKey="ORtg_Rank_Norm" stroke="#2ECC71"
            strokeWidth={1.2} strokeOpacity={0.55}      dot={false} isAnimationActive={false} />
          <Line yAxisId="rank" dataKey="DRtg_Rank_Norm" stroke="#E74C3C"
            strokeWidth={1.2} strokeOpacity={0.55}      dot={false} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Controls */}
      <div className="controls">
        <button className="btn" onClick={handlePrev}>◀</button>
        {playing
          ? <button className="btn btn-play" onClick={() => setPlaying(false)}>⏸ Pause</button>
          : <button className="btn btn-play" onClick={handlePlay}>▶ Play</button>}
        <button className="btn" onClick={handleNext}>▶</button>
        <input type="range" min={0} max={DATA.length - 1} value={frame}
          onChange={e => { setPlaying(false); setFrame(Number(e.target.value)); }}
          style={{flex:1, minWidth:'120px'}} />
        <span className="season-label">{current.SEASON}</span>
        <span className="season-count">Season {frame + 1} of {DATA.length}</span>
      </div>

      {/* Info panels */}
      <div className="info-grid">
        <div className="card card-blue">
          <div className="card-title">SEASON: {current.SEASON} &nbsp;|&nbsp; Era: {current.era_name}</div>
          <div>RECORD: {current.W}-{current.L} &nbsp; ({fmt3(current.WIN_PCT)})</div>
          <div>PLAYOFFS: {current.Playoff_Result || 'N/A'}</div>
          <div style={{marginTop:'6px'}}>HEAD COACH: {current.Head_Coach || 'N/A'}</div>
          <div>GENERAL MANAGER: {current.General_Manager || 'N/A'}</div>
          <div style={{marginTop:'8px', fontWeight:'bold'}}>TEAM PERFORMANCE:</div>
          <div>• Net Rating: {fmt1(current.NRtg)}{nrtgRank ? ` (Rank: #${nrtgRank})` : ''}</div>
          <div>• Off Rating: {fmt1(current.ORtg)}{ortgRank ? ` (Rank: #${ortgRank})` : ''}</div>
          <div>• Def Rating: {fmt1(current.DRtg)}{drtgRank ? ` (Rank: #${drtgRank})` : ''}</div>
          <div>• Pace: {fmt1(current.Pace)} &nbsp;|&nbsp; 3PA: {fmt1(current['3PA'])}</div>
        </div>

        <div className="card card-yellow">
          <div className="card-title">FRANCHISE TOTALS (Through {current.SEASON}):</div>
          <div>• All-Time Record: {cumWins}-{cumLosses} &nbsp; ({fmt3(cumWinPct)})</div>
          <div>• Playoff Appearances: {cumPO}</div>
          <div>• Championships: {cumChamp}</div>
        </div>
      </div>

      {/* Legend */}
      <div className="legend-bar">
        {Object.entries(PLAYOFF_LABELS).map(([k, v]) => (
          <span key={k} className="legend-item">
            <span className="legend-box" style={{background: PLAYOFF_COLORS[k]}}></span>
            {v}
          </span>
        ))}
        <span className="legend-item">
          <span className="legend-line" style={{background:'#CE1141', height:'3px'}}></span>
          Win %
        </span>
        <span className="legend-item">
          <span className="legend-line" style={{background:'#2ECC71', height:'2px', opacity:0.6}}></span>
          Off Rank
        </span>
        <span className="legend-item">
          <span className="legend-line" style={{background:'#E74C3C', height:'2px', opacity:0.6}}></span>
          Def Rank
        </span>
        <span className="legend-item">
          <span style={{display:'inline-block', width:'28px', borderTop:'2px dashed #888'}}></span>
          .500 line
        </span>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
</script>
</body>
</html>
"""


def main():
    data = build_data()
    eras = build_eras(data)

    data_json = json.dumps(data, indent=None)
    eras_json = json.dumps(eras, indent=None)

    html = HTML_TEMPLATE \
        .replace('__DATA__', data_json) \
        .replace('__ERAS__', eras_json)

    out_file = 'Toronto_Raptors_recharts.html'
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Written: {out_file}  ({len(html):,} bytes)")
    print(f"  Seasons:  {len(data)}")
    print(f"  Eras:     {len(eras)}")
    print("Open in a browser to view.")


if __name__ == '__main__':
    main()
