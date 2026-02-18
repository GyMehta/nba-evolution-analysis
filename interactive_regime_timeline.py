"""
NBA Franchise Regime Timeline - Interactive Version
Uses Plotly for mouseover tooltips showing coach/GM details

Author: GY Mehta
Date: February 2026
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Playoff depth colors
PLAYOFF_COLORS = {
    0: 'rgba(232, 232, 232, 0.6)',  # No Playoffs
    1: 'rgba(255, 165, 0, 0.6)',    # First Round
    2: 'rgba(255, 107, 107, 0.6)',  # Second Round
    3: 'rgba(155, 89, 182, 0.6)',   # Conference Finals
    4: 'rgba(52, 152, 219, 0.6)',   # Finals
    5: 'rgba(255, 215, 0, 0.6)',    # Champions
}

PLAYOFF_HEIGHTS = {
    0: 0.05,
    1: 0.20,
    2: 0.35,
    3: 0.50,
    4: 0.70,
    5: 0.95,
}

PLAYOFF_LABELS = {
    0: 'No Playoffs',
    1: 'First Round',
    2: 'Second Round',
    3: 'Conference Finals',
    4: 'NBA Finals',
    5: 'NBA Champions'
}


def create_interactive_timeline(team_df, team_name='Toronto Raptors'):
    """
    Create interactive Plotly timeline with mouseover details
    """
    
    # Check data
    if 'Head_Coach' not in team_df.columns or 'General_Manager' not in team_df.columns:
        print("⚠ Missing coach or GM data.")
        return None
    
    # Fill NaN
    team_df['Head_Coach'] = team_df['Head_Coach'].fillna('Unknown')
    team_df['General_Manager'] = team_df['General_Manager'].fillna('Unknown')
    
    # Determine win% column
    win_pct_col = 'WIN_PCT' if 'WIN_PCT' in team_df.columns else 'W_PCT'
    
    seasons = team_df['SEASON'].tolist()
    win_pcts = team_df[win_pct_col].values
    
    # Create figure
    fig = go.Figure()
    
    # --- ADD PLAYOFF DEPTH BARS ---
    if 'Playoff_Depth' in team_df.columns:
        for i, (season, depth) in enumerate(zip(seasons, team_df['Playoff_Depth'])):
            if pd.notna(depth):
                depth = int(depth)
                color = PLAYOFF_COLORS.get(depth, 'rgba(232, 232, 232, 0.6)')
                height = PLAYOFF_HEIGHTS.get(depth, 0.05)
                
                fig.add_trace(go.Bar(
                    x=[season],
                    y=[height],
                    name=PLAYOFF_LABELS.get(depth, 'Unknown'),
                    marker_color=color,
                    showlegend=(i == 0 and depth in PLAYOFF_LABELS),  # Only show one of each in legend
                    hovertext=f"{season}<br>{PLAYOFF_LABELS.get(depth, 'Unknown')}",
                    hoverinfo='text',
                    width=0.9
                ))
    
    # --- ADD WIN% LINE WITH DETAILED TOOLTIPS ---
    hover_text = []
    for i, row in team_df.iterrows():
        coach = row['Head_Coach']
        gm = row['General_Manager']
        season = row['SEASON']
        win_pct = row[win_pct_col]
        
        # Get additional stats if available
        ortg = f"<br>ORtg: {row['ORtg']:.1f}" if 'ORtg' in row and pd.notna(row['ORtg']) else ""
        drtg = f"<br>DRtg: {row['DRtg']:.1f}" if 'DRtg' in row and pd.notna(row['DRtg']) else ""
        playoff = f"<br>Playoff: {row['Playoff_Result']}" if 'Playoff_Result' in row and pd.notna(row['Playoff_Result']) else ""
        
        text = (f"<b>{season}</b><br>"
                f"Win%: {win_pct:.3f}<br>"
                f"Coach: {coach}<br>"
                f"GM: {gm}"
                f"{playoff}"
                f"{ortg}"
                f"{drtg}")
        hover_text.append(text)
    
    fig.add_trace(go.Scatter(
        x=seasons,
        y=win_pcts,
        mode='lines+markers',
        name='Win %',
        line=dict(color='#CE1141', width=3.5),
        marker=dict(size=10, color='#CE1141', 
                   line=dict(color='white', width=2)),
        hovertext=hover_text,
        hoverinfo='text'
    ))
    
    # --- ADD .500 REFERENCE LINE ---
    fig.add_trace(go.Scatter(
        x=seasons,
        y=[0.500] * len(seasons),
        mode='lines',
        name='.500 (Break Even)',
        line=dict(color='gray', width=2, dash='dash'),
        hoverinfo='skip'
    ))
    
    # --- ADD REGIME CHANGE MARKERS ---
    prev_coach = None
    prev_gm = None
    
    for i, (season, coach, gm) in enumerate(zip(seasons, team_df['Head_Coach'], team_df['General_Manager'])):
        if i > 0:
            if coach != prev_coach:
                fig.add_vline(
                    x=i - 0.5,
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    opacity=0.7,
                    annotation_text="Coach Change",
                    annotation_position="top",
                    annotation_font_size=8,
                    annotation_font_color="red"
                )
            
            if gm != prev_gm:
                fig.add_vline(
                    x=i - 0.5,
                    line_dash="dot",
                    line_color="blue",
                    line_width=2,
                    opacity=0.7,
                    annotation_text="GM Change",
                    annotation_position="bottom",
                    annotation_font_size=8,
                    annotation_font_color="blue"
                )
        
        prev_coach = coach
        prev_gm = gm
    
    # --- LAYOUT ---
    fig.update_layout(
        title=dict(
            text=f'{team_name} - Interactive Franchise Timeline<br>'
                 '<sub>Hover over points for coach/GM details</sub>',
            font=dict(size=20, color='#000000')
        ),
        xaxis_title='Season',
        yaxis_title='Win %',
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1600,
        height=800,
        font=dict(size=12),
        xaxis=dict(
            tickangle=45,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            range=[-0.05, 1.15],
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Save as HTML
    filename = f"{team_name.replace(' ', '_')}_interactive_timeline.html"
    fig.write_html(filename)
    print(f"\n✓ Saved interactive version: {filename}")
    print("  Open this file in your web browser for mouseover tooltips!")
    
    return fig


def main():
    """Main execution"""
    
    print("=" * 100)
    print("NBA FRANCHISE INTERACTIVE TIMELINE")
    print("=" * 100)
    
    # Load data
    df = pd.read_csv('nba_team_stats_1980_2025.csv')
    
    # Select team
    print("\nDefault: Toronto Raptors")
    team_name = input("Enter team name (or press Enter for Raptors): ").strip()
    if not team_name:
        team_name = 'Toronto Raptors'
    
    # Get team data
    team_df = df[df['Team'] == team_name].copy()
    
    if len(team_df) == 0:
        print(f"Error: No data found for '{team_name}'")
        return
    
    # Load historical data
    historical_file = f"{team_name.replace(' ', '_').lower()}_historical_data.csv"
    
    try:
        historical_df = pd.read_csv(historical_file)
        print(f"✓ Loaded historical data: {historical_file}")
        
        team_df = team_df.merge(
            historical_df[['SEASON', 'Head_Coach', 'General_Manager', 'Playoff_Result', 'Playoff_Depth']], 
            on='SEASON', 
            how='left'
        )
        print(f"✓ Merged historical data")
        
    except FileNotFoundError:
        print(f"⚠ No historical data found: {historical_file}")
        return
    
    team_df = team_df.sort_values('SEASON').reset_index(drop=True)
    
    print(f"\n{'='*100}")
    print(f"ANALYZING: {team_name}")
    print(f"{'='*100}")
    
    # Create visualization
    create_interactive_timeline(team_df, team_name)
    
    print("\n" + "=" * 100)
    print("✓ INTERACTIVE TIMELINE CREATED!")
    print("Open the HTML file in your browser to see mouseover tooltips")
    print("=" * 100)


if __name__ == "__main__":
    main()
