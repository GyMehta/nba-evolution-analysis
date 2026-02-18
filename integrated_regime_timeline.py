"""
NBA Franchise Regime Timeline - Enhanced Visualization
Shows coaches, GMs, win%, and playoff success with regime markers

Author: GY Mehta
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")

# Playoff depth colors and heights
PLAYOFF_COLORS = {
    0: '#E8E8E8',  # Did Not Qualify - Light gray
    1: '#FFA500',  # First Round - Orange
    2: '#FF6B6B',  # Second Round - Red
    3: '#9B59B6',  # Conference Finals - Purple
    4: '#3498DB',  # NBA Finals - Blue
    5: '#FFD700',  # NBA Champions - Gold
}

# Playoff bar heights (as fraction of chart)
PLAYOFF_HEIGHTS = {
    0: 0.05,   # Barely visible
    1: 0.20,   # 20% height
    2: 0.35,   # 35% height
    3: 0.50,   # 50% height
    4: 0.70,   # 70% height
    5: 0.95,   # 95% height (champions!)
}

PLAYOFF_LABELS = {
    0: 'No Playoffs',
    1: 'First Round',
    2: 'Second Round',
    3: 'Conf Finals',
    4: 'Finals',
    5: 'Champions'
}


def create_integrated_regime_chart(team_df, team_name='Toronto Raptors'):
    """
    Create single integrated chart with regime changes, win%, and playoff results
    """
    
    # Check data
    if 'Head_Coach' not in team_df.columns or 'General_Manager' not in team_df.columns:
        print("⚠ Missing coach or GM data. Cannot create regime timeline.")
        return None
    
    # Fill NaN values
    team_df['Head_Coach'] = team_df['Head_Coach'].fillna('Unknown')
    team_df['General_Manager'] = team_df['General_Manager'].fillna('Unknown')
    
    # Determine win% column
    win_pct_col = 'WIN_PCT' if 'WIN_PCT' in team_df.columns else 'W_PCT'
    
    fig, ax = plt.subplots(figsize=(24, 10))
    
    seasons = team_df['SEASON'].tolist()
    x_positions = np.arange(len(seasons))
    win_pcts = team_df[win_pct_col].values
    
    # --- PLAYOFF DEPTH BARS (Background) ---
    if 'Playoff_Depth' in team_df.columns:
        for i, depth in enumerate(team_df['Playoff_Depth']):
            if pd.notna(depth):
                depth = int(depth)
                color = PLAYOFF_COLORS.get(depth, '#E8E8E8')
                height = PLAYOFF_HEIGHTS.get(depth, 0.05)
                
                # Draw bar from 0 to height
                ax.bar(i, height, width=0.9, bottom=0, 
                      color=color, alpha=0.6, zorder=1, 
                      edgecolor='white', linewidth=1)
    
    # --- REGIME CHANGE MARKERS ---
    # Track coach changes and add vertical lines
    prev_coach = None
    prev_gm = None
    
    for i, (coach, gm) in enumerate(zip(team_df['Head_Coach'], team_df['General_Manager'])):
        # Check for coach change
        if i > 0 and coach != prev_coach:
            ax.axvline(x=i - 0.5, color='red', linestyle='--', 
                      linewidth=2, alpha=0.7, zorder=2)
            # Add annotation
            ax.text(i - 0.5, 1.02, 'Coach Change', 
                   rotation=90, fontsize=8, color='red', 
                   ha='right', va='bottom', fontweight='bold')
        
        # Check for GM change
        if i > 0 and gm != prev_gm:
            ax.axvline(x=i - 0.5, color='blue', linestyle=':', 
                      linewidth=2, alpha=0.7, zorder=2)
            ax.text(i - 0.5, -0.08, 'GM Change', 
                   rotation=90, fontsize=8, color='blue', 
                   ha='right', va='top', fontweight='bold')
        
        prev_coach = coach
        prev_gm = gm
    
    # --- WIN% LINE ---
    ax.plot(x_positions, win_pcts, 'o-', linewidth=3.5, markersize=10, 
           color='#CE1141', label='Win %', zorder=4, 
           markeredgecolor='white', markeredgewidth=2)
    
    # Add data labels for championship/notable seasons
    for i, (season, win_pct, depth) in enumerate(zip(seasons, win_pcts, team_df['Playoff_Depth'])):
        if pd.notna(depth) and depth >= 4:  # Finals or Champions
            ax.annotate(f'{win_pct:.3f}', 
                       xy=(i, win_pct), 
                       xytext=(0, 10), 
                       textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='yellow', alpha=0.8))
    
    # --- REFERENCE LINES ---
    ax.axhline(y=0.500, color='gray', linestyle='--', linewidth=1.5, 
              alpha=0.5, label='.500 (Break Even)', zorder=3)
    
    # --- REGIME ANNOTATIONS (Top of chart) ---
    # Group consecutive seasons by coach+GM combo
    current_regime = None
    regime_start = 0
    y_pos = 1.08
    
    for i in range(len(team_df)):
        coach = team_df.iloc[i]['Head_Coach']
        gm = team_df.iloc[i]['General_Manager']
        regime = f"{coach} / {gm}"
        
        if regime != current_regime:
            if current_regime is not None:
                # Draw previous regime label
                mid_point = regime_start + (i - regime_start) / 2
                
                # Shorten names if needed
                coach_short = current_regime.split(' / ')[0]
                if '/' in coach_short:
                    coach_short = coach_short.split('/')[0].strip()
                gm_short = current_regime.split(' / ')[1]
                
                label = f"{coach_short}\n{gm_short}"
                
                ax.text(mid_point, y_pos, label, 
                       ha='center', va='bottom', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor='lightblue', alpha=0.7,
                                edgecolor='black', linewidth=1))
            
            current_regime = regime
            regime_start = i
    
    # Add final regime label
    if current_regime is not None:
        mid_point = regime_start + (len(team_df) - regime_start) / 2
        coach_short = current_regime.split(' / ')[0]
        if '/' in coach_short:
            coach_short = coach_short.split('/')[0].strip()
        gm_short = current_regime.split(' / ')[1]
        label = f"{coach_short}\n{gm_short}"
        
        ax.text(mid_point, y_pos, label, 
               ha='center', va='bottom', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='lightblue', alpha=0.7,
                        edgecolor='black', linewidth=1))
    
    # --- FORMATTING ---
    ax.set_xlim(-0.5, len(seasons) - 0.5)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(seasons, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Win %', fontsize=14, fontweight='bold')
    ax.set_xlabel('Season', fontsize=14, fontweight='bold')
    ax.set_title(f'{team_name} - Franchise Timeline: Regimes, Win%, & Playoff Success', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Create custom legend
    legend_elements = [
        plt.Line2D([0], [0], color='#CE1141', linewidth=3.5, 
                  marker='o', markersize=10, label='Win %'),
        plt.Line2D([0], [0], color='gray', linestyle='--', 
                  linewidth=1.5, label='.500 Line'),
        plt.Line2D([0], [0], color='red', linestyle='--', 
                  linewidth=2, label='Coach Change'),
        plt.Line2D([0], [0], color='blue', linestyle=':', 
                  linewidth=2, label='GM Change'),
        mpatches.Patch(facecolor=PLAYOFF_COLORS[5], label='Champions', alpha=0.6),
        mpatches.Patch(facecolor=PLAYOFF_COLORS[4], label='Finals', alpha=0.6),
        mpatches.Patch(facecolor=PLAYOFF_COLORS[3], label='Conf Finals', alpha=0.6),
        mpatches.Patch(facecolor=PLAYOFF_COLORS[2], label='2nd Round', alpha=0.6),
        mpatches.Patch(facecolor=PLAYOFF_COLORS[1], label='1st Round', alpha=0.6),
        mpatches.Patch(facecolor=PLAYOFF_COLORS[0], label='No Playoffs', alpha=0.6),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', 
             fontsize=10, ncol=2, framealpha=0.9)
    
    ax.grid(True, alpha=0.3, axis='y', zorder=0)
    
    plt.tight_layout()
    
    filename = f"{team_name.replace(' ', '_')}_integrated_timeline.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {filename}")
    
    return fig


def main():
    """Main execution"""
    
    print("=" * 100)
    print("NBA FRANCHISE INTEGRATED REGIME TIMELINE")
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
    create_integrated_regime_chart(team_df, team_name)
    
    print("\n" + "=" * 100)
    print("✓ INTEGRATED TIMELINE CREATED!")
    print("=" * 100)


if __name__ == "__main__":
    main()
