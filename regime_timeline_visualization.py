"""
NBA Franchise Regime Timeline Visualization
Shows coaches, GMs, win%, and playoff success in one integrated view

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
sns.set_style("white")

# Playoff depth colors
PLAYOFF_COLORS = {
    0: '#E8E8E8',  # Did Not Qualify - Light gray
    1: '#FFA500',  # First Round - Orange
    2: '#FF6B6B',  # Second Round - Red
    3: '#9B59B6',  # Conference Finals - Purple
    4: '#3498DB',  # NBA Finals - Blue
    5: '#FFD700',  # NBA Champions - Gold
}

PLAYOFF_LABELS = {
    0: 'No Playoffs',
    1: 'First Round',
    2: 'Second Round',
    3: 'Conf Finals',
    4: 'Finals',
    5: 'Champions'
}


def create_regime_timeline(team_df, team_name='Toronto Raptors'):
    """
    Create comprehensive regime timeline showing coaches, GMs, win%, and playoff results
    """
    
    # Check if we have the necessary columns
    if 'Head_Coach' not in team_df.columns or 'General_Manager' not in team_df.columns:
        print("⚠ Missing coach or GM data. Cannot create regime timeline.")
        return None
    
    # Fill NaN values
    team_df['Head_Coach'] = team_df['Head_Coach'].fillna('Unknown')
    team_df['General_Manager'] = team_df['General_Manager'].fillna('Unknown')
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 12), 
                                         gridspec_kw={'height_ratios': [1, 1, 2]})
    
    seasons = team_df['SEASON'].tolist()
    years = team_df['YEAR'].tolist()
    
    # Determine column name for win percentage
    win_pct_col = 'WIN_PCT' if 'WIN_PCT' in team_df.columns else 'W_PCT'
    
    # --- PANEL 1: HEAD COACHES TIMELINE ---
    ax1.set_xlim(-0.5, len(seasons) - 0.5)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    
    # Track coach changes
    current_coach = None
    coach_start = 0
    coach_segments = []
    
    for i, (season, coach) in enumerate(zip(seasons, team_df['Head_Coach'])):
        if coach != current_coach:
            if current_coach is not None:
                coach_segments.append({
                    'coach': current_coach,
                    'start': coach_start,
                    'end': i,
                    'seasons': i - coach_start
                })
            current_coach = coach
            coach_start = i
    
    # Add final segment
    if current_coach is not None:
        coach_segments.append({
            'coach': current_coach,
            'start': coach_start,
            'end': len(seasons),
            'seasons': len(seasons) - coach_start
        })
    
    # Draw coach segments
    colors = plt.cm.Set3(np.linspace(0, 1, len(coach_segments)))
    
    for idx, segment in enumerate(coach_segments):
        start = segment['start']
        end = segment['end']
        width = end - start
        
        rect = Rectangle((start, 0.2), width, 0.6, 
                         facecolor=colors[idx], edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
        
        # Add coach name
        mid_point = start + width / 2
        coach_name = segment['coach']
        
        # Handle NaN or missing coach names
        if pd.isna(coach_name):
            coach_name = 'Unknown'
        else:
            coach_name = str(coach_name)
        
        # Shorten name if it's too long or has "/"
        if '/' in coach_name:
            coach_name = coach_name.split('/')[0].strip() + '*'  # Mark transition with *
        
        ax1.text(mid_point, 0.5, coach_name, 
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax1.set_title(f'{team_name} - HEAD COACHES TIMELINE', 
                 fontsize=14, fontweight='bold', pad=10)
    ax1.set_xticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    # --- PANEL 2: GENERAL MANAGERS TIMELINE ---
    ax2.set_xlim(-0.5, len(seasons) - 0.5)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    
    # Track GM changes
    current_gm = None
    gm_start = 0
    gm_segments = []
    
    for i, (season, gm) in enumerate(zip(seasons, team_df['General_Manager'])):
        if gm != current_gm:
            if current_gm is not None:
                gm_segments.append({
                    'gm': current_gm,
                    'start': gm_start,
                    'end': i,
                    'seasons': i - gm_start
                })
            current_gm = gm
            gm_start = i
    
    # Add final segment
    if current_gm is not None:
        gm_segments.append({
            'gm': current_gm,
            'start': gm_start,
            'end': len(seasons),
            'seasons': len(seasons) - gm_start
        })
    
    # Draw GM segments
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(gm_segments)))
    
    for idx, segment in enumerate(gm_segments):
        start = segment['start']
        end = segment['end']
        width = end - start
        
        rect = Rectangle((start, 0.2), width, 0.6, 
                         facecolor=colors[idx], edgecolor='black', linewidth=2)
        ax2.add_patch(rect)
        
        # Add GM name
        mid_point = start + width / 2
        gm_name = segment['gm']
        
        # Handle NaN or missing GM names
        if pd.isna(gm_name):
            gm_name = 'Unknown'
        else:
            gm_name = str(gm_name)
        
        ax2.text(mid_point, 0.5, gm_name, 
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax2.set_title('GENERAL MANAGERS TIMELINE', 
                 fontsize=14, fontweight='bold', pad=10)
    ax2.set_xticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    # --- PANEL 3: WIN% AND PLAYOFF RESULTS ---
    x_positions = np.arange(len(seasons))
    
    # Plot win percentage as line
    win_pcts = team_df[win_pct_col].values
    ax3.plot(x_positions, win_pcts, 'o-', linewidth=3, markersize=8, 
            color='#CE1141', label='Win %', zorder=3)
    
    # Add playoff depth as colored bars
    if 'Playoff_Depth' in team_df.columns:
        for i, (x, depth) in enumerate(zip(x_positions, team_df['Playoff_Depth'])):
            if pd.notna(depth):
                color = PLAYOFF_COLORS.get(int(depth), '#E8E8E8')
                ax3.bar(x, 1.0, width=0.8, bottom=0, alpha=0.3, 
                       color=color, zorder=1, edgecolor='none')
    
    # Add 0.500 reference line
    ax3.axhline(y=0.500, color='gray', linestyle='--', linewidth=1.5, 
               alpha=0.5, label='.500 (Break Even)')
    
    ax3.set_xlim(-0.5, len(seasons) - 0.5)
    ax3.set_ylim(0, 1.05)
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(seasons, rotation=45, ha='right')
    ax3.set_ylabel('Win %', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Season', fontsize=13, fontweight='bold')
    ax3.set_title('WIN PERCENTAGE & PLAYOFF SUCCESS', 
                 fontsize=14, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(loc='upper left', fontsize=11)
    
    # Add playoff depth legend
    legend_elements = [
        mpatches.Patch(facecolor=PLAYOFF_COLORS[5], label='Champions', alpha=0.3),
        mpatches.Patch(facecolor=PLAYOFF_COLORS[4], label='Finals', alpha=0.3),
        mpatches.Patch(facecolor=PLAYOFF_COLORS[3], label='Conf Finals', alpha=0.3),
        mpatches.Patch(facecolor=PLAYOFF_COLORS[2], label='2nd Round', alpha=0.3),
        mpatches.Patch(facecolor=PLAYOFF_COLORS[1], label='1st Round', alpha=0.3),
        mpatches.Patch(facecolor=PLAYOFF_COLORS[0], label='No Playoffs', alpha=0.3),
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=9, 
              title='Playoff Results', ncol=2)
    
    plt.suptitle(f'{team_name} - FRANCHISE REGIME TIMELINE', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    filename = f"{team_name.replace(' ', '_')}_regime_timeline.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {filename}")
    
    return fig


def create_regime_summary_table(team_df, team_name='Toronto Raptors'):
    """
    Create a detailed table showing each regime (coach + GM combo) with stats
    """
    
    if 'Head_Coach' not in team_df.columns or 'General_Manager' not in team_df.columns:
        print("⚠ Missing coach or GM data. Cannot create regime summary.")
        return None
    
    # Fill NaN values
    team_df['Head_Coach'] = team_df['Head_Coach'].fillna('Unknown')
    team_df['General_Manager'] = team_df['General_Manager'].fillna('Unknown')
    
    # Determine column name for win percentage
    win_pct_col = 'WIN_PCT' if 'WIN_PCT' in team_df.columns else 'W_PCT'
    
    # Create regime identifier (Coach + GM combination)
    team_df['Regime'] = team_df['Head_Coach'] + ' / ' + team_df['General_Manager']
    
    print("\n" + "=" * 140)
    print(f"{team_name.upper()} - REGIME ANALYSIS (COACH + GM COMBINATIONS)")
    print("=" * 140)
    
    print(f"\n{'Coach':<25} {'GM':<20} {'Seasons':<12} {'Seasons':<8} {'Avg Win%':<12} "
          f"{'Best Playoff':<20} {'Years':<15}")
    print("-" * 140)
    
    # Group by regime
    regime_groups = team_df.groupby(['Head_Coach', 'General_Manager'], sort=False)
    
    for (coach, gm), group in regime_groups:
        seasons = len(group)
        first_season = group['SEASON'].iloc[0]
        last_season = group['SEASON'].iloc[-1]
        avg_win_pct = group[win_pct_col].mean()
        
        # Best playoff result
        if 'Playoff_Depth' in group.columns:
            best_depth = group['Playoff_Depth'].max()
            best_result = PLAYOFF_LABELS.get(int(best_depth), 'Unknown') if pd.notna(best_depth) else 'Unknown'
        else:
            best_result = 'N/A'
        
        years_range = f"{first_season} to {last_season}" if first_season != last_season else first_season
        
        print(f"{coach:<25} {gm:<20} {first_season} - {last_season:<12} {seasons:<8} {avg_win_pct:<12.3f} "
              f"{best_result:<20} {years_range:<15}")
    
    print("=" * 140)
    
    # Summary statistics
    print(f"\nTOTAL COACHING CHANGES: {team_df['Head_Coach'].nunique()}")
    print(f"TOTAL GM CHANGES: {team_df['General_Manager'].nunique()}")
    print(f"TOTAL UNIQUE REGIMES: {team_df['Regime'].nunique()}")
    
    # Most successful regime
    regime_success = team_df.groupby('Regime').agg({
        win_pct_col: 'mean',
        'Playoff_Depth': 'max'
    }).sort_values(win_pct_col, ascending=False)
    
    if len(regime_success) > 0:
        best_regime = regime_success.index[0]
        best_win_pct = regime_success.iloc[0][win_pct_col]
        print(f"\nMOST SUCCESSFUL REGIME (by Win%): {best_regime}")
        print(f"  Average Win%: {best_win_pct:.3f}")


def main():
    """
    Main execution
    """
    
    print("=" * 100)
    print("NBA FRANCHISE REGIME TIMELINE")
    print("=" * 100)
    
    # Load main NBA data
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
    
    # Try to load historical data
    historical_file = f"{team_name.replace(' ', '_').lower()}_historical_data.csv"
    
    try:
        historical_df = pd.read_csv(historical_file)
        print(f"✓ Loaded historical data: {historical_file}")
        
        # Merge
        team_df = team_df.merge(
            historical_df[['SEASON', 'Head_Coach', 'General_Manager', 'Playoff_Result', 'Playoff_Depth']], 
            on='SEASON', 
            how='left'
        )
        print(f"✓ Merged historical data")
        
    except FileNotFoundError:
        print(f"⚠ No historical data found: {historical_file}")
        print("Please ensure the historical data file is in the same directory.")
        return
    
    # Sort by season
    team_df = team_df.sort_values('SEASON').reset_index(drop=True)
    
    print(f"\n{'='*100}")
    print(f"ANALYZING: {team_name}")
    print(f"{'='*100}")
    print(f"Seasons: {len(team_df)}")
    print(f"Range: {team_df['SEASON'].min()} to {team_df['SEASON'].max()}")
    
    # Create visualizations
    create_regime_timeline(team_df, team_name)
    create_regime_summary_table(team_df, team_name)
    
    print("\n" + "=" * 100)
    print("✓ REGIME ANALYSIS COMPLETE!")
    print("=" * 100)


if __name__ == "__main__":
    main()
