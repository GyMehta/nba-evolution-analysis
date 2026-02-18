"""
NBA Franchise Evolution Analysis
Deep dive into a team's journey over time with playoff performance

Author: GY Mehta
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

# Playoff results by season (manual lookup - based on historical records)
PLAYOFF_RESULTS = {
    'Toronto Raptors': {
        '1995-96': 'Did Not Qualify',
        '1996-97': 'Did Not Qualify',
        '1997-98': 'Did Not Qualify',
        '1998-99': 'Did Not Qualify',
        '1999-00': 'First Round',
        '2000-01': 'Second Round',
        '2001-02': 'First Round',
        '2002-03': 'Did Not Qualify',
        '2003-04': 'Did Not Qualify',
        '2004-05': 'Did Not Qualify',
        '2005-06': 'Did Not Qualify',
        '2006-07': 'First Round',
        '2007-08': 'First Round',
        '2008-09': 'Did Not Qualify',
        '2009-10': 'Did Not Qualify',
        '2010-11': 'Did Not Qualify',
        '2011-12': 'Did Not Qualify',
        '2012-13': 'Did Not Qualify',
        '2013-14': 'First Round',
        '2014-15': 'First Round',
        '2015-16': 'Conference Finals',
        '2016-17': 'Second Round',
        '2017-18': 'Second Round',
        '2018-19': 'NBA Champions',
        '2019-20': 'Second Round',
        '2020-21': 'Did Not Qualify',
        '2021-22': 'First Round',
        '2022-23': 'Did Not Qualify',
        '2023-24': 'First Round',
    }
}

# Playoff depth scoring
PLAYOFF_DEPTH = {
    'Did Not Qualify': 0,
    'First Round': 1,
    'Second Round': 2,
    'Conference Finals': 3,
    'NBA Finals': 4,
    'NBA Champions': 5
}

# Define franchise eras (can be customized per team)
FRANCHISE_ERAS = {
    'Toronto Raptors': [
        {'name': 'Expansion', 'start': 1995, 'end': 1998, 'color': '#C8C8C8'},
        {'name': 'Vince Carter Era', 'start': 1998, 'end': 2004, 'color': '#CE1141'},
        {'name': 'Rebuild', 'start': 2004, 'end': 2012, 'color': '#A5ACAF'},
        {'name': 'Lowry/DeRozan', 'start': 2012, 'end': 2018, 'color': '#000000'},
        {'name': 'Championship', 'start': 2018, 'end': 2019, 'color': '#FFD700'},
        {'name': 'Post-Kawhi', 'start': 2019, 'end': 2025, 'color': '#CE1141'},
    ]
}


def load_data(filename='nba_team_stats_1980_2025.csv'):
    """Load NBA data"""
    try:
        df = pd.read_csv(filename)
        print(f"✓ Data loaded: {len(df)} rows")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find '{filename}'")
        return None


def get_franchise_data(df, team_name='Toronto Raptors'):
    """
    Extract data for a specific franchise and merge with historical data
    """
    team_df = df[df['Team'] == team_name].copy()
    
    if len(team_df) == 0:
        print(f"Error: No data found for '{team_name}'")
        print(f"Available teams: {sorted(df['Team'].unique())}")
        return None
    
    # Try to load historical data (coaches, GMs, playoff results)
    historical_file = f"{team_name.replace(' ', '_').lower()}_historical_data.csv"
    
    try:
        historical_df = pd.read_csv(historical_file)
        print(f"  ✓ Loaded historical data: {historical_file}")
        
        # Merge with main data
        team_df = team_df.merge(
            historical_df[['SEASON', 'Head_Coach', 'General_Manager', 'Playoff_Result', 'Playoff_Depth']], 
            on='SEASON', 
            how='left'
        )
        print(f"  ✓ Merged historical data (coaches, GMs, playoff results)")
        
    except FileNotFoundError:
        print(f"  ⚠ No historical data file found: {historical_file}")
        print(f"  Continuing with basic stats only...")
        
        # Add default playoff data if available from the existing script
        if team_name in PLAYOFF_RESULTS:
            team_df['Playoff_Result'] = team_df['SEASON'].map(PLAYOFF_RESULTS[team_name])
            team_df['Playoff_Depth'] = team_df['Playoff_Result'].map(PLAYOFF_DEPTH)
    
    # Sort by season
    team_df = team_df.sort_values('SEASON').reset_index(drop=True)
    
    return team_df


def calculate_league_averages(df):
    """
    Calculate league averages by season for comparison
    """
    metrics = ['ORtg', 'DRtg', '3PA', 'Pace', 'eFG%', 'TS%', 'AST%', 'TOV%']
    
    league_avg = df.groupby('SEASON')[metrics].mean().reset_index()
    league_avg = league_avg.add_suffix('_League')
    league_avg = league_avg.rename(columns={'SEASON_League': 'SEASON'})
    
    return league_avg


def create_main_visualization(team_df, league_avg, team_name='Toronto Raptors'):
    """
    Create comprehensive multi-panel visualization
    """
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.25)
    
    # Merge with league averages
    plot_df = team_df.merge(league_avg, on='SEASON', how='left')
    
    # Get eras for this team
    eras = FRANCHISE_ERAS.get(team_name, [])
    
    # --- PANEL 1: Win% & Playoff Success ---
    ax1 = fig.add_subplot(gs[0, :])
    
    # Use WIN_PCT column (not W_PCT)
    win_pct_col = 'WIN_PCT' if 'WIN_PCT' in plot_df.columns else 'W_PCT'
    
    # Win% line
    ax1.plot(plot_df['SEASON'], plot_df[win_pct_col], 
            'o-', linewidth=3, markersize=8, color='#CE1141', label='Win%', zorder=3)
    
    # Add playoff depth as bars in background
    if 'Playoff_Depth' in plot_df.columns:
        ax1_twin = ax1.twinx()
        colors = ['#E8E8E8', '#FFA500', '#FF6B6B', '#9B59B6', '#3498DB', '#FFD700']
        
        for idx, row in plot_df.iterrows():
            depth = row['Playoff_Depth']
            ax1_twin.bar(row['SEASON'], depth, alpha=0.3, 
                        color=colors[int(depth)] if not pd.isna(depth) else '#E8E8E8',
                        zorder=1)
        
        ax1_twin.set_ylabel('Playoff Depth', fontsize=12, fontweight='bold')
        ax1_twin.set_ylim(0, 6)
        ax1_twin.set_yticks(range(6))
        ax1_twin.set_yticklabels(['No Playoffs', 'R1', 'R2', 'Conf Finals', 'Finals', 'Champions'])
    
    # Shade eras
    for era in eras:
        start_year = era['start']
        end_year = era['end']
        ax1.axvspan(f"{start_year-1}-{str(start_year)[-2:]}", 
                   f"{end_year-1}-{str(end_year)[-2:]}", 
                   alpha=0.15, color=era['color'], zorder=0)
    
    ax1.set_xlabel('Season', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Win %', fontsize=13, fontweight='bold')
    ax1.set_title(f'{team_name} - Win% & Playoff Performance Over Time', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.set_ylim(0, 1)
    
    # --- PANEL 2: Offensive & Defensive Rating ---
    ax2 = fig.add_subplot(gs[1, 0])
    
    if 'ORtg' in plot_df.columns:
        ax2.plot(plot_df['SEASON'], plot_df['ORtg'], 
                'o-', linewidth=2.5, markersize=6, color='#CE1141', label='Team ORtg')
        ax2.plot(plot_df['SEASON'], plot_df['ORtg_League'], 
                '--', linewidth=2, color='gray', alpha=0.6, label='League Avg ORtg')
    
    if 'DRtg' in plot_df.columns:
        ax2.plot(plot_df['SEASON'], plot_df['DRtg'], 
                'o-', linewidth=2.5, markersize=6, color='#000000', label='Team DRtg')
        ax2.plot(plot_df['SEASON'], plot_df['DRtg_League'], 
                '--', linewidth=2, color='gray', alpha=0.6, label='League Avg DRtg')
    
    ax2.set_xlabel('Season', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Rating (per 100 poss)', fontsize=12, fontweight='bold')
    ax2.set_title('Offensive & Defensive Efficiency', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # --- PANEL 3: 3-Point Attempts Evolution ---
    ax3 = fig.add_subplot(gs[1, 1])
    
    if '3PA' in plot_df.columns:
        ax3.plot(plot_df['SEASON'], plot_df['3PA'], 
                'o-', linewidth=2.5, markersize=6, color='#CE1141', label='Team 3PA')
        ax3.plot(plot_df['SEASON'], plot_df['3PA_League'], 
                '--', linewidth=2, color='gray', alpha=0.6, label='League Avg 3PA')
        ax3.fill_between(plot_df['SEASON'], plot_df['3PA'], plot_df['3PA_League'],
                        where=(plot_df['3PA'] >= plot_df['3PA_League']), 
                        color='green', alpha=0.1, label='Above League Avg')
        ax3.fill_between(plot_df['SEASON'], plot_df['3PA'], plot_df['3PA_League'],
                        where=(plot_df['3PA'] < plot_df['3PA_League']), 
                        color='red', alpha=0.1, label='Below League Avg')
    
    ax3.set_xlabel('Season', fontsize=12, fontweight='bold')
    ax3.set_ylabel('3-Point Attempts per Game', fontsize=12, fontweight='bold')
    ax3.set_title('3-Point Shooting Evolution', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # --- PANEL 4: Pace ---
    ax4 = fig.add_subplot(gs[2, 0])
    
    if 'Pace' in plot_df.columns:
        ax4.plot(plot_df['SEASON'], plot_df['Pace'], 
                'o-', linewidth=2.5, markersize=6, color='#CE1141', label='Team Pace')
        ax4.plot(plot_df['SEASON'], plot_df['Pace_League'], 
                '--', linewidth=2, color='gray', alpha=0.6, label='League Avg')
    
    ax4.set_xlabel('Season', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Possessions per 48 min', fontsize=12, fontweight='bold')
    ax4.set_title('Playing Speed (Pace)', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # --- PANEL 5: Shooting Efficiency ---
    ax5 = fig.add_subplot(gs[2, 1])
    
    if 'eFG%' in plot_df.columns and 'TS%' in plot_df.columns:
        ax5.plot(plot_df['SEASON'], plot_df['eFG%'], 
                'o-', linewidth=2.5, markersize=6, color='#CE1141', label='Team eFG%')
        ax5.plot(plot_df['SEASON'], plot_df['TS%'], 
                's-', linewidth=2.5, markersize=6, color='#000000', label='Team TS%')
        ax5.plot(plot_df['SEASON'], plot_df['eFG%_League'], 
                '--', linewidth=2, color='gray', alpha=0.5, label='League Avg eFG%')
    
    ax5.set_xlabel('Season', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Shooting %', fontsize=12, fontweight='bold')
    ax5.set_title('Shooting Efficiency', fontsize=14, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # --- PANEL 6: Ball Movement (AST% & TOV%) ---
    ax6 = fig.add_subplot(gs[3, :])
    
    if 'AST%' in plot_df.columns:
        ax6.plot(plot_df['SEASON'], plot_df['AST%'], 
                'o-', linewidth=2.5, markersize=6, color='#CE1141', label='Team AST%')
        ax6.plot(plot_df['SEASON'], plot_df['AST%_League'], 
                '--', linewidth=2, color='gray', alpha=0.6, label='League Avg AST%')
    
    # Add TOV% on secondary y-axis
    if 'TOV%' in plot_df.columns:
        ax6_twin = ax6.twinx()
        ax6_twin.plot(plot_df['SEASON'], plot_df['TOV%'], 
                     's-', linewidth=2.5, markersize=6, color='#000000', label='Team TOV%')
        ax6_twin.plot(plot_df['SEASON'], plot_df['TOV%_League'], 
                     '--', linewidth=2, color='#555555', alpha=0.6, label='League Avg TOV%')
        ax6_twin.set_ylabel('Turnover %', fontsize=12, fontweight='bold')
        ax6_twin.legend(loc='upper right', fontsize=9)
        ax6_twin.grid(False)
    
    ax6.set_xlabel('Season', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Assist %', fontsize=12, fontweight='bold')
    ax6.set_title('Ball Movement & Turnovers', fontsize=14, fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    ax6.legend(loc='upper left', fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'{team_name} Franchise Evolution Analysis (1995-2024)', 
                fontsize=18, fontweight='bold', y=0.995)
    
    filename = f"{team_name.replace(' ', '_')}_evolution.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {filename}")
    
    return fig


def create_comparison_to_league(team_df, league_avg, team_name='Toronto Raptors'):
    """
    Create heatmap showing team vs league average
    """
    
    # Merge data
    plot_df = team_df.merge(league_avg, on='SEASON', how='left')
    
    # Metrics to compare
    metrics = ['ORtg', 'DRtg', '3PA', 'Pace', 'eFG%', 'TS%', 'AST%', 'TOV%']
    
    # Calculate difference from league average
    diff_data = []
    
    for _, row in plot_df.iterrows():
        row_data = {'SEASON': row['SEASON']}
        for metric in metrics:
            if metric in plot_df.columns and f'{metric}_League' in plot_df.columns:
                team_val = row[metric]
                league_val = row[f'{metric}_League']
                
                if not pd.isna(team_val) and not pd.isna(league_val):
                    # For DRtg and TOV%, lower is better, so invert the difference
                    if metric in ['DRtg', 'TOV%']:
                        diff = league_val - team_val  # Better if positive
                    else:
                        diff = team_val - league_val
                    
                    row_data[metric] = diff
        
        diff_data.append(row_data)
    
    diff_df = pd.DataFrame(diff_data)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Prepare data for heatmap
    heatmap_data = diff_df.set_index('SEASON')[metrics].T
    
    # Create heatmap
    sns.heatmap(heatmap_data, cmap='RdYlGn', center=0, 
               annot=True, fmt='.1f', cbar_kws={'label': 'Difference from League Avg'},
               linewidths=0.5, ax=ax)
    
    ax.set_title(f'{team_name} vs League Average\n(Green = Better than league, Red = Worse than league)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Season', fontsize=13, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=13, fontweight='bold')
    
    filename = f"{team_name.replace(' ', '_')}_vs_league.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    
    return fig


def create_era_summary(team_df, team_name='Toronto Raptors'):
    """
    Create summary table by franchise era
    """
    
    eras = FRANCHISE_ERAS.get(team_name, [])
    
    if not eras:
        print(f"No eras defined for {team_name}")
        return None
    
    print("\n" + "=" * 120)
    print(f"{team_name.upper()} - FRANCHISE ERA SUMMARY")
    print("=" * 120)
    
    metrics = ['WIN_PCT', 'ORtg', 'DRtg', '3PA', 'Pace', 'eFG%', 'TS%', 'AST%']
    
    print(f"\n{'Era':<30} {'Years':<12} {'Seasons':<10} {'Avg Win%':<12} {'Avg ORtg':<12} "
          f"{'Avg DRtg':<12} {'Playoff Success':<30}")
    print("-" * 120)
    
    for era in eras:
        era_df = team_df[(team_df['YEAR'] >= era['start']) & (team_df['YEAR'] <= era['end'])]
        
        if len(era_df) == 0:
            continue
        
        # Calculate averages
        avg_win = era_df['WIN_PCT'].mean() if 'WIN_PCT' in era_df.columns else 0
        avg_ortg = era_df['ORtg'].mean() if 'ORtg' in era_df.columns else 0
        avg_drtg = era_df['DRtg'].mean() if 'DRtg' in era_df.columns else 0
        
        # Playoff success
        if 'Playoff_Result' in era_df.columns:
            playoff_results = era_df['Playoff_Result'].value_counts()
            best_result = era_df.loc[era_df['Playoff_Depth'].idxmax(), 'Playoff_Result'] if 'Playoff_Depth' in era_df.columns else 'N/A'
        else:
            best_result = 'N/A'
        
        year_range = f"{era['start']}-{era['end']}"
        
        print(f"{era['name']:<30} {year_range:<12} {len(era_df):<10} {avg_win:<12.3f} {avg_ortg:<12.1f} "
              f"{avg_drtg:<12.1f} {best_result:<30}")
    
    print("=" * 120)
    
    # Add regime analysis if we have coach/GM data
    if 'Head_Coach' in team_df.columns and 'General_Manager' in team_df.columns:
        print("\n" + "=" * 120)
        print(f"{team_name.upper()} - COACHING & GM REGIMES")
        print("=" * 120)
        
        # Coach summary
        print("\nHEAD COACHES:")
        print("-" * 80)
        
        coach_summary = team_df.groupby('Head_Coach').agg({
            'SEASON': ['first', 'last', 'count'],
            'WIN_PCT': 'mean',
            'Playoff_Result': lambda x: 'Made Playoffs' if any('Round' in str(v) or 'Finals' in str(v) or 'Champions' in str(v) for v in x) else 'No Playoffs'
        }).round(3)
        
        for coach, data in coach_summary.iterrows():
            seasons = int(data[('SEASON', 'count')])
            first_season = data[('SEASON', 'first')]
            last_season = data[('SEASON', 'last')]
            avg_win_pct = data[('WIN_PCT', 'mean')]
            playoffs = data[('Playoff_Result', '<lambda>')]
            
            print(f"{coach:<25} | {first_season} to {last_season} | {seasons} seasons | Win%: {avg_win_pct:.3f} | {playoffs}")
        
        # GM summary
        print("\n\nGENERAL MANAGERS:")
        print("-" * 80)
        
        gm_summary = team_df.groupby('General_Manager').agg({
            'SEASON': ['first', 'last', 'count'],
            'WIN_PCT': 'mean',
            'Playoff_Depth': 'max'
        }).round(3)
        
        for gm, data in gm_summary.iterrows():
            seasons = int(data[('SEASON', 'count')])
            first_season = data[('SEASON', 'first')]
            last_season = data[('SEASON', 'last')]
            avg_win_pct = data[('WIN_PCT', 'mean')]
            best_playoff = int(data[('Playoff_Depth', 'max')]) if pd.notna(data[('Playoff_Depth', 'max')]) else 0
            
            playoff_labels = ['None', 'R1', 'R2', 'Conf Finals', 'Finals', 'Champions']
            best_result = playoff_labels[best_playoff] if best_playoff < len(playoff_labels) else 'Unknown'
            
            print(f"{gm:<25} | {first_season} to {last_season} | {seasons} seasons | Win%: {avg_win_pct:.3f} | Best: {best_result}")
        
        print("=" * 120)


def main():
    """
    Main execution
    """
    
    print("=" * 100)
    print("NBA FRANCHISE EVOLUTION ANALYSIS")
    print("=" * 100)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Select team
    print("\nDefault: Toronto Raptors")
    team_name = input("Enter team name (or press Enter for Raptors): ").strip()
    if not team_name:
        team_name = 'Toronto Raptors'
    
    # Get franchise data
    print(f"\n{'='*100}")
    print(f"ANALYZING: {team_name}")
    print(f"{'='*100}")
    
    team_df = get_franchise_data(df, team_name)
    if team_df is None:
        return
    
    print(f"\n✓ Found {len(team_df)} seasons of data for {team_name}")
    print(f"  Range: {team_df['SEASON'].min()} to {team_df['SEASON'].max()}")
    
    # Calculate league averages
    league_avg = calculate_league_averages(df)
    
    # Create visualizations
    print(f"\n{'='*100}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*100}")
    
    create_main_visualization(team_df, league_avg, team_name)
    create_comparison_to_league(team_df, league_avg, team_name)
    
    # Era summary
    create_era_summary(team_df, team_name)
    
    # Save data
    output_file = f"{team_name.replace(' ', '_')}_data.csv"
    team_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved data: {output_file}")
    
    print("\n" + "=" * 100)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 100)
    print("\nFiles created:")
    print(f"  1. {team_name.replace(' ', '_')}_evolution.png - Main visualization")
    print(f"  2. {team_name.replace(' ', '_')}_vs_league.png - Comparison heatmap")
    print(f"  3. {output_file} - Raw data")
    
    return team_df


if __name__ == "__main__":
    team_df = main()
