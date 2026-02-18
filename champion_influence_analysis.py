"""
Champion Influence Analysis
Do NBA teams copy the champion's playing style in subsequent seasons?

Tests the hypothesis: "Teams adjust their playing style toward the previous champion"

Author: GY Mehta
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

def load_data(filename='nba_team_stats_1980_2025.csv'):
    """Load the NBA data"""
    try:
        df = pd.read_csv(filename)
        print(f"✓ Data loaded: {len(df)} rows")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find '{filename}'")
        return None


def identify_champions(df):
    """
    Use actual NBA playoff champions (not just most wins)
    """
    
    # Actual NBA Champions (Playoff Winners) 1997-2025
    NBA_CHAMPIONS = {
        '1996-97': 'Chicago Bulls',
        '1997-98': 'Chicago Bulls',
        '1998-99': 'San Antonio Spurs',
        '1999-00': 'Los Angeles Lakers',
        '2000-01': 'Los Angeles Lakers',
        '2001-02': 'Los Angeles Lakers',
        '2002-03': 'San Antonio Spurs',
        '2003-04': 'Detroit Pistons',
        '2004-05': 'San Antonio Spurs',
        '2005-06': 'Miami Heat',
        '2006-07': 'San Antonio Spurs',
        '2007-08': 'Boston Celtics',
        '2008-09': 'Los Angeles Lakers',
        '2009-10': 'Los Angeles Lakers',
        '2010-11': 'Dallas Mavericks',
        '2011-12': 'Miami Heat',
        '2012-13': 'Miami Heat',
        '2013-14': 'San Antonio Spurs',
        '2014-15': 'Golden State Warriors',
        '2015-16': 'Cleveland Cavaliers',
        '2016-17': 'Golden State Warriors',
        '2017-18': 'Golden State Warriors',
        '2018-19': 'Toronto Raptors',
        '2019-20': 'Los Angeles Lakers',
        '2020-21': 'Milwaukee Bucks',
        '2021-22': 'Golden State Warriors',
        '2022-23': 'Denver Nuggets',
        '2023-24': 'Boston Celtics',
    }
    
    champions = []
    
    for season in sorted(df['SEASON'].unique()):
        if season not in NBA_CHAMPIONS:
            continue
        
        champ_name = NBA_CHAMPIONS[season]
        season_df = df[df['SEASON'] == season].copy()
        
        # Find the champion's data
        champ_row = season_df[season_df['Team'] == champ_name]
        
        if len(champ_row) == 0:
            print(f"  ⚠ Warning: {champ_name} not found in {season} data")
            continue
        
        champ_row = champ_row.iloc[0]
        
        champ_data = {
            'SEASON': season,
            'YEAR': champ_row['YEAR'] if 'YEAR' in champ_row else None,
            'Team': champ_name,
            'Wins': champ_row['W'] if 'W' in champ_row else None,
        }
        
        # Get key style metrics
        style_metrics = ['3PA', '3P%', 'Pace', 'ORtg', 'DRtg', 'eFG%', 'TS%', 
                       'AST%', 'ORB%', 'DRB%', 'TOV%', 'FTA_RATE']
        
        for metric in style_metrics:
            if metric in season_df.columns:
                champ_data[f'Champ_{metric}'] = champ_row[metric]
        
        champions.append(champ_data)
    
    return pd.DataFrame(champions)


def calculate_league_convergence(df, champions_df, key_metrics=['3PA', 'Pace', 'ORtg', 'DRtg']):
    """
    Test if league average moves toward champion's style in following season
    
    Method:
    1. For each season, calculate how far league average is from champion's stats
    2. In next season, check if league average got closer to previous champion
    3. If teams are copying champions, distance should decrease
    """
    
    results = []
    
    seasons = sorted(df['SEASON'].unique())
    
    for i in range(len(seasons) - 1):
        current_season = seasons[i]
        next_season = seasons[i + 1]
        
        # Get current season's champion
        champ_row = champions_df[champions_df['SEASON'] == current_season]
        if len(champ_row) == 0:
            continue
        
        champ_row = champ_row.iloc[0]
        
        # Get league averages for current and next season
        current_league = df[df['SEASON'] == current_season]
        next_league = df[df['SEASON'] == next_season]
        
        if len(current_league) < 5 or len(next_league) < 5:
            continue
        
        row_data = {
            'Champion_Season': current_season,
            'Following_Season': next_season,
            'Champion': champ_row['Team']
        }
        
        for metric in key_metrics:
            champ_metric = f'Champ_{metric}'
            
            if champ_metric not in champ_row or pd.isna(champ_row[champ_metric]):
                continue
            
            if metric not in current_league.columns or metric not in next_league.columns:
                continue
            
            # Champion's value
            champ_value = champ_row[champ_metric]
            
            # League averages
            current_avg = current_league[metric].mean()
            next_avg = next_league[metric].mean()
            
            # Distance from champion
            current_distance = abs(current_avg - champ_value)
            next_distance = abs(next_avg - champ_value)
            
            # Did league move toward champion?
            moved_toward = next_distance < current_distance
            change = next_avg - current_avg
            
            row_data[f'{metric}_ChampValue'] = champ_value
            row_data[f'{metric}_CurrentAvg'] = current_avg
            row_data[f'{metric}_NextAvg'] = next_avg
            row_data[f'{metric}_CurrentDist'] = current_distance
            row_data[f'{metric}_NextDist'] = next_distance
            row_data[f'{metric}_MovedToward'] = moved_toward
            row_data[f'{metric}_Change'] = change
        
        results.append(row_data)
    
    return pd.DataFrame(results)


def analyze_individual_team_shifts(df, champions_df, key_metrics=['3PA', 'Pace']):
    """
    Analyze if individual teams shift their style toward champion
    
    For each team, compare their change in playing style to the champion's style
    """
    
    results = []
    
    seasons = sorted(df['SEASON'].unique())
    
    for i in range(len(seasons) - 1):
        current_season = seasons[i]
        next_season = seasons[i + 1]
        
        # Get champion
        champ_row = champions_df[champions_df['SEASON'] == current_season]
        if len(champ_row) == 0:
            continue
        champ_row = champ_row.iloc[0]
        
        # Get teams in both seasons
        current_teams = df[df['SEASON'] == current_season].copy()
        next_teams = df[df['SEASON'] == next_season].copy()
        
        for _, team_current in current_teams.iterrows():
            team_name = team_current['Team']
            
            # Skip the champion itself
            if team_name == champ_row['Team']:
                continue
            
            # Find same team next season
            team_next = next_teams[next_teams['Team'] == team_name]
            
            if len(team_next) == 0:
                continue
            
            team_next = team_next.iloc[0]
            
            row_data = {
                'Champion_Season': current_season,
                'Team': team_name,
                'Champion': champ_row['Team']
            }
            
            for metric in key_metrics:
                champ_metric = f'Champ_{metric}'
                
                if champ_metric not in champ_row or pd.isna(champ_row[champ_metric]):
                    continue
                
                if metric not in current_teams.columns:
                    continue
                
                champ_value = champ_row[champ_metric]
                team_current_value = team_current[metric]
                team_next_value = team_next[metric]
                
                # Distance from champion before and after
                dist_before = abs(team_current_value - champ_value)
                dist_after = abs(team_next_value - champ_value)
                
                moved_toward = dist_after < dist_before
                
                row_data[f'{metric}_MovedToward'] = moved_toward
                row_data[f'{metric}_Change'] = team_next_value - team_current_value
                row_data[f'{metric}_DistChange'] = dist_after - dist_before
            
            results.append(row_data)
    
    return pd.DataFrame(results)


def create_summary_report(convergence_df, team_shifts_df, key_metrics=['3PA', 'Pace', 'ORtg', 'DRtg']):
    """
    Create summary report of findings
    """
    
    print("\n" + "=" * 100)
    print("CHAMPION INFLUENCE ANALYSIS - DO TEAMS COPY THE CHAMPION?")
    print("=" * 100)
    
    # League-level analysis
    print("\n" + "=" * 100)
    print("LEAGUE-LEVEL CONVERGENCE")
    print("=" * 100)
    print("\nDoes the league average move toward the champion's style?")
    print()
    
    print(f"{'Metric':<15} {'Times Moved':<15} {'Total Seasons':<15} {'% Moved Toward':<20} {'Copying?':<15}")
    print("-" * 100)
    
    for metric in key_metrics:
        col = f'{metric}_MovedToward'
        if col in convergence_df.columns:
            moved_count = convergence_df[col].sum()
            total = len(convergence_df[col].dropna())
            pct = (moved_count / total * 100) if total > 0 else 0
            
            # If > 50%, league is copying champion
            copying = "YES ✓" if pct > 50 else "NO ✗"
            
            print(f"{metric:<15} {moved_count:<15.0f} {total:<15} {pct:<20.1f}% {copying:<15}")
    
    # Team-level analysis
    if not team_shifts_df.empty:
        print("\n" + "=" * 100)
        print("INDIVIDUAL TEAM BEHAVIOR")
        print("=" * 100)
        print("\nDo individual teams shift toward the champion's style?")
        print()
        
        print(f"{'Metric':<15} {'Teams Moved':<15} {'Total Teams':<15} {'% Moved Toward':<20} {'Copying?':<15}")
        print("-" * 100)
        
        for metric in key_metrics:
            col = f'{metric}_MovedToward'
            if col in team_shifts_df.columns:
                moved_count = team_shifts_df[col].sum()
                total = len(team_shifts_df[col].dropna())
                pct = (moved_count / total * 100) if total > 0 else 0
                
                copying = "YES ✓" if pct > 50 else "NO ✗"
                
                print(f"{metric:<15} {moved_count:<15.0f} {total:<15} {pct:<20.1f}% {copying:<15}")
    
    # Statistical significance test
    print("\n" + "=" * 100)
    print("STATISTICAL SIGNIFICANCE TEST")
    print("=" * 100)
    print("\nBinomial test: Is copying more common than random chance (50%)?")
    print()
    
    for metric in key_metrics:
        col = f'{metric}_MovedToward'
        if col in convergence_df.columns:
            moved_count = convergence_df[col].sum()
            total = len(convergence_df[col].dropna())
            
            if total > 0:
                # Binomial test (null hypothesis: p = 0.5)
                from scipy.stats import binomtest
                result = binomtest(moved_count, total, 0.5, alternative='greater')
                p_value = result.pvalue
                
                significant = "SIGNIFICANT ✓✓✓" if p_value < 0.05 else "Not significant"
                
                print(f"{metric:<15} p-value: {p_value:.4f}  {significant}")


def create_visualizations(convergence_df, champions_df, key_metrics=['3PA', 'Pace']):
    """
    Create visualizations of champion influence
    """
    
    n_metrics = len(key_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 5 * n_metrics))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(key_metrics):
        ax = axes[idx]
        
        # Get champion values and league averages over time
        champ_col = f'Champ_{metric}'
        next_avg_col = f'{metric}_NextAvg'
        
        if champ_col not in champions_df.columns or next_avg_col not in convergence_df.columns:
            continue
        
        # Merge data
        plot_data = convergence_df[['Champion_Season', 'Following_Season', 'Champion', 
                                    f'{metric}_ChampValue', f'{metric}_NextAvg']].copy()
        
        # Plot champion values
        ax.plot(plot_data['Champion_Season'], plot_data[f'{metric}_ChampValue'], 
               'ro-', linewidth=3, markersize=10, label='Champion Value', alpha=0.7)
        
        # Plot league average in following season
        ax.plot(plot_data['Following_Season'], plot_data[f'{metric}_NextAvg'], 
               'bs--', linewidth=2, markersize=8, label='League Avg (Next Season)', alpha=0.7)
        
        ax.set_xlabel('Season', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric}: Champion vs League Average Over Time', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('champion_influence.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: champion_influence.png")
    
    return fig


def main():
    """
    Main execution
    """
    
    print("=" * 100)
    print("CHAMPION INFLUENCE ANALYSIS")
    print("=" * 100)
    print("\nHypothesis: Teams copy the NBA CHAMPION's (playoff winner) playing style")
    print("Method: Check if league/teams move toward champion's stats in following season")
    print("\nNote: Using actual playoff champions, not regular season winners")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    print(f"\nAnalyzing {df['SEASON'].nunique()} seasons...")
    
    # Identify champions
    print("\n" + "=" * 100)
    print("STEP 1: Identifying Champions")
    print("=" * 100)
    
    champions_df = identify_champions(df)
    print(f"\n✓ Found {len(champions_df)} champions")
    print("\nRecent champions:")
    print(champions_df[['SEASON', 'Team', 'Wins']].tail(10).to_string(index=False))
    
    # Key metrics to analyze
    key_metrics = ['3PA', 'Pace', 'ORtg', 'DRtg', 'eFG%', 'TS%']
    
    # League convergence analysis
    print("\n" + "=" * 100)
    print("STEP 2: League Convergence Analysis")
    print("=" * 100)
    print("\nAnalyzing if league average moves toward champion...")
    
    convergence_df = calculate_league_convergence(df, champions_df, key_metrics)
    print(f"✓ Analyzed {len(convergence_df)} season transitions")
    
    # Individual team shifts
    print("\n" + "=" * 100)
    print("STEP 3: Individual Team Analysis")
    print("=" * 100)
    print("\nAnalyzing if teams shift toward champion...")
    
    team_shifts_df = analyze_individual_team_shifts(df, champions_df, key_metrics)
    print(f"✓ Analyzed {len(team_shifts_df)} team transitions")
    
    # Create summary report
    create_summary_report(convergence_df, team_shifts_df, key_metrics)
    
    # Create visualizations
    print("\n" + "=" * 100)
    print("STEP 4: Creating Visualizations")
    print("=" * 100)
    
    create_visualizations(convergence_df, champions_df, key_metrics=['3PA', 'Pace'])
    
    # Save data
    convergence_df.to_csv('champion_influence_league.csv', index=False)
    team_shifts_df.to_csv('champion_influence_teams.csv', index=False)
    champions_df.to_csv('nba_champions_by_season.csv', index=False)
    
    print(f"\n✓ Saved data files:")
    print(f"  - champion_influence_league.csv")
    print(f"  - champion_influence_teams.csv")
    print(f"  - nba_champions_by_season.csv")
    
    print("\n" + "=" * 100)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 100)
    
    print("\n KEY FINDINGS SUMMARY:")
    print("  Check the report above to see:")
    print("  ✓ Which metrics show copying behavior (>50% convergence)")
    print("  ✓ Whether it's statistically significant (p < 0.05)")
    print("  ✓ If trends differ between league-level vs individual teams")
    
    return convergence_df, team_shifts_df, champions_df


if __name__ == "__main__":
    convergence_df, team_shifts_df, champions_df = main()
