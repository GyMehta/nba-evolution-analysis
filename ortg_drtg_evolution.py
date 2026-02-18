"""
Evolution of Offensive vs Defensive Rating Importance
Track how ORtg and DRtg correlations with winning have changed over 10 seasons

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
plt.rcParams['figure.figsize'] = (14, 8)

def load_data(filename='nba_team_stats_1980_2025.csv'):
    """Load the NBA data"""
    try:
        df = pd.read_csv(filename)
        print(f"✓ Data loaded: {len(df)} rows")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find '{filename}'")
        return None


def calculate_correlation_by_season(df, metrics=['ORtg', 'DRtg'], target='WIN_PCT'):
    """
    Calculate correlations for specific metrics across all seasons
    """
    
    seasons = sorted(df['SEASON'].unique())
    
    results = []
    
    for season in seasons:
        season_df = df[df['SEASON'] == season].copy()
        
        if len(season_df) < 5:  # Need enough teams
            continue
        
        row_data = {'SEASON': season}
        
        # Extract year for plotting
        year = int(season.split('-')[0]) + 1  # Convert "2018-19" to 2019
        row_data['YEAR'] = year
        
        for metric in metrics:
            if metric not in season_df.columns:
                row_data[f'{metric}_Correlation'] = None
                row_data[f'{metric}_PValue'] = None
                continue
            
            # Remove NaN values
            valid_data = season_df[[metric, target]].dropna()
            
            if len(valid_data) > 3:
                corr, p_value = stats.pearsonr(valid_data[metric], valid_data[target])
                row_data[f'{metric}_Correlation'] = corr
                row_data[f'{metric}_PValue'] = p_value
            else:
                row_data[f'{metric}_Correlation'] = None
                row_data[f'{metric}_PValue'] = None
        
        results.append(row_data)
    
    results_df = pd.DataFrame(results)
    
    return results_df


def plot_correlation_evolution(results_df, metrics=['ORtg', 'DRtg'], n_seasons=10):
    """
    Create visualization showing how correlations changed over time
    """
    
    # Get last N seasons
    recent_df = results_df.tail(n_seasons).copy()
    
    if len(recent_df) == 0:
        print("Error: No data to plot")
        return
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Color scheme
    colors = {'ORtg': '#d62728', 'DRtg': '#1f77b4'}  # Red for offense, blue for defense
    
    # Plot 1: Line chart of correlations over time
    for metric in metrics:
        corr_col = f'{metric}_Correlation'
        if corr_col in recent_df.columns:
            # For DRtg, plot absolute value since negative correlation is "good"
            if metric == 'DRtg':
                values = recent_df[corr_col].abs()
                label = f'{metric} (Absolute)'
            else:
                values = recent_df[corr_col]
                label = metric
            
            ax1.plot(recent_df['SEASON'], values, 
                    marker='o', linewidth=3, markersize=10,
                    color=colors.get(metric, 'gray'), 
                    label=label, alpha=0.8)
    
    ax1.set_xlabel('Season', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Correlation with Win%', fontsize=13, fontweight='bold')
    ax1.set_title('Evolution of Offensive vs Defensive Rating Importance\nCorrelation with Winning (Last 10 Seasons)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Rotate x-axis labels
    ax1.tick_params(axis='x', rotation=45)
    
    # Add annotations for highest/lowest points
    for metric in metrics:
        corr_col = f'{metric}_Correlation'
        if corr_col in recent_df.columns:
            values = recent_df[corr_col].abs() if metric == 'DRtg' else recent_df[corr_col]
            max_idx = values.idxmax()
            max_season = recent_df.loc[max_idx, 'SEASON']
            max_val = values.loc[max_idx]
            
            ax1.annotate(f'{max_val:.3f}', 
                        xy=(max_season, max_val),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=colors.get(metric, 'gray'), alpha=0.3),
                        arrowprops=dict(arrowstyle='->', color=colors.get(metric, 'gray'), lw=2))
    
    # Plot 2: Bar chart comparing most recent vs 10 years ago
    if len(recent_df) >= 2:
        first_season = recent_df.iloc[0]
        last_season = recent_df.iloc[-1]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        first_vals = [abs(first_season[f'{m}_Correlation']) if m == 'DRtg' else first_season[f'{m}_Correlation'] 
                     for m in metrics]
        last_vals = [abs(last_season[f'{m}_Correlation']) if m == 'DRtg' else last_season[f'{m}_Correlation'] 
                    for m in metrics]
        
        bars1 = ax2.bar(x - width/2, first_vals, width, 
                       label=f"{first_season['SEASON']}", alpha=0.8, color='#95a5a6')
        bars2 = ax2.bar(x + width/2, last_vals, width, 
                       label=f"{last_season['SEASON']}", alpha=0.8, color='#2ecc71')
        
        ax2.set_xlabel('Metric', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Correlation Strength', fontsize=13, fontweight='bold')
        ax2.set_title(f'Then vs Now: Change in Importance Over {n_seasons} Seasons', 
                     fontsize=14, fontweight='bold', pad=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{m}\n(Absolute)' if m == 'DRtg' else m for m in metrics])
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ortg_drtg_evolution.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: ortg_drtg_evolution.png")
    
    return fig


def get_season_stats(df, season):
    """
    Get detailed stats for a season: averages, best teams, champion
    """
    season_df = df[df['SEASON'] == season].copy()
    
    if len(season_df) == 0:
        return None
    
    stats = {
        'SEASON': season,
        'Avg_ORtg': season_df['ORtg'].mean(),
        'Avg_DRtg': season_df['DRtg'].mean(),
    }
    
    # Best ORtg
    if 'ORtg' in season_df.columns:
        best_ortg_idx = season_df['ORtg'].idxmax()
        stats['Best_ORtg'] = season_df.loc[best_ortg_idx, 'ORtg']
        stats['Best_ORtg_Team'] = season_df.loc[best_ortg_idx, 'Team']
    
    # Best DRtg (lowest is best)
    if 'DRtg' in season_df.columns:
        best_drtg_idx = season_df['DRtg'].idxmin()
        stats['Best_DRtg'] = season_df.loc[best_drtg_idx, 'DRtg']
        stats['Best_DRtg_Team'] = season_df.loc[best_drtg_idx, 'Team']
    
    # Champion (team with most wins)
    if 'W' in season_df.columns:
        champ_idx = season_df['W'].idxmax()
        stats['Champion_Team'] = season_df.loc[champ_idx, 'Team']
        stats['Champion_Wins'] = season_df.loc[champ_idx, 'W']
        stats['Champion_ORtg'] = season_df.loc[champ_idx, 'ORtg']
        stats['Champion_DRtg'] = season_df.loc[champ_idx, 'DRtg']
    
    return stats


def create_detailed_season_table(df, results_df, n_seasons=10):
    """
    Create detailed table with season stats
    """
    recent_seasons = results_df.tail(n_seasons)['SEASON'].tolist()
    
    print("\n" + "=" * 120)
    print(f"DETAILED SEASON ANALYSIS - LAST {n_seasons} SEASONS")
    print("=" * 120)
    
    # Header
    print(f"\n{'Season':<10} {'Avg':<8} {'Best':<8} {'Best Team':<20} "
          f"{'Avg':<8} {'Best':<8} {'Best Team':<20} {'Champion':<20}")
    print(f"{'':10} {'ORtg':8} {'ORtg':8} {'(Offense)':20} "
          f"{'DRtg':8} {'DRtg':8} {'(Defense)':20} {'(Most Wins)':20}")
    print("-" * 120)
    
    all_season_stats = []
    
    for season in recent_seasons:
        stats = get_season_stats(df, season)
        if stats is None:
            continue
        
        all_season_stats.append(stats)
        
        print(f"{stats['SEASON']:<10} "
              f"{stats.get('Avg_ORtg', 0):>7.1f} "
              f"{stats.get('Best_ORtg', 0):>7.1f} "
              f"{stats.get('Best_ORtg_Team', 'N/A'):<20} "
              f"{stats.get('Avg_DRtg', 0):>7.1f} "
              f"{stats.get('Best_DRtg', 0):>7.1f} "
              f"{stats.get('Best_DRtg_Team', 'N/A'):<20} "
              f"{stats.get('Champion_Team', 'N/A'):<20}")
    
    # Champion details
    print("\n" + "=" * 120)
    print("CHAMPION RATINGS")
    print("=" * 120)
    print(f"\n{'Season':<10} {'Champion':<25} {'Wins':<6} {'ORtg':<8} {'DRtg':<8} {'Net Rtg':<8}")
    print("-" * 120)
    
    for stats in all_season_stats:
        if 'Champion_Team' in stats:
            net_rtg = stats.get('Champion_ORtg', 0) - stats.get('Champion_DRtg', 0)
            print(f"{stats['SEASON']:<10} "
                  f"{stats.get('Champion_Team', 'N/A'):<25} "
                  f"{stats.get('Champion_Wins', 0):<6.0f} "
                  f"{stats.get('Champion_ORtg', 0):>7.1f} "
                  f"{stats.get('Champion_DRtg', 0):>7.1f} "
                  f"{net_rtg:>7.1f}")
    
    return all_season_stats


def create_summary_table(results_df, metrics=['ORtg', 'DRtg'], n_seasons=10):
    """
    Create a summary table of correlations
    """
    
    recent_df = results_df.tail(n_seasons).copy()
    
    print("\n" + "=" * 80)
    print(f"CORRELATION EVOLUTION - LAST {n_seasons} SEASONS")
    print("=" * 80)
    print("\nNote: DRtg shown as absolute value (negative correlation = good defense matters)")
    print()
    
    # Format output
    print(f"{'Season':<12} {'ORtg Corr':>12} {'DRtg Corr':>12} {'Offense > Defense?':>20}")
    print("-" * 80)
    
    for _, row in recent_df.iterrows():
        season = row['SEASON']
        ortg_corr = row.get('ORtg_Correlation', None)
        drtg_corr = row.get('DRtg_Correlation', None)
        
        if ortg_corr is not None and drtg_corr is not None:
            # Use absolute value for DRtg comparison
            drtg_abs = abs(drtg_corr)
            
            if ortg_corr > drtg_abs:
                winner = "✓ Offense"
            elif drtg_abs > ortg_corr:
                winner = "✓ Defense"
            else:
                winner = "Tie"
            
            print(f"{season:<12} {ortg_corr:>12.3f} {drtg_abs:>12.3f} {winner:>20}")
    
    # Calculate averages
    print("-" * 80)
    ortg_avg = recent_df['ORtg_Correlation'].mean()
    drtg_avg = recent_df['DRtg_Correlation'].abs().mean()
    
    print(f"{'AVERAGE':<12} {ortg_avg:>12.3f} {drtg_avg:>12.3f}")
    
    # Trend analysis
    print("\n" + "=" * 80)
    print("TREND ANALYSIS")
    print("=" * 80)
    
    first_half = recent_df.head(n_seasons // 2)
    second_half = recent_df.tail(n_seasons // 2)
    
    ortg_first = first_half['ORtg_Correlation'].mean()
    ortg_second = second_half['ORtg_Correlation'].mean()
    ortg_change = ortg_second - ortg_first
    
    drtg_first = first_half['DRtg_Correlation'].abs().mean()
    drtg_second = second_half['DRtg_Correlation'].abs().mean()
    drtg_change = drtg_second - drtg_first
    
    print(f"\nOffensive Rating:")
    print(f"  Early period avg: {ortg_first:.3f}")
    print(f"  Recent period avg: {ortg_second:.3f}")
    print(f"  Change: {ortg_change:+.3f} {'(INCREASING)' if ortg_change > 0 else '(DECREASING)'}")
    
    print(f"\nDefensive Rating:")
    print(f"  Early period avg: {drtg_first:.3f}")
    print(f"  Recent period avg: {drtg_second:.3f}")
    print(f"  Change: {drtg_change:+.3f} {'(INCREASING)' if drtg_change > 0 else '(DECREASING)'}")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    
    if ortg_avg > drtg_avg:
        diff = ortg_avg - drtg_avg
        print(f"\nOver the last {n_seasons} seasons, OFFENSE has been MORE important than defense")
        print(f"for winning, with an average correlation difference of {diff:.3f}")
    else:
        diff = drtg_avg - ortg_avg
        print(f"\nOver the last {n_seasons} seasons, DEFENSE has been MORE important than offense")
        print(f"for winning, with an average correlation difference of {diff:.3f}")
    
    if abs(ortg_change) > 0.05 or abs(drtg_change) > 0.05:
        print("\nThe balance between offense and defense IS SHIFTING over time!")
    else:
        print("\nThe balance between offense and defense has remained relatively STABLE.")


def main():
    """
    Main execution
    """
    
    print("=" * 80)
    print("OFFENSIVE vs DEFENSIVE RATING EVOLUTION ANALYSIS")
    print("=" * 80)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Calculate correlations by season
    print("\nCalculating correlations for each season...")
    results_df = calculate_correlation_by_season(df, metrics=['ORtg', 'DRtg'])
    
    # Get number of seasons to analyze
    print(f"\n✓ Found {len(results_df)} seasons with data")
    
    n_seasons = 10
    user_input = input(f"\nAnalyze last how many seasons? (default 10, press Enter): ").strip()
    if user_input.isdigit():
        n_seasons = int(user_input)
    
    # Create visualization
    print(f"\n{'='*80}")
    print(f"ANALYZING LAST {n_seasons} SEASONS")
    print(f"{'='*80}")
    
    plot_correlation_evolution(results_df, n_seasons=n_seasons)
    
    # Create detailed season stats table
    season_stats = create_detailed_season_table(df, results_df, n_seasons=n_seasons)
    
    # Create summary table
    create_summary_table(results_df, n_seasons=n_seasons)
    
    # Save data
    output_file = f'ortg_drtg_evolution_{n_seasons}seasons.csv'
    results_df.tail(n_seasons).to_csv(output_file, index=False)
    print(f"\n✓ Data saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nFiles created:")
    print("  1. ortg_drtg_evolution.png - Visual comparison")
    print(f"  2. {output_file} - Correlation data")
    
    return results_df


if __name__ == "__main__":
    results_df = main()
