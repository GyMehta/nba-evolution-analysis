"""
Single Season NBA Analysis
Analyze one season to understand correlations with Win%

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
plt.rcParams['font.size'] = 10

def load_data(filename='nba_team_stats_1980_2025.csv'):
    """Load the NBA data"""
    try:
        df = pd.read_csv(filename)
        print(f"✓ Data loaded: {len(df)} rows, {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find '{filename}'")
        print("Make sure the CSV file is in the same directory as this script.")
        return None


def explore_seasons(df):
    """Show available seasons"""
    print("\n" + "=" * 70)
    print("AVAILABLE SEASONS")
    print("=" * 70)
    
    seasons = df['SEASON'].unique()
    print(f"Total seasons: {len(seasons)}")
    print(f"Range: {df['SEASON'].min()} to {df['SEASON'].max()}")
    
    # Show seasons by era
    if 'ERA' in df.columns:
        print("\nSeasons by era:")
        print(df.groupby('ERA')['SEASON'].nunique())
    
    return seasons


def select_season(df, season=None):
    """
    Select a season for analysis
    Default: 2018-19 (interesting modern season)
    """
    
    if season is None:
        # Default to 2018-19 - good modern season before COVID
        season = '2018-19'
    
    season_df = df[df['SEASON'] == season].copy()
    
    if len(season_df) == 0:
        print(f"Error: Season '{season}' not found in data")
        return None
    
    print(f"\n✓ Selected season: {season}")
    print(f"  Teams: {len(season_df)}")
    
    return season_df


def identify_key_metrics(df):
    """
    Identify which key metrics are available in the dataset
    """
    
    # Metrics we want to analyze
    desired_metrics = {
        'Offensive': ['ORtg', 'PTS', 'FG%', '3P%', '3PA', 'eFG%', 'TS%', 
                     'AST', 'AST%', 'ORB%', 'FTA', 'FT%'],
        'Defensive': ['DRtg', 'OPP_PTS', 'DRB%', 'STL', 'BLK', 'OPP_FG%', 
                     'OPP_3P%', 'OPP_FTA'],
        'Style': ['Pace', 'TOV%', '3PA', 'FTA'],
    }
    
    available = {}
    for category, metrics in desired_metrics.items():
        available[category] = [m for m in metrics if m in df.columns]
    
    print("\n" + "=" * 70)
    print("AVAILABLE METRICS FOR ANALYSIS")
    print("=" * 70)
    
    for category, metrics in available.items():
        if metrics:
            print(f"\n{category}:")
            print(f"  {', '.join(metrics)}")
    
    return available


def calculate_correlations(df, target='WIN_PCT'):
    """
    Calculate correlations between all numeric metrics and Win%
    """
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove non-stat columns
    exclude_cols = ['TEAM_ID', 'GP', 'W', 'L', 'MIN', 'YEAR', 'W_PCT']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Calculate correlations with WIN_PCT
    correlations = []
    
    for col in numeric_cols:
        if col == target:
            continue
        
        # Remove NaN values
        valid_data = df[[col, target]].dropna()
        
        if len(valid_data) > 3:  # Need at least a few data points
            corr, p_value = stats.pearsonr(valid_data[col], valid_data[target])
            correlations.append({
                'Metric': col,
                'Correlation': corr,
                'Abs_Correlation': abs(corr),
                'P_Value': p_value,
                'Significant': p_value < 0.05
            })
    
    # Create dataframe and sort
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    
    return corr_df


def display_top_correlations(corr_df, n=15):
    """
    Display the top correlations with Win%
    """
    
    print("\n" + "=" * 70)
    print(f"TOP {n} CORRELATIONS WITH WIN%")
    print("=" * 70)
    
    top_corr = corr_df.head(n)
    
    for idx, row in top_corr.iterrows():
        sig = "***" if row['Significant'] else ""
        print(f"{row['Metric']:20s} {row['Correlation']:7.3f} {sig}")
    
    print("\n*** = Statistically significant (p < 0.05)")
    
    return top_corr


def create_correlation_plots(season_df, corr_df, n=6):
    """
    Create scatter plots for top correlations
    """
    
    top_metrics = corr_df.head(n)['Metric'].tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(top_metrics):
        ax = axes[idx]
        
        # Remove NaN
        plot_data = season_df[[metric, 'WIN_PCT', 'Team']].dropna()
        
        # Scatter plot
        ax.scatter(plot_data[metric], plot_data['WIN_PCT'], 
                  alpha=0.6, s=100, color='steelblue')
        
        # Add trend line
        z = np.polyfit(plot_data[metric], plot_data['WIN_PCT'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(plot_data[metric].min(), plot_data[metric].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
        
        # Get correlation
        corr = corr_df[corr_df['Metric'] == metric]['Correlation'].values[0]
        
        # Labels
        ax.set_xlabel(metric, fontsize=11, fontweight='bold')
        ax.set_ylabel('Win %', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric} vs Win%\nCorrelation: {corr:.3f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    season = season_df['SEASON'].iloc[0]
    plt.savefig(f'correlation_analysis_{season}.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot: correlation_analysis_{season}.png")
    
    return fig


def create_summary_report(season_df, corr_df):
    """
    Create a text summary report
    """
    
    season = season_df['SEASON'].iloc[0]
    
    print("\n" + "=" * 70)
    print(f"ANALYSIS SUMMARY - {season} SEASON")
    print("=" * 70)
    
    print(f"\nDataset:")
    print(f"  Teams analyzed: {len(season_df)}")
    print(f"  Metrics analyzed: {len(corr_df)}")
    
    # Top positive correlations (help win)
    print(f"\nTop 5 factors associated with WINNING:")
    top_positive = corr_df[corr_df['Correlation'] > 0].head(5)
    for idx, row in top_positive.iterrows():
        print(f"  {row['Metric']:20s} (r = {row['Correlation']:.3f})")
    
    # Top negative correlations (hurt winning)
    print(f"\nTop 5 factors associated with LOSING:")
    top_negative = corr_df[corr_df['Correlation'] < 0].head(5)
    for idx, row in top_negative.iterrows():
        print(f"  {row['Metric']:20s} (r = {row['Correlation']:.3f})")
    
    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    # Offensive vs Defensive rating
    if 'ORtg' in corr_df['Metric'].values and 'DRtg' in corr_df['Metric'].values:
        ortg_corr = corr_df[corr_df['Metric'] == 'ORtg']['Correlation'].values[0]
        drtg_corr = corr_df[corr_df['Metric'] == 'DRtg']['Correlation'].values[0]
        
        print(f"\nOffense vs Defense:")
        print(f"  Offensive Rating correlation: {ortg_corr:+.3f}")
        print(f"  Defensive Rating correlation: {drtg_corr:+.3f}")
        
        if abs(ortg_corr) > abs(drtg_corr):
            print(f"  → Offense was MORE important this season")
        else:
            print(f"  → Defense was MORE important this season")
    
    # 3-point shooting
    three_metrics = ['3P%', '3PA', '3PM']
    three_found = [m for m in three_metrics if m in corr_df['Metric'].values]
    
    if three_found:
        print(f"\n3-Point Shooting:")
        for metric in three_found:
            corr_val = corr_df[corr_df['Metric'] == metric]['Correlation'].values[0]
            print(f"  {metric} correlation: {corr_val:+.3f}")
    
    print("\n" + "=" * 70)


def main():
    """
    Main execution
    """
    
    print("=" * 70)
    print("SINGLE SEASON NBA ANALYSIS")
    print("=" * 70)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Show available seasons
    seasons = explore_seasons(df)
    
    # Select season (you can change this)
    print("\n" + "=" * 70)
    season_choice = input("Enter season to analyze (or press Enter for 2018-19): ").strip()
    if not season_choice:
        season_choice = '2018-19'
    
    season_df = select_season(df, season_choice)
    if season_df is None:
        return
    
    # Identify available metrics
    available_metrics = identify_key_metrics(season_df)
    
    # Calculate correlations
    print("\n" + "=" * 70)
    print("CALCULATING CORRELATIONS...")
    print("=" * 70)
    
    corr_df = calculate_correlations(season_df)
    
    # Display results
    top_corr = display_top_correlations(corr_df, n=15)
    
    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS...")
    print("=" * 70)
    
    create_correlation_plots(season_df, corr_df, n=6)
    
    # Summary report
    create_summary_report(season_df, corr_df)
    
    # Save correlation data
    output_file = f"correlations_{season_choice.replace('-', '_')}.csv"
    corr_df.to_csv(output_file, index=False)
    print(f"\n✓ Correlation data saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review the scatter plots to see relationships")
    print("  2. Compare this season to other eras")
    print("  3. Build multi-season analysis to track evolution")
    
    return season_df, corr_df


if __name__ == "__main__":
    season_df, corr_df = main()
