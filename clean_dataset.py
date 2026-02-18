"""
NBA Data Cleanup - Keep Only Key Advanced Stats
Filter dataset to include only the most important metrics for analysis

Author: GY Mehta
Date: February 2026
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def clean_dataset(input_file='nba_team_stats_1980_2025.csv', 
                  output_file='nba_team_stats_clean.csv'):
    """
    Filter dataset to keep only key metrics
    """
    
    print("=" * 80)
    print("NBA DATASET CLEANUP - KEEPING ONLY KEY METRICS")
    print("=" * 80)
    
    # Load data
    try:
        df = pd.read_csv(input_file)
        print(f"\n✓ Loaded: {input_file}")
        print(f"  Original: {len(df)} rows, {len(df.columns)} columns")
    except FileNotFoundError:
        print(f"\nError: Could not find '{input_file}'")
        return None
    
    # Define columns to keep
    keep_columns = [
        # Identifiers
        'Team',
        'SEASON',
        'YEAR',
        'ERA',
        
        # Record
        'GP',
        'W',
        'L',
        'WIN_PCT',
        
        # Overall Team Quality
        'ORtg',      # Offensive Rating
        'DRtg',      # Defensive Rating
        'NRtg',      # Net Rating
        
        # Shooting Efficiency
        'eFG%',      # Effective FG%
        'TS%',       # True Shooting%
        
        # 3-Point Shooting (Evolution)
        '3PA',       # 3-Point Attempts
        '3P%',       # 3-Point Percentage
        'FG3M',      # 3-Pointers Made (for context)
        
        # Playing Style
        'Pace',      # Pace (possessions per 48 min)
        
        # Ball Control & Movement
        'AST%',      # Assist Percentage
        'AST_TO',    # Assist to Turnover Ratio
        'AST_RATIO', # Assists per 100 possessions
        'TOV%',      # Turnover Percentage
        
        # Rebounding Dominance
        'ORB%',      # Offensive Rebound %
        'DRB%',      # Defensive Rebound %
        'REB_PCT',   # Total Rebound %
        
        # Other Advanced
        'FTA_RATE',  # Free Throw Rate
        'PIE',       # Player Impact Estimate
        
        # Basic Stats (for context)
        'PTS',       # Points per game
        'FG%',       # Field Goal %
        'FT%',       # Free Throw %
        
        # Opponent Stats
        'OPP_OPP_PTS',    # Opponent Points
        'OPP_OPP_FG_PCT', # Opponent FG%
        'OPP_OPP_3P_PCT', # Opponent 3P%
    ]
    
    # Check which columns exist
    available_columns = [col for col in keep_columns if col in df.columns]
    missing_columns = [col for col in keep_columns if col not in df.columns]
    
    print("\n" + "=" * 80)
    print("COLUMN AVAILABILITY CHECK")
    print("=" * 80)
    print(f"\n✓ Found {len(available_columns)} of {len(keep_columns)} desired columns")
    
    if missing_columns:
        print(f"\n⚠ Missing {len(missing_columns)} columns:")
        for col in missing_columns:
            print(f"  - {col}")
    
    # Create cleaned dataset
    df_clean = df[available_columns].copy()
    
    # Sort by season and team
    df_clean = df_clean.sort_values(['SEASON', 'Team']).reset_index(drop=True)
    
    # Save
    df_clean.to_csv(output_file, index=False)
    print(f"\n✓ Saved cleaned dataset: {output_file}")
    print(f"  Final: {len(df_clean)} rows, {len(df_clean.columns)} columns")
    
    # Show summary
    print("\n" + "=" * 80)
    print("CLEANED DATASET SUMMARY")
    print("=" * 80)
    
    print(f"\nSeasons: {df_clean['SEASON'].min()} to {df_clean['SEASON'].max()}")
    print(f"Total seasons: {df_clean['SEASON'].nunique()}")
    print(f"Total teams: {df_clean['Team'].nunique()}")
    
    if 'ERA' in df_clean.columns:
        print("\nTeams per era:")
        print(df_clean.groupby('ERA').size())
    
    # Show sample
    print("\n" + "=" * 80)
    print("SAMPLE DATA (First 5 rows)")
    print("=" * 80)
    
    # Show key columns
    sample_cols = ['Team', 'SEASON', 'W', 'L', 'WIN_PCT', 'ORtg', 'DRtg', 
                   'eFG%', 'TS%', '3PA', '3P%', 'Pace']
    sample_cols = [col for col in sample_cols if col in df_clean.columns]
    
    print(df_clean[sample_cols].head().to_string())
    
    # Data quality check
    print("\n" + "=" * 80)
    print("DATA QUALITY CHECK")
    print("=" * 80)
    
    missing_pct = (df_clean.isnull().sum() / len(df_clean) * 100).sort_values(ascending=False)
    
    if missing_pct.max() > 0:
        print("\nColumns with missing data:")
        print(missing_pct[missing_pct > 0])
    else:
        print("\n✓ No missing data - excellent!")
    
    # Show all columns organized by category
    print("\n" + "=" * 80)
    print("ALL COLUMNS IN CLEANED DATASET")
    print("=" * 80)
    
    categories = {
        'Identifiers': ['Team', 'SEASON', 'YEAR', 'ERA'],
        'Record': ['GP', 'W', 'L', 'WIN_PCT'],
        'Overall Quality': ['ORtg', 'DRtg', 'NRtg'],
        'Shooting Efficiency': ['eFG%', 'TS%', 'FG%', 'FT%'],
        '3-Point Evolution': ['3PA', '3P%', 'FG3M'],
        'Playing Style': ['Pace'],
        'Ball Control': ['AST%', 'AST_TO', 'AST_RATIO', 'TOV%'],
        'Rebounding': ['ORB%', 'DRB%', 'REB_PCT'],
        'Other Advanced': ['FTA_RATE', 'PIE'],
        'Basic Stats': ['PTS'],
        'Opponent Stats': ['OPP_OPP_PTS', 'OPP_OPP_FG_PCT', 'OPP_OPP_3P_PCT']
    }
    
    for category, cols in categories.items():
        available = [col for col in cols if col in df_clean.columns]
        if available:
            print(f"\n{category}:")
            print(f"  {', '.join(available)}")
    
    print("\n" + "=" * 80)
    print("✓ CLEANUP COMPLETE!")
    print("=" * 80)
    print(f"\nYou can now use '{output_file}' for your analysis.")
    print("This file contains only the key metrics you specified.")
    
    return df_clean


def main():
    """
    Main execution
    """
    
    input_file = input("Enter input filename (default: nba_team_stats_1980_2025.csv): ").strip()
    if not input_file:
        input_file = 'nba_team_stats_1980_2025.csv'
    
    output_file = input("Enter output filename (default: nba_team_stats_clean.csv): ").strip()
    if not output_file:
        output_file = 'nba_team_stats_clean.csv'
    
    df_clean = clean_dataset(input_file, output_file)
    
    if df_clean is not None:
        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("\n1. Use this cleaned file for all your analysis scripts")
        print("2. Update your scripts to load 'nba_team_stats_clean.csv'")
        print("3. Run correlation analysis on this focused dataset")
        print("4. Create visualizations showing NBA evolution")
        print("\nThe cleaned dataset will make your analysis:")
        print("  ✓ Faster (fewer columns to process)")
        print("  ✓ Clearer (only relevant metrics)")
        print("  ✓ More professional (focused on what matters)")


if __name__ == "__main__":
    main()
