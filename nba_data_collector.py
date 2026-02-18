"""
NBA Team Stats Data Collector - Using Official NBA API
Collects team-level stats from 1980-2025 using nba_api

Author: GY Mehta
Date: February 2026
"""

import pandas as pd
import time
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams
import warnings
warnings.filterwarnings('ignore')

# Define the range of seasons
START_YEAR = 1980
END_YEAR = 2024  # NBA API uses the starting year of the season

def get_season_string(year):
    """Convert year to NBA season format (e.g., 2023 -> '2023-24')"""
    next_year = str(year + 1)[-2:]
    return f"{year}-{next_year}"


def collect_team_stats_for_season(season):
    """
    Collect team stats for a single season using NBA API
    
    Parameters:
    -----------
    season : str
        Season in format 'YYYY-YY' (e.g., '2023-24')
    
    Returns:
    --------
    pandas.DataFrame or None
        Team stats for the season
    """
    
    try:
        # Get Traditional stats (basic box score stats)
        traditional = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense='Base',
            per_mode_detailed='PerGame',
            season_type_all_star='Regular Season'
        )
        traditional_df = traditional.get_data_frames()[0]
        
        # Get Advanced stats (ORtg, DRtg, Pace, etc.)
        time.sleep(0.6)  # Rate limiting
        advanced = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense='Advanced',
            per_mode_detailed='PerGame',
            season_type_all_star='Regular Season'
        )
        advanced_df = advanced.get_data_frames()[0]
        
        # Get Four Factors (eFG%, TOV%, ORB%, FT Rate)
        time.sleep(0.6)
        four_factors = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense='Four Factors',
            per_mode_detailed='PerGame',
            season_type_all_star='Regular Season'
        )
        four_factors_df = four_factors.get_data_frames()[0]
        
        # Get Opponent stats
        time.sleep(0.6)
        opponent = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense='Opponent',
            per_mode_detailed='PerGame',
            season_type_all_star='Regular Season'
        )
        opponent_df = opponent.get_data_frames()[0]
        
        # Merge all dataframes
        # Start with traditional as base
        merged = traditional_df.copy()
        
        # Add advanced stats
        advanced_cols = ['TEAM_ID', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 
                        'PIE', 'AST_PCT', 'AST_TO', 'AST_RATIO', 'OREB_PCT', 'DREB_PCT',
                        'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT']
        advanced_subset = advanced_df[[col for col in advanced_cols if col in advanced_df.columns]]
        merged = merged.merge(advanced_subset, on='TEAM_ID', how='left', suffixes=('', '_ADV'))
        
        # Add four factors
        ff_cols = ['TEAM_ID', 'EFG_PCT', 'FTA_RATE', 'TM_TOV_PCT', 'OREB_PCT']
        ff_subset = four_factors_df[[col for col in ff_cols if col in four_factors_df.columns]]
        merged = merged.merge(ff_subset, on='TEAM_ID', how='left', suffixes=('', '_FF'))
        
        # Add opponent stats with OPP_ prefix
        opponent_df = opponent_df.add_prefix('OPP_')
        opponent_df = opponent_df.rename(columns={'OPP_TEAM_ID': 'TEAM_ID'})
        merged = merged.merge(opponent_df, on='TEAM_ID', how='left')
        
        # Add season info
        merged['SEASON'] = season
        
        return merged
        
    except Exception as e:
        print(f"  Error: {str(e)}")
        return None


def collect_all_seasons(start_year=START_YEAR, end_year=END_YEAR):
    """
    Collect data for all seasons
    """
    
    all_data = []
    
    for year in range(start_year, end_year + 1):
        season = get_season_string(year)
        print(f"Fetching {season} season...")
        
        df = collect_team_stats_for_season(season)
        
        if df is not None and len(df) > 0:
            all_data.append(df)
            print(f"  ✓ Collected {len(df)} teams")
        else:
            print(f"  ✗ Failed to collect data")
        
        # Rate limiting - be respectful to NBA API
        time.sleep(1)
    
    if len(all_data) > 0:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()


def clean_and_prepare_data(df):
    """
    Clean and prepare the dataset
    """
    
    if df.empty:
        return df
    
    # Calculate Win%
    if 'W' in df.columns and 'L' in df.columns:
        df['WIN_PCT'] = df['W'] / (df['W'] + df['L'])
    elif 'W_PCT' in df.columns:
        df['WIN_PCT'] = df['W_PCT']
    
    # Add year column
    if 'SEASON' in df.columns:
        df['YEAR'] = df['SEASON'].str[:4].astype(int) + 1  # Convert '2023-24' to 2024
    
    # Add era categories
    def assign_era(year):
        if year <= 1990:
            return '1980s'
        elif year <= 2000:
            return '1990s'
        elif year <= 2010:
            return '2000s'
        elif year <= 2020:
            return '2010s'
        else:
            return '2020s'
    
    if 'YEAR' in df.columns:
        df['ERA'] = df['YEAR'].apply(assign_era)
    
    # Rename columns for clarity
    column_mapping = {
        'TEAM_NAME': 'Team',
        'OFF_RATING': 'ORtg',
        'DEF_RATING': 'DRtg',
        'NET_RATING': 'NRtg',
        'PACE': 'Pace',
        'FG_PCT': 'FG%',
        'FG3_PCT': '3P%',
        'FT_PCT': 'FT%',
        'FG3A': '3PA',
        'FTA': 'FTA',
        'AST_PCT': 'AST%',
        'OREB_PCT': 'ORB%',
        'DREB_PCT': 'DRB%',
        'TM_TOV_PCT': 'TOV%',
        'EFG_PCT': 'eFG%',
        'TS_PCT': 'TS%'
    }
    
    df = df.rename(columns=column_mapping)
    
    return df


def main():
    """
    Main execution
    """
    
    print("=" * 70)
    print("NBA TEAM STATS DATA COLLECTION - Official NBA API")
    print("=" * 70)
    print(f"Collecting data from {START_YEAR}-{START_YEAR+1} to {END_YEAR}-{END_YEAR+1}")
    print("Estimated time: 5-7 minutes (due to API rate limits)")
    print("=" * 70)
    print()
    
    # Collect data
    raw_data = collect_all_seasons()
    
    if raw_data.empty:
        print("\n❌ ERROR: No data collected!")
        print("If you're running from a cloud server, NBA API may block datacenter IPs.")
        print("Try running from your local machine.")
        return None
    
    print(f"\n✓ Raw data collected: {len(raw_data)} rows")
    
    # Clean data
    clean_df = clean_and_prepare_data(raw_data)
    
    # Save to CSV
    output_file = 'nba_team_stats_1980_2025.csv'
    clean_df.to_csv(output_file, index=False)
    print(f"✓ Data saved to: {output_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"Total rows: {len(clean_df)}")
    print(f"Total columns: {len(clean_df.columns)}")
    print(f"Seasons: {clean_df['SEASON'].nunique()}")
    print(f"Date range: {clean_df['SEASON'].min()} to {clean_df['SEASON'].max()}")
    
    print("\n" + "=" * 70)
    print("KEY METRICS AVAILABLE")
    print("=" * 70)
    
    key_metrics = ['Team', 'SEASON', 'W', 'L', 'WIN_PCT', 'ORtg', 'DRtg', 'Pace',
                   'FG%', '3P%', 'FT%', '3PA', 'eFG%', 'TS%', 'AST%', 'ORB%', 
                   'DRB%', 'TOV%', 'PTS']
    
    available = [col for col in key_metrics if col in clean_df.columns]
    print("Available key columns:", ', '.join(available))
    
    print("\n" + "=" * 70)
    print("SAMPLE DATA")
    print("=" * 70)
    sample_cols = [col for col in ['Team', 'SEASON', 'W', 'L', 'WIN_PCT', 'ORtg', 'DRtg', 'Pace'] 
                   if col in clean_df.columns]
    if sample_cols:
        print(clean_df[sample_cols].head(10))
    
    print("\n" + "=" * 70)
    print("ERA BREAKDOWN")
    print("=" * 70)
    if 'ERA' in clean_df.columns:
        print(clean_df.groupby('ERA').size())
    
    print("\n" + "=" * 70)
    print("✓ SUCCESS! Ready for Day 2 (Analysis)")
    print("=" * 70)
    
    return clean_df


if __name__ == "__main__":
    df = main()
