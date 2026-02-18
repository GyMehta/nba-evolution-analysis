"""
Diagnostic: Check Rating Ranks Calculation
Verify that ranks are being calculated correctly
"""

import pandas as pd

# Load data
df = pd.read_csv('nba_team_stats_1980_2025.csv')
team_name = 'Toronto Raptors'

team_df = df[df['Team'] == team_name].copy()
team_df = team_df.sort_values('SEASON')

print("=" * 100)
print(f"{team_name.upper()} - RATING RANKS BY SEASON")
print("=" * 100)

print(f"\n{'Season':<12} {'ORtg':<10} {'ORtg Rank':<12} {'DRtg':<10} {'DRtg Rank':<12} {'NRtg':<10} {'NRtg Rank':<12}")
print("-" * 100)

for _, team_row in team_df.iterrows():
    season = team_row['SEASON']
    
    # Get all teams for this season
    season_league = df[df['SEASON'] == season]
    
    if 'ORtg' in team_row and pd.notna(team_row['ORtg']):
        # ORtg rank (higher is better)
        ortg_val = team_row['ORtg']
        ortg_rank = (season_league['ORtg'] > ortg_val).sum() + 1
        ortg_norm = 1 - (ortg_rank - 1) / 30
        
        # DRtg rank (lower is better)
        drtg_val = team_row['DRtg']
        drtg_rank = (season_league['DRtg'] < drtg_val).sum() + 1
        drtg_norm = 1 - (drtg_rank - 1) / 30
        
        # NRtg rank (higher is better)
        if 'NRtg' in team_row and pd.notna(team_row['NRtg']):
            nrtg_val = team_row['NRtg']
            nrtg_rank = (season_league['NRtg'] > nrtg_val).sum() + 1
            nrtg_norm = 1 - (nrtg_rank - 1) / 30
        else:
            nrtg_rank = 'N/A'
            nrtg_norm = 'N/A'
            nrtg_val = 'N/A'
        
        print(f"{season:<12} {ortg_val:<10.1f} {ortg_rank:<4} ({ortg_norm:.3f})  {drtg_val:<10.1f} "
              f"{drtg_rank:<4} ({drtg_norm:.3f})  {str(nrtg_val):<10} {str(nrtg_rank):<4} ({str(nrtg_norm):<10})")

print("\n" + "=" * 100)
print("INTERPRETATION:")
print("=" * 100)
print("Rank 1 = Best in league")
print("Rank 30 = Worst in league")
print("Normalized value: 1.0 = Best (rank 1), 0.0 = Worst (rank 30)")
print("\nIf all normalized values show 1.0, there's a bug in the calculation")

# Check how many teams per season
print("\n" + "=" * 100)
print("TEAMS PER SEASON (for rank calculation)")
print("=" * 100)

for season in team_df['SEASON'].unique()[:5]:  # Show first 5 seasons
    season_teams = df[df['SEASON'] == season]
    print(f"{season}: {len(season_teams)} teams")
