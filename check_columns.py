"""
Check Available Columns in NBA Dataset
Quick diagnostic to see what columns we have
"""

import pandas as pd

# Load your data
df = pd.read_csv('nba_team_stats_1980_2025.csv')

print("=" * 80)
print("COLUMN NAME CHECK")
print("=" * 80)

# Check for pace-related columns
print("\nPACE-RELATED COLUMNS:")
pace_cols = [col for col in df.columns if 'PACE' in col.upper()]
print(pace_cols if pace_cols else "  ⚠ No PACE columns found")

# Check for eFG% columns
print("\nEFFECTIVE FG% COLUMNS:")
efg_cols = [col for col in df.columns if 'EFG' in col.upper() or 'E_FG' in col.upper()]
print(efg_cols if efg_cols else "  ⚠ No eFG% columns found")

# Check for AST% columns
print("\nASSIST% COLUMNS:")
ast_cols = [col for col in df.columns if 'AST' in col.upper()]
print(ast_cols if ast_cols else "  ⚠ No AST% columns found")

# Check for TOV% columns
print("\nTURNOVER% COLUMNS:")
tov_cols = [col for col in df.columns if 'TOV' in col.upper() or 'TO_' in col.upper()]
print(tov_cols if tov_cols else "  ⚠ No TOV% columns found")

# Show all columns for reference
print("\n" + "=" * 80)
print("ALL COLUMNS IN DATASET")
print("=" * 80)
print(df.columns.tolist())

# Check for Toronto Raptors data
print("\n" + "=" * 80)
print("SAMPLE TORONTO RAPTORS DATA")
print("=" * 80)
raptors = df[df['Team'] == 'Toronto Raptors'].head(1)
if len(raptors) > 0:
    print(raptors.iloc[0])
else:
    print("  ⚠ No Toronto Raptors data found")
