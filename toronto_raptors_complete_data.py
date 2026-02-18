"""
Toronto Raptors Historical Data (1995-2024)
Complete dataset with Coaches, GMs, and Playoff Results

Compiled from: Basketball-Reference, Wikipedia, NBA.com
Author: GY Mehta
Date: February 2026
"""

import pandas as pd

# Toronto Raptors Complete Historical Data
RAPTORS_DATA = [
    # Format: Season, Year, Head_Coach, General_Manager, Playoff_Result, Playoff_Depth
    
    # Isiah Thomas Era (GM)
    {'SEASON': '1995-96', 'YEAR': 1996, 'Head_Coach': 'Brendan Malone', 'General_Manager': 'Isiah Thomas', 'Playoff_Result': 'Did Not Qualify', 'Playoff_Depth': 0},
    {'SEASON': '1996-97', 'YEAR': 1997, 'Head_Coach': 'Darrell Walker', 'General_Manager': 'Isiah Thomas', 'Playoff_Result': 'Did Not Qualify', 'Playoff_Depth': 0},
    {'SEASON': '1997-98', 'YEAR': 1998, 'Head_Coach': 'Darrell Walker / Butch Carter', 'General_Manager': 'Isiah Thomas', 'Playoff_Result': 'Did Not Qualify', 'Playoff_Depth': 0},
    {'SEASON': '1998-99', 'YEAR': 1999, 'Head_Coach': 'Butch Carter', 'General_Manager': 'Glen Grunwald', 'Playoff_Result': 'Did Not Qualify', 'Playoff_Depth': 0},
    
    # Vince Carter / Grunwald Era
    {'SEASON': '1999-00', 'YEAR': 2000, 'Head_Coach': 'Butch Carter', 'General_Manager': 'Glen Grunwald', 'Playoff_Result': 'First Round', 'Playoff_Depth': 1},
    {'SEASON': '2000-01', 'YEAR': 2001, 'Head_Coach': 'Lenny Wilkens', 'General_Manager': 'Glen Grunwald', 'Playoff_Result': 'Second Round', 'Playoff_Depth': 2},
    {'SEASON': '2001-02', 'YEAR': 2002, 'Head_Coach': 'Lenny Wilkens', 'General_Manager': 'Glen Grunwald', 'Playoff_Result': 'First Round', 'Playoff_Depth': 1},
    {'SEASON': '2002-03', 'YEAR': 2003, 'Head_Coach': 'Lenny Wilkens', 'General_Manager': 'Glen Grunwald', 'Playoff_Result': 'Did Not Qualify', 'Playoff_Depth': 0},
    {'SEASON': '2003-04', 'YEAR': 2004, 'Head_Coach': 'Kevin O\'Neill', 'General_Manager': 'Glen Grunwald', 'Playoff_Result': 'Did Not Qualify', 'Playoff_Depth': 0},
    
    # Rob Babcock Era (Dark Ages)
    {'SEASON': '2004-05', 'YEAR': 2005, 'Head_Coach': 'Sam Mitchell', 'General_Manager': 'Rob Babcock', 'Playoff_Result': 'Did Not Qualify', 'Playoff_Depth': 0},
    {'SEASON': '2005-06', 'YEAR': 2006, 'Head_Coach': 'Sam Mitchell', 'General_Manager': 'Rob Babcock', 'Playoff_Result': 'Did Not Qualify', 'Playoff_Depth': 0},
    
    # Bryan Colangelo Era
    {'SEASON': '2006-07', 'YEAR': 2007, 'Head_Coach': 'Sam Mitchell', 'General_Manager': 'Bryan Colangelo', 'Playoff_Result': 'First Round', 'Playoff_Depth': 1},
    {'SEASON': '2007-08', 'YEAR': 2008, 'Head_Coach': 'Sam Mitchell', 'General_Manager': 'Bryan Colangelo', 'Playoff_Result': 'First Round', 'Playoff_Depth': 1},
    {'SEASON': '2008-09', 'YEAR': 2009, 'Head_Coach': 'Sam Mitchell / Jay Triano', 'General_Manager': 'Bryan Colangelo', 'Playoff_Result': 'Did Not Qualify', 'Playoff_Depth': 0},
    {'SEASON': '2009-10', 'YEAR': 2010, 'Head_Coach': 'Jay Triano', 'General_Manager': 'Bryan Colangelo', 'Playoff_Result': 'Did Not Qualify', 'Playoff_Depth': 0},
    {'SEASON': '2010-11', 'YEAR': 2011, 'Head_Coach': 'Jay Triano', 'General_Manager': 'Bryan Colangelo', 'Playoff_Result': 'Did Not Qualify', 'Playoff_Depth': 0},
    
    # Dwane Casey Era
    {'SEASON': '2011-12', 'YEAR': 2012, 'Head_Coach': 'Dwane Casey', 'General_Manager': 'Bryan Colangelo', 'Playoff_Result': 'Did Not Qualify', 'Playoff_Depth': 0},
    {'SEASON': '2012-13', 'YEAR': 2013, 'Head_Coach': 'Dwane Casey', 'General_Manager': 'Bryan Colangelo', 'Playoff_Result': 'Did Not Qualify', 'Playoff_Depth': 0},
    
    # Masai Ujiri Era - Lowry/DeRozan
    {'SEASON': '2013-14', 'YEAR': 2014, 'Head_Coach': 'Dwane Casey', 'General_Manager': 'Masai Ujiri', 'Playoff_Result': 'First Round', 'Playoff_Depth': 1},
    {'SEASON': '2014-15', 'YEAR': 2015, 'Head_Coach': 'Dwane Casey', 'General_Manager': 'Masai Ujiri', 'Playoff_Result': 'First Round', 'Playoff_Depth': 1},
    {'SEASON': '2015-16', 'YEAR': 2016, 'Head_Coach': 'Dwane Casey', 'General_Manager': 'Masai Ujiri', 'Playoff_Result': 'Conference Finals', 'Playoff_Depth': 3},
    {'SEASON': '2016-17', 'YEAR': 2017, 'Head_Coach': 'Dwane Casey', 'General_Manager': 'Masai Ujiri', 'Playoff_Result': 'Second Round', 'Playoff_Depth': 2},
    {'SEASON': '2017-18', 'YEAR': 2018, 'Head_Coach': 'Dwane Casey', 'General_Manager': 'Masai Ujiri', 'Playoff_Result': 'Second Round', 'Playoff_Depth': 2},
    
    # Championship Era - Nick Nurse
    {'SEASON': '2018-19', 'YEAR': 2019, 'Head_Coach': 'Nick Nurse', 'General_Manager': 'Masai Ujiri', 'Playoff_Result': 'NBA Champions', 'Playoff_Depth': 5},
    {'SEASON': '2019-20', 'YEAR': 2020, 'Head_Coach': 'Nick Nurse', 'General_Manager': 'Masai Ujiri', 'Playoff_Result': 'Second Round', 'Playoff_Depth': 2},
    {'SEASON': '2020-21', 'YEAR': 2021, 'Head_Coach': 'Nick Nurse', 'General_Manager': 'Masai Ujiri', 'Playoff_Result': 'Did Not Qualify', 'Playoff_Depth': 0},
    {'SEASON': '2021-22', 'YEAR': 2022, 'Head_Coach': 'Nick Nurse', 'General_Manager': 'Bobby Webster', 'Playoff_Result': 'First Round', 'Playoff_Depth': 1},
    {'SEASON': '2022-23', 'YEAR': 2023, 'Head_Coach': 'Nick Nurse', 'General_Manager': 'Bobby Webster', 'Playoff_Result': 'Did Not Qualify', 'Playoff_Depth': 0},
    
    # Darko Rajaković Era
    {'SEASON': '2023-24', 'YEAR': 2024, 'Head_Coach': 'Darko Rajaković', 'General_Manager': 'Bobby Webster', 'Playoff_Result': 'First Round', 'Playoff_Depth': 1},
]

# Define franchise eras
RAPTORS_ERAS = [
    {'name': 'Expansion (Isiah Thomas)', 'start': 1995, 'end': 1998, 'color': '#C8C8C8', 'description': 'Inaugural seasons, building foundation'},
    {'name': 'Vince Carter Era', 'start': 1998, 'end': 2004, 'color': '#CE1141', 'description': 'VC highlights, first playoffs'},
    {'name': 'Rebuild (Babcock/Early Colangelo)', 'start': 2004, 'end': 2012, 'color': '#A5ACAF', 'description': 'Dark ages, Bosh departure'},
    {'name': 'Lowry/DeRozan (Casey)', 'start': 2012, 'end': 2018, 'color': '#000000', 'description': 'Consistent playoffs, division titles'},
    {'name': 'Championship (Kawhi)', 'start': 2018, 'end': 2019, 'color': '#FFD700', 'description': 'NBA Champions!'},
    {'name': 'Post-Kawhi (Nurse/Rajaković)', 'start': 2019, 'end': 2024, 'color': '#CE1141', 'description': 'Transition, rebuilding'},
]

def create_raptors_dataset():
    """Create the complete Raptors dataset"""
    df = pd.DataFrame(RAPTORS_DATA)
    return df

def save_raptors_data():
    """Save Raptors data to CSV"""
    df = create_raptors_dataset()
    
    filename = 'toronto_raptors_historical_data.csv'
    df.to_csv(filename, index=False)
    
    print("=" * 80)
    print("TORONTO RAPTORS HISTORICAL DATA")
    print("=" * 80)
    print(f"\n✓ Created: {filename}")
    print(f"  Seasons: {len(df)}")
    print(f"  Range: {df['SEASON'].min()} to {df['SEASON'].max()}")
    print(f"  Unique Coaches: {df['Head_Coach'].nunique()}")
    print(f"  Unique GMs: {df['General_Manager'].nunique()}")
    
    print("\n" + "=" * 80)
    print("FRANCHISE ERAS")
    print("=" * 80)
    for era in RAPTORS_ERAS:
        print(f"\n{era['name']} ({era['start']}-{era['end']})")
        print(f"  {era['description']}")
    
    print("\n" + "=" * 80)
    print("COMPLETE DATASET")
    print("=" * 80)
    print(df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("COACHING CHANGES")
    print("=" * 80)
    coaches = df.groupby('Head_Coach').size().sort_values(ascending=False)
    print(coaches)
    
    print("\n" + "=" * 80)
    print("GM CHANGES")
    print("=" * 80)
    gms = df.groupby('General_Manager').size().sort_values(ascending=False)
    print(gms)
    
    print("\n" + "=" * 80)
    print("PLAYOFF SUCCESS")
    print("=" * 80)
    playoff_summary = df.groupby('Playoff_Result').size().sort_values(ascending=False)
    print(playoff_summary)
    
    return df

if __name__ == "__main__":
    df = save_raptors_data()
