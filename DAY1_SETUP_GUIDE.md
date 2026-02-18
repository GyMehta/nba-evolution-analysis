# NBA Evolution Analysis - Day 1 Setup Guide

## Project: NBA Team Stats Correlation Analysis (1980-2025)
**Objective:** Analyze how different basketball statistics correlate with winning across NBA eras

---

## Day 1: Data Collection & Setup

### Step 1: Environment Setup

1. **Create a project directory:**
```bash
mkdir nba-evolution-analysis
cd nba-evolution-analysis
```

2. **Create a virtual environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Step 2: Data Collection

Run the data collector script:
```bash
python nba_data_collector.py
```

**What this script does:**
- Fetches team stats for seasons 1980-81 through 2024-25
- Collects basic stats, opponent stats, and advanced stats
- Respects Basketball Reference's rate limit (20 requests/minute)
- Takes approximately 3-4 minutes to complete
- Outputs: `nba_team_stats_1980_2025.csv`

**Expected output columns:**
- Basic: TEAM, W, L, PTS, FG, FGA, FG%, 3P, 3PA, 3P%, FT, FTA, FT%, ORB, DRB, TRB, AST, STL, BLK, TOV, PF
- Opponent: Same stats with OPP_ prefix
- Advanced: ORtg, DRtg, NRtg (Net Rating), Pace, TS%, eFG%, TOV%, ORB%, DRB%, FTr, 3PAr
- Calculated: WIN_PCT, ERA (decade grouping)

### Step 3: Initial Data Exploration

After collection, create a quick exploration notebook:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('nba_team_stats_1980_2025.csv')

# Quick checks
print(f"Total records: {len(df)}")
print(f"Date range: {df['SEASON'].min()} to {df['SEASON'].max()}")
print(f"Unique teams: {df['TEAM'].nunique()}")

# Check for missing data
print("\nMissing data by column:")
print(df.isnull().sum().sort_values(ascending=False).head(15))

# Basic stats by era
print("\nTeams per era:")
print(df.groupby('ERA')['TEAM'].count())
```

### Step 4: Key Metrics to Focus On

For your correlation analysis, prioritize these metrics:

**Offensive Efficiency:**
- ORtg (Offensive Rating)
- TS% (True Shooting %)
- eFG% (Effective FG%)
- 3P% and 3PA
- AST% (Assist percentage)
- ORB% (Offensive rebound %)
- TOV% (Turnover %)

**Defensive Efficiency:**
- DRtg (Defensive Rating)
- DRB% (Defensive rebound %)
- STL (Steals)
- BLK (Blocks)
- OPP_PTS (Opponent points)

**Pace & Style:**
- Pace
- FTr (Free throw rate)
- 3PAr (3-point attempt rate)

### Step 5: Data Quality Checks

Before moving to Day 2, verify:

1. **No duplicate rows:**
```python
duplicates = df[df.duplicated(['TEAM', 'SEASON'])]
print(f"Duplicate rows: {len(duplicates)}")
```

2. **Win% calculation is correct:**
```python
# Verify win percentage
df['WIN_PCT_CHECK'] = df['W'] / (df['W'] + df['L'])
print(f"Win% accuracy: {(df['WIN_PCT'] == df['WIN_PCT_CHECK']).sum() / len(df) * 100:.1f}%")
```

3. **Data completeness by era:**
```python
print("\nData completeness by era:")
print(df.groupby('ERA')['ORtg'].count())
```

### Troubleshooting

**Issue: Rate limit errors**
- The script already includes 3.5-second delays
- If you still get errors, increase the sleep time in the script

**Issue: Missing columns**
- Some advanced stats may not be available for very old seasons
- This is expected and will be handled in Day 2 cleaning

**Issue: SSL/connection errors**
- Try running the script again
- Basketball Reference can occasionally have connection issues

### Expected Completion Time
- Setup: 10-15 minutes
- Data collection: 3-5 minutes
- Initial exploration: 10 minutes
- **Total: ~30 minutes**

---

## Next Steps (Day 2)

Tomorrow you'll:
1. Clean and standardize team names (handle relocations like Seattle → OKC)
2. Handle missing values strategically
3. Create derived metrics if needed
4. Validate data quality
5. Prepare the dataset for correlation analysis

---

## Pro Tips for Day 1

1. **Document as you go:** Keep notes on any data issues you spot
2. **Save checkpoints:** Back up your raw CSV immediately
3. **Ask Claude:** If you hit any snags with the script or data, ping me!
4. **Quick win:** Once data is collected, create one simple chart (e.g., average 3PA by era) - helps you feel momentum

Good luck! 🏀📊
