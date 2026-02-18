# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NBA team statistics analysis project studying how basketball statistics correlate with winning across different eras (1980-2025). The focus is on correlation analysis, trend tracking, champion influence hypothesis testing, and interactive visualization.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running the Pipeline

Scripts must be run in order — each stage depends on the output of the previous:

```bash
# 1. Collect data (takes ~5-7 min due to NBA API rate limiting at 0.6s/request)
python nba_data_collector.py
# Output: nba_team_stats_1980_2025.csv

# 2. Clean and filter to key metrics
python clean_dataset.py
# Output: nba_team_stats_clean.csv

# 3. Run any analysis or visualization script (standalone after data is ready)
python single_season_analysis.py
python ortg_drtg_evolution.py
python franchise_evolution_analysis.py
python champion_influence_analysis.py
python interactive_regime_timeline.py  # Plotly interactive dashboard
python franchise_evolution_animation.py
```

There is no test framework, linter config, or build system in this project.

## Architecture

**Data flow:** NBA API → collector → raw CSV → cleaner → clean CSV → analysis scripts → visualization scripts

### Layer breakdown

- **Collection:** `nba_data_collector.py` — fetches traditional, advanced, four-factors, and opponent stats via `nba_api`. Rate-limited to respect API constraints.
- **Processing:** `clean_dataset.py`, `check_columns.py`, `check_ranks.py` — standardize column names, filter to ~30 key metrics, validate data.
- **Analysis:** `single_season_analysis.py`, `ortg_drtg_evolution.py`, `franchise_evolution_analysis.py`, `champion_influence_analysis.py` — correlation analysis, trend tracking, hypothesis testing. All read from `nba_team_stats_clean.csv`.
- **Reference data:** `nba_champions_lookup.py` — hardcoded champion records 1996-2024; `toronto_raptors_complete_data.py` — team-specific analysis.
- **Visualization:** `regime_timeline_visualization.py` (Matplotlib static), `interactive_regime_timeline.py` / `integrated_regime_timeline.py` (Plotly interactive), `franchise_evolution_animation.py` (animated).

### Key data columns

- **Win metrics:** `W`, `L`, `WIN_PCT`, `GP`
- **Advanced:** `ORtg`, `DRtg`, `NRtg`, `TS%`, `eFG%`, `Pace`, `PIE`
- **Four factors:** `AST%`, `ORB%`, `DRB%`, `TOV%`
- **3-point evolution:** `3PA`, `3P%`, `FG3M`
- **Opponent stats:** prefixed with `OPP_`
- **Derived:** `SEASON`, `YEAR`, `ERA` (groups by decade: 1980s–2020s)

### Era grouping

The `ERA` column partitions data into decades (1980s, 1990s, 2000s, 2010s, 2020s) for cross-era comparison. Many analysis scripts pivot on this column.

## Notes

- The NBA API (`nba_api`) can be flaky; the collector script includes retry logic and deliberate sleep delays.
- `nba_champions_by_season.csv` is a generated reference file; `nba_champions_lookup.py` is the authoritative hardcoded source for 1996-2024 champions.
- Analysis scripts are designed to run independently once the cleaned CSV exists — no shared module imports between them.
