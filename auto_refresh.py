"""
NBA 2025-26 Auto-Refresh
Runs the complete pipeline end-to-end in ~60 seconds.

  ESPN     → team stats  (W/L/NRtg/ORtg/DRtg)  — fast, no rate limiting
  NBA API  → player stats (Base + Advanced)      — ~20 s when reachable
  pipeline → ratings → profiles → similarity → HTML report

Usage:
  python auto_refresh.py            # refresh + regenerate report
  python auto_refresh.py --commit   # also git-commit the result

The script is safe to run any time; it always replaces 2025-26 rows only.
"""

import subprocess, sys, os, argparse, datetime

PYTHON = sys.executable
FLAGS  = ['-X', 'utf8']
REPO   = os.path.dirname(os.path.abspath(__file__))


def run(script, extra_args=None, allow_fail=False):
    """Run a pipeline script and return True on success."""
    cmd = [PYTHON] + FLAGS + [os.path.join(REPO, script)] + (extra_args or [])
    result = subprocess.run(cmd, cwd=REPO)
    ok = (result.returncode == 0)
    if not ok and not allow_fail:
        print(f'\n✗  {script} failed (exit {result.returncode}) — stopping.')
        sys.exit(1)
    return ok


def step(n, total, label):
    print(f'\n[{n}/{total}] {label}')
    print('─' * 60)


def main():
    parser = argparse.ArgumentParser(description='NBA 2025-26 auto-refresh pipeline')
    parser.add_argument('--commit', action='store_true',
                        help='Git-commit the refreshed data after pipeline completes')
    parser.add_argument('--skip-players', action='store_true',
                        help='Skip player stats refresh (use cached player_stats_by_season.csv)')
    args = parser.parse_args()

    STEPS = 5 if not args.skip_players else 4

    print('━' * 60)
    print('NBA 2025-26  AUTO-REFRESH')
    print(f'Started: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('━' * 60)

    # ── 1. Team stats via ESPN ──────────────────────────────────────────────
    step(1, STEPS, 'Team stats — ESPN (W/L/NRtg/ORtg/DRtg)')
    run('fetch_espn_team_stats.py')

    # ── 2. Player stats via NBA API ─────────────────────────────────────────
    if not args.skip_players:
        step(2, STEPS, 'Player stats — NBA API (Base + Advanced)')
        # --player-only skips the slow (and currently blocked) team-stats fetch
        run('refresh_current_season.py', extra_args=['--player-only'], allow_fail=True)
        # allow_fail=True: if NBA API is down, we continue with cached player data
    else:
        print(f'\n[–/{STEPS}] Player stats — skipped (--skip-players)')

    # ── 3. Player rating engine ─────────────────────────────────────────────
    idx = 3 if not args.skip_players else 2
    step(idx, STEPS, 'Compute player ratings + tier assignments')
    run('player_rating_engine.py')

    # ── 4. Team profile builder ─────────────────────────────────────────────
    idx += 1
    step(idx, STEPS, 'Build team profiles (n_superstars, n_elite_stars, depth_score …)')
    run('team_profile_builder.py')

    # ── 5. Similarity + HTML report ─────────────────────────────────────────
    idx += 1
    step(idx, STEPS, 'Compute 3-year similarity + generate HTML report')
    run('generate_multiyear_similarity.py')
    run('generate_multiyear_report.py')

    # ── Optional: git commit ────────────────────────────────────────────────
    if args.commit:
        import subprocess as _sp
        date_str = datetime.datetime.now().strftime('%Y-%m-%d')
        msg = (f'Auto-refresh {date_str}: update 2025-26 stats via ESPN\n\n'
               f'Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>')
        _sp.run(['git', 'add',
                 'nba_team_stats_clean.csv', 'player_stats_by_season.csv',
                 'player_ratings.csv', 'team_profiles.csv',
                 'nba_multiyear_similarity.csv', 'nba_multiyear_report.html'],
                cwd=REPO)
        _sp.run(['git', 'commit', '-m', msg], cwd=REPO)
        print('\n✓ Changes committed.')

    print('\n' + '━' * 60)
    print('✓  Refresh complete.')
    print(f'   Finished: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('   Open nba_multiyear_report.html to view the updated report.')
    if not args.commit:
        print('   Run with --commit to also commit the result to git.')
    print('━' * 60)


if __name__ == '__main__':
    main()
