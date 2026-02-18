"""
NBA Franchise Evolution Animation
Creates a video showing season-by-season progression

Author: GY Mehta
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")

# Playoff colors and heights
PLAYOFF_COLORS = {
    0: '#E8E8E8',
    1: '#FFA500',
    2: '#FF6B6B',
    3: '#9B59B6',
    4: '#3498DB',
    5: '#FFD700',
}

PLAYOFF_HEIGHTS = {
    0: 0.05,
    1: 0.20,
    2: 0.35,
    3: 0.50,
    4: 0.70,
    5: 0.95,
}

PLAYOFF_LABELS = {
    0: 'No Playoffs',
    1: 'First Round',
    2: 'Second Round',
    3: 'Conf Finals',
    4: 'Finals',
    5: 'Champions'
}

# Define franchise eras (can be customized per team)
FRANCHISE_ERAS = {
    'Toronto Raptors': [
        {'name': 'Expansion', 'start': 1995, 'end': 1998, 'color': '#C8C8C8'},
        {'name': 'Vince Carter', 'start': 1998, 'end': 2004, 'color': '#CE1141'},
        {'name': 'Rebuild', 'start': 2004, 'end': 2012, 'color': '#A5ACAF'},
        {'name': 'Lowry/DeRozan', 'start': 2012, 'end': 2018, 'color': '#000000'},
        {'name': 'Championship', 'start': 2018, 'end': 2019, 'color': '#FFD700'},
        {'name': 'Post-Kawhi', 'start': 2019, 'end': 2025, 'color': '#CE1141'},
    ]
}


def create_franchise_animation(team_df, df, team_name='Toronto Raptors', fps=2):
    """
    Create animated visualization showing franchise evolution season by season
    
    Parameters:
    -----------
    team_df : DataFrame
        Team data with all seasons
    df : DataFrame
        Full league dataset (needed for rank calculations)
    team_name : str
        Name of the team
    fps : int
        Frames per second (lower = slower, more time to read each season)
    """
    
    # Check data
    if 'Head_Coach' not in team_df.columns or 'General_Manager' not in team_df.columns:
        print("⚠ Missing coach or GM data.")
        return None
    
    # Fill NaN
    team_df['Head_Coach'] = team_df['Head_Coach'].fillna('Unknown')
    team_df['General_Manager'] = team_df['General_Manager'].fillna('Unknown')
    
    # Determine win% column
    win_pct_col = 'WIN_PCT' if 'WIN_PCT' in team_df.columns else 'W_PCT'
    
    # Pre-calculate ranks for all team seasons (need full df for comparison)
    print("Pre-calculating league ranks...")
    team_df['ORtg_Rank_Norm'] = None
    team_df['DRtg_Rank_Norm'] = None
    team_df['NRtg_Rank_Norm'] = None
    
    for idx, team_row in team_df.iterrows():
        season = team_row['SEASON']
        season_league = df[df['SEASON'] == season]  # Get ALL teams for this season
        
        if pd.notna(team_row['ORtg']) and len(season_league) > 0:
            # ORtg rank (higher is better)
            ortg_rank = (season_league['ORtg'] > team_row['ORtg']).sum() + 1
            team_df.at[idx, 'ORtg_Rank_Norm'] = 1 - (ortg_rank - 1) / 30
            
            # DRtg rank (lower is better)
            drtg_rank = (season_league['DRtg'] < team_row['DRtg']).sum() + 1
            team_df.at[idx, 'DRtg_Rank_Norm'] = 1 - (drtg_rank - 1) / 30
            
            # NRtg rank (higher is better)
            if 'NRtg' in team_row and pd.notna(team_row['NRtg']):
                nrtg_rank = (season_league['NRtg'] > team_row['NRtg']).sum() + 1
                team_df.at[idx, 'NRtg_Rank_Norm'] = 1 - (nrtg_rank - 1) / 30
    
    print("✓ Ranks calculated")
    
    seasons = team_df['SEASON'].tolist()
    total_seasons = len(seasons)
    
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.subplots_adjust(bottom=0.10)  # Reserve bottom margin for the figure legend
    
    # Initialize plot elements
    def init():
        ax1.clear()
        ax2.clear()
        return []
    
    def animate(frame):
        """
        Animate one frame (one season)
        """
        ax1.clear()
        ax2.clear()
        
        # Get data up to current frame
        current_data = team_df.iloc[:frame+1]
        current_seasons = seasons[:frame+1]
        x_positions = np.arange(len(current_seasons))
        
        # --- MAIN CHART: WIN% AND PLAYOFF SUCCESS ---
        
        # Add era background shading first (lowest z-order)
        if team_name in FRANCHISE_ERAS:
            for era in FRANCHISE_ERAS[team_name]:
                # Find start and end indices for this era in current_seasons
                era_start_year = era['start']
                era_end_year = era['end']
                
                # Find which seasons fall in this era
                for i, season in enumerate(current_seasons):
                    season_year = int(season.split('-')[0]) + 1  # Convert '2018-19' to 2019
                    
                    if era_start_year <= season_year <= era_end_year:
                        # Found start of era in current timeline
                        era_start_idx = i
                        
                        # Find end of era in current timeline
                        era_end_idx = i
                        for j in range(i, len(current_seasons)):
                            check_year = int(current_seasons[j].split('-')[0]) + 1
                            if check_year <= era_end_year:
                                era_end_idx = j
                            else:
                                break
                        
                        # Draw era shading
                        ax1.axvspan(era_start_idx - 0.5, era_end_idx + 0.5, 
                                   alpha=0.15, color=era['color'], zorder=0)
                        
                        # Add era label at top (only if era is visible)
                        if era_end_idx > era_start_idx or i == 0:
                            mid_point = (era_start_idx + era_end_idx) / 2
                            ax1.text(mid_point, 0.98, era['name'], 
                                   ha='center', va='top', fontsize=8,
                                   color=era['color'], fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', 
                                           facecolor='white', alpha=0.7,
                                           edgecolor=era['color'], linewidth=1.5),
                                   transform=ax1.get_xaxis_transform())
                        break
        
        # Plot playoff bars
        if 'Playoff_Depth' in current_data.columns:
            for i, depth in enumerate(current_data['Playoff_Depth']):
                if pd.notna(depth):
                    depth = int(depth)
                    color = PLAYOFF_COLORS.get(depth, '#E8E8E8')
                    height = PLAYOFF_HEIGHTS.get(depth, 0.05)
                    
                    ax1.bar(i, height, width=0.9, bottom=0, 
                           color=color, alpha=0.6, zorder=1,
                           edgecolor='white', linewidth=1)
        
        # Plot win% line (primary y-axis)
        win_pcts = current_data[win_pct_col].values
        ax1.plot(x_positions, win_pcts, 'o-', linewidth=3, markersize=10,
                color='#CE1141', zorder=3, label='Win %',
                markeredgecolor='white', markeredgewidth=2)
        
        # Add ORtg and DRtg ranks on secondary y-axis (subtle, just for context)
        if 'ORtg_Rank_Norm' in current_data.columns and 'DRtg_Rank_Norm' in current_data.columns:
            ax1_twin = ax1.twinx()
            
            # Use pre-calculated normalized ranks
            ortg_ranks = current_data['ORtg_Rank_Norm'].values
            drtg_ranks = current_data['DRtg_Rank_Norm'].values
            
            # Plot very subtle lines (thin and transparent - context only, not main story)
            ax1_twin.plot(x_positions, ortg_ranks, '^-', linewidth=1.2, markersize=4,
                         color='#2ECC71', alpha=0.4, label='Offense', zorder=1.8)
            ax1_twin.plot(x_positions, drtg_ranks, 'v-', linewidth=1.2, markersize=4,
                         color='#E74C3C', alpha=0.4, label='Defense', zorder=1.8)
            
            # Net rating even more subtle
            if 'NRtg_Rank_Norm' in current_data.columns:
                nrtg_ranks = current_data['NRtg_Rank_Norm'].values
                ax1_twin.plot(x_positions, nrtg_ranks, '-', linewidth=0.6,
                             color='#999999', alpha=0.2, zorder=1.5)
            
            ax1_twin.set_ylim(-0.05, 1.05)
            ax1_twin.set_ylabel('League Rank', fontsize=9, color='#888')
            ax1_twin.tick_params(axis='y', labelcolor='#888', labelsize=8)
            ax1_twin.legend(loc='lower right', fontsize=8, framealpha=0.7)
            ax1_twin.grid(False)
        
        # Add .500 line
        ax1.axhline(y=0.500, color='gray', linestyle='--', linewidth=1.5,
                   alpha=0.5, zorder=2)
        
        # Formatting
        ax1.set_xlim(-0.5, total_seasons - 0.5)  # Keep x-axis constant
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(current_seasons, rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('Win %', fontsize=13, fontweight='bold')
        ax1.set_title(f'{team_name} - Franchise Evolution', 
                     fontsize=16, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, axis='y', zorder=0)
        
        # Add current season highlight
        current_season = current_seasons[-1]
        current_win_pct = win_pcts[-1]
        current_coach = current_data.iloc[-1]['Head_Coach']
        current_gm = current_data.iloc[-1]['General_Manager']
        
        # Highlight current season
        ax1.plot(frame, current_win_pct, 'o', markersize=18,
                color='yellow', zorder=4, markeredgecolor='red', 
                markeredgewidth=3)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='#CE1141', linewidth=3, 
                      marker='o', markersize=10, label='Win % (Left Axis)', markeredgecolor='white', markeredgewidth=2),
            plt.Line2D([0], [0], color='#9B59B6', linewidth=3,
                      marker='s', markersize=8, label='Net Rating Rank (Right Axis)', markeredgecolor='white', markeredgewidth=1.5),
            plt.Line2D([0], [0], color='gray', linestyle='--', 
                      linewidth=1.5, label='.500 / League Avg'),
            mpatches.Patch(facecolor=PLAYOFF_COLORS[5], label='Champions', alpha=0.6),
            mpatches.Patch(facecolor=PLAYOFF_COLORS[4], label='Finals', alpha=0.6),
            mpatches.Patch(facecolor=PLAYOFF_COLORS[3], label='Conf Finals', alpha=0.6),
            mpatches.Patch(facecolor=PLAYOFF_COLORS[2], label='2nd Rd', alpha=0.6),
            mpatches.Patch(facecolor=PLAYOFF_COLORS[1], label='1st Rd', alpha=0.6),
        ]
        # Place legend in the figure margin below both subplots (outside all chart areas)
        for leg in fig.legends:
            leg.remove()
        fig.legend(handles=legend_elements, loc='lower center',
                   bbox_to_anchor=(0.5, 0.01), fontsize=8, ncol=4, framealpha=0.95)
        
        # --- INFO PANEL: CURRENT SEASON DETAILS ---
        
        ax2.axis('off')
        
        # Get current season stats
        current_row = current_data.iloc[-1]
        
        playoff_result = current_row['Playoff_Result'] if 'Playoff_Result' in current_row else 'N/A'
        wins = int(current_row['W']) if 'W' in current_row else 'N/A'
        losses = int(current_row['L']) if 'L' in current_row else 'N/A'
        ortg = f"{current_row['ORtg']:.1f}" if 'ORtg' in current_row and pd.notna(current_row['ORtg']) else 'N/A'
        drtg = f"{current_row['DRtg']:.1f}" if 'DRtg' in current_row and pd.notna(current_row['DRtg']) else 'N/A'
        nrtg = f"{current_row['NRtg']:.1f}" if 'NRtg' in current_row and pd.notna(current_row['NRtg']) else 'N/A'
        
        # Get ranks
        ortg_rank = 'N/A'
        drtg_rank = 'N/A'
        nrtg_rank = 'N/A'
        if 'ORtg_Rank_Norm' in current_row and pd.notna(current_row['ORtg_Rank_Norm']):
            ortg_rank_num = int((1 - current_row['ORtg_Rank_Norm']) * 30 + 1)
            ortg_rank = f"#{ortg_rank_num}"
        if 'DRtg_Rank_Norm' in current_row and pd.notna(current_row['DRtg_Rank_Norm']):
            drtg_rank_num = int((1 - current_row['DRtg_Rank_Norm']) * 30 + 1)
            drtg_rank = f"#{drtg_rank_num}"
        if 'NRtg_Rank_Norm' in current_row and pd.notna(current_row['NRtg_Rank_Norm']):
            nrtg_rank_num = int((1 - current_row['NRtg_Rank_Norm']) * 30 + 1)
            nrtg_rank = f"#{nrtg_rank_num}"
        
        pace = f"{current_row['Pace']:.1f}" if 'Pace' in current_row and pd.notna(current_row['Pace']) else 'N/A'
        three_pa = f"{current_row['3PA']:.1f}" if '3PA' in current_row and pd.notna(current_row['3PA']) else 'N/A'
        
        # Create info text
        info_text = f"""
        SEASON: {current_season} (Season {frame + 1} of {total_seasons})
        
        RECORD: {wins}-{losses} ({current_win_pct:.3f})
        PLAYOFFS: {playoff_result}
        
        HEAD COACH: {current_coach}
        GENERAL MANAGER: {current_gm}
        
        TEAM PERFORMANCE:
        • Net Rating: {nrtg} (Rank: {nrtg_rank})
        • Off Rating: {ortg} (Rank: {ortg_rank})
        • Def Rating: {drtg} (Rank: {drtg_rank})
        • Pace: {pace} | 3PA: {three_pa}
        """
        
        ax2.text(0.05, 0.5, info_text, fontsize=12, 
                verticalalignment='center',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8),
                family='monospace')
        
        # Add cumulative stats
        cumulative_wins = current_data['W'].sum() if 'W' in current_data else 0
        cumulative_losses = current_data['L'].sum() if 'L' in current_data else 0
        cumulative_win_pct = cumulative_wins / (cumulative_wins + cumulative_losses) if (cumulative_wins + cumulative_losses) > 0 else 0
        playoff_appearances = current_data['Playoff_Depth'].apply(lambda x: 1 if pd.notna(x) and x > 0 else 0).sum() if 'Playoff_Depth' in current_data else 0
        championships = current_data['Playoff_Depth'].apply(lambda x: 1 if pd.notna(x) and x == 5 else 0).sum() if 'Playoff_Depth' in current_data else 0
        
        cumulative_text = f"""
        FRANCHISE TOTALS (Through {current_season}):
        
        • All-Time Record: {int(cumulative_wins)}-{int(cumulative_losses)} ({cumulative_win_pct:.3f})
        • Playoff Appearances: {int(playoff_appearances)}
        • Championships: {int(championships)}
        """
        
        ax2.text(0.55, 0.5, cumulative_text, fontsize=12,
                verticalalignment='center',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8),
                family='monospace')
        
        return []
    
    # Create animation
    print(f"\nCreating animation with {total_seasons} frames...")
    print(f"Animation speed: {fps} frames per second")
    print("Adding 3-second hold at end to show complete timeline...")
    print("This may take a few minutes...")
    
    # Add extra frames at the end to hold final image
    hold_frames = int(fps * 3)  # Hold for 3 seconds
    total_frames = total_seasons + hold_frames
    
    def animate_with_hold(frame):
        """Animate with hold frames at end"""
        # Cap frame at last season for hold period
        actual_frame = min(frame, total_seasons - 1)
        return animate(actual_frame)
    
    anim = FuncAnimation(fig, animate_with_hold, init_func=init,
                        frames=total_frames, interval=1000/fps,
                        blit=True, repeat=False)  # Changed to False
    
    # Save as GIF
    gif_filename = f"{team_name.replace(' ', '_')}_evolution.gif"
    writer = PillowWriter(fps=fps)
    
    print(f"\nSaving animation to {gif_filename}...")
    # Note: GIF format loops by default - can't be disabled in file format
    anim.save(gif_filename, writer=writer, dpi=100, 
              savefig_kwargs={'facecolor': 'white'})
    
    plt.close()
    
    print(f"✓ Animation saved: {gif_filename}")

    # Post-process: patch the NETSCAPE loop count bytes in the raw GIF file.
    # This avoids re-encoding frames (which corrupts palettes) and simply
    # changes the 2-byte loop count from 0x0000 (infinite) to 0x0001 (play once).
    with open(gif_filename, 'rb') as f:
        data = bytearray(f.read())
    netscape_tag = b'NETSCAPE2.0'
    idx = data.find(netscape_tag)
    if idx != -1:
        # Layout after tag: 0x03 0x01 [LO] [HI] 0x00
        loop_offset = idx + len(netscape_tag) + 2
        data[loop_offset] = 1      # low byte  → loop count = 1 (play once)
        data[loop_offset + 1] = 0  # high byte
        with open(gif_filename, 'wb') as f:
            f.write(data)
    print(f"  Looping disabled: GIF will play once and stop on the final frame.")
    print(f"  File size may be large. Open with any image viewer or browser.")
    
    return anim


def create_video_mp4(team_df, df, team_name='Toronto Raptors', fps=2):
    """
    Create MP4 video (requires ffmpeg installed)
    
    Parameters:
    -----------
    team_df : DataFrame
        Team data with all seasons
    df : DataFrame
        Full league dataset (needed for rank calculations)
    team_name : str
        Name of the team
    fps : int
        Frames per second
    """
    
    try:
        from matplotlib.animation import FFMpegWriter
        
        # Check data
        if 'Head_Coach' not in team_df.columns or 'General_Manager' not in team_df.columns:
            print("⚠ Missing coach or GM data.")
            return None
        
        # Fill NaN
        team_df['Head_Coach'] = team_df['Head_Coach'].fillna('Unknown')
        team_df['General_Manager'] = team_df['General_Manager'].fillna('Unknown')
        
        # Determine win% column
        win_pct_col = 'WIN_PCT' if 'WIN_PCT' in team_df.columns else 'W_PCT'
        
        # Pre-calculate ranks
        print("Pre-calculating league ranks for MP4...")
        team_df['ORtg_Rank_Norm'] = None
        team_df['DRtg_Rank_Norm'] = None
        team_df['NRtg_Rank_Norm'] = None
        
        for idx, team_row in team_df.iterrows():
            season = team_row['SEASON']
            season_league = df[df['SEASON'] == season]
            
            if pd.notna(team_row['ORtg']) and len(season_league) > 0:
                ortg_rank = (season_league['ORtg'] > team_row['ORtg']).sum() + 1
                team_df.at[idx, 'ORtg_Rank_Norm'] = 1 - (ortg_rank - 1) / 30
                
                drtg_rank = (season_league['DRtg'] < team_row['DRtg']).sum() + 1
                team_df.at[idx, 'DRtg_Rank_Norm'] = 1 - (drtg_rank - 1) / 30
                
                if 'NRtg' in team_row and pd.notna(team_row['NRtg']):
                    nrtg_rank = (season_league['NRtg'] > team_row['NRtg']).sum() + 1
                    team_df.at[idx, 'NRtg_Rank_Norm'] = 1 - (nrtg_rank - 1) / 30
        
        print("✓ Ranks calculated")
        
        seasons = team_df['SEASON'].tolist()
        total_seasons = len(seasons)
        
        # Add hold frames
        hold_frames = int(fps * 3)
        total_frames = total_seasons + hold_frames
        
        # Set up the figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        # Use same animate function as GIF version
        def init():
            ax1.clear()
            ax2.clear()
            return []
        
        def animate(frame):
            # Cap at last season for hold period
            actual_frame = min(frame, total_seasons - 1)
            ax1.clear()
            ax2.clear()
            
            current_data = team_df.iloc[:actual_frame+1]
            current_seasons = seasons[:actual_frame+1]
            x_positions = np.arange(len(current_seasons))
            
            # Add era background shading
            if team_name in FRANCHISE_ERAS:
                for era in FRANCHISE_ERAS[team_name]:
                    era_start_year = era['start']
                    era_end_year = era['end']
                    
                    for i, season in enumerate(current_seasons):
                        season_year = int(season.split('-')[0]) + 1
                        
                        if era_start_year <= season_year <= era_end_year:
                            era_start_idx = i
                            era_end_idx = i
                            for j in range(i, len(current_seasons)):
                                check_year = int(current_seasons[j].split('-')[0]) + 1
                                if check_year <= era_end_year:
                                    era_end_idx = j
                                else:
                                    break
                            
                            ax1.axvspan(era_start_idx - 0.5, era_end_idx + 0.5, 
                                       alpha=0.15, color=era['color'], zorder=0)
                            
                            if era_end_idx > era_start_idx or i == 0:
                                mid_point = (era_start_idx + era_end_idx) / 2
                                ax1.text(mid_point, 0.98, era['name'], 
                                       ha='center', va='top', fontsize=8,
                                       color=era['color'], fontweight='bold',
                                       bbox=dict(boxstyle='round,pad=0.3', 
                                               facecolor='white', alpha=0.7,
                                               edgecolor=era['color'], linewidth=1.5),
                                       transform=ax1.get_xaxis_transform())
                            break
            
            # Plot playoff bars
            if 'Playoff_Depth' in current_data.columns:
                for i, depth in enumerate(current_data['Playoff_Depth']):
                    if pd.notna(depth):
                        depth = int(depth)
                        color = PLAYOFF_COLORS.get(depth, '#E8E8E8')
                        height = PLAYOFF_HEIGHTS.get(depth, 0.05)
                        
                        ax1.bar(i, height, width=0.9, bottom=0, 
                               color=color, alpha=0.6, zorder=1,
                               edgecolor='white', linewidth=1)
            
            # Plot win% line
            win_pcts = current_data[win_pct_col].values
            ax1.plot(x_positions, win_pcts, 'o-', linewidth=3, markersize=10,
                    color='#CE1141', zorder=3, label='Win %',
                    markeredgecolor='white', markeredgewidth=2)
            
            # Add ORtg and DRtg ranks on secondary y-axis
            if 'ORtg_Rank_Norm' in current_data.columns and 'DRtg_Rank_Norm' in current_data.columns:
                ax1_twin = ax1.twinx()
                
                ortg_ranks = current_data['ORtg_Rank_Norm'].values
                drtg_ranks = current_data['DRtg_Rank_Norm'].values
                
                ax1_twin.plot(x_positions, ortg_ranks, '^-', linewidth=1.2, markersize=4,
                             color='#2ECC71', alpha=0.4, label='Offense', zorder=1.8)
                ax1_twin.plot(x_positions, drtg_ranks, 'v-', linewidth=1.2, markersize=4,
                             color='#E74C3C', alpha=0.4, label='Defense', zorder=1.8)
                
                if 'NRtg_Rank_Norm' in current_data.columns:
                    nrtg_ranks = current_data['NRtg_Rank_Norm'].values
                    ax1_twin.plot(x_positions, nrtg_ranks, '-', linewidth=0.6,
                                 color='#999999', alpha=0.2, zorder=1.5)
                
                ax1_twin.set_ylim(-0.05, 1.05)
                ax1_twin.set_ylabel('League Rank', fontsize=9, color='#888')
                ax1_twin.tick_params(axis='y', labelcolor='#888', labelsize=8)
                ax1_twin.legend(loc='lower right', fontsize=8, framealpha=0.7)
                ax1_twin.grid(False)
            
            ax1.axhline(y=0.500, color='gray', linestyle='--', linewidth=1.5,
                       alpha=0.5, zorder=2)
            
            ax1.set_xlim(-0.5, total_seasons - 0.5)
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_xticks(x_positions)
            ax1.set_xticklabels(current_seasons, rotation=45, ha='right', fontsize=9)
            ax1.set_ylabel('Win %', fontsize=13, fontweight='bold')
            ax1.set_title(f'{team_name} - Franchise Evolution', 
                         fontsize=16, fontweight='bold', pad=15)
            ax1.grid(True, alpha=0.3, axis='y', zorder=0)
            
            # Highlight current
            current_win_pct = win_pcts[-1]
            ax1.plot(actual_frame, current_win_pct, 'o', markersize=18,
                    color='yellow', zorder=4, markeredgecolor='red', 
                    markeredgewidth=3)
            
            # Info panel
            ax2.axis('off')
            
            current_row = current_data.iloc[-1]
            current_season = current_seasons[-1]
            current_coach = current_data.iloc[-1]['Head_Coach']
            current_gm = current_data.iloc[-1]['General_Manager']
            
            playoff_result = current_row['Playoff_Result'] if 'Playoff_Result' in current_row else 'N/A'
            wins = int(current_row['W']) if 'W' in current_row else 'N/A'
            losses = int(current_row['L']) if 'L' in current_row else 'N/A'
            
            ortg = f"{current_row['ORtg']:.1f}" if 'ORtg' in current_row and pd.notna(current_row['ORtg']) else 'N/A'
            drtg = f"{current_row['DRtg']:.1f}" if 'DRtg' in current_row and pd.notna(current_row['DRtg']) else 'N/A'
            nrtg = f"{current_row['NRtg']:.1f}" if 'NRtg' in current_row and pd.notna(current_row['NRtg']) else 'N/A'
            
            # Get ranks
            ortg_rank = 'N/A'
            drtg_rank = 'N/A'
            nrtg_rank = 'N/A'
            if 'ORtg_Rank_Norm' in current_row and pd.notna(current_row['ORtg_Rank_Norm']):
                ortg_rank_num = int((1 - current_row['ORtg_Rank_Norm']) * 30 + 1)
                ortg_rank = f"#{ortg_rank_num}"
            if 'DRtg_Rank_Norm' in current_row and pd.notna(current_row['DRtg_Rank_Norm']):
                drtg_rank_num = int((1 - current_row['DRtg_Rank_Norm']) * 30 + 1)
                drtg_rank = f"#{drtg_rank_num}"
            if 'NRtg_Rank_Norm' in current_row and pd.notna(current_row['NRtg_Rank_Norm']):
                nrtg_rank_num = int((1 - current_row['NRtg_Rank_Norm']) * 30 + 1)
                nrtg_rank = f"#{nrtg_rank_num}"
            
            info_text = f"""
            SEASON: {current_season} (Season {actual_frame + 1} of {total_seasons})
            
            RECORD: {wins}-{losses} ({current_win_pct:.3f})
            PLAYOFFS: {playoff_result}
            
            COACH: {current_coach}
            GM: {current_gm}
            
            TEAM PERFORMANCE:
            • Net Rating: {nrtg} (Rank: {nrtg_rank})
            • Off Rating: {ortg} (Rank: {ortg_rank})
            • Def Rating: {drtg} (Rank: {drtg_rank})
            """
            
            ax2.text(0.05, 0.5, info_text, fontsize=12, 
                    verticalalignment='center',
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8),
                    family='monospace')
            
            return []
        
        anim = FuncAnimation(fig, animate, init_func=init,
                            frames=total_frames, interval=1000/fps,
                            blit=True, repeat=False)  # Changed to False
        
        mp4_filename = f"{team_name.replace(' ', '_')}_evolution.mp4"
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        
        print(f"\nSaving video to {mp4_filename}...")
        anim.save(mp4_filename, writer=writer, dpi=100)
        
        plt.close()
        
        print(f"✓ Video saved: {mp4_filename}")
        
        return anim
        
    except ImportError:
        print("⚠ FFmpeg not found. Cannot create MP4 video.")
        print("  Install FFmpeg to enable MP4 export, or use GIF format instead.")
        return None


def main():
    """Main execution"""
    
    print("=" * 100)
    print("NBA FRANCHISE EVOLUTION ANIMATION")
    print("=" * 100)
    
    # Load data
    df = pd.read_csv('nba_team_stats_1980_2025.csv')
    
    # Select team
    print("\nDefault: Toronto Raptors")
    team_name = input("Enter team name (or press Enter for Raptors): ").strip()
    if not team_name:
        team_name = 'Toronto Raptors'
    
    # Get team data
    team_df = df[df['Team'] == team_name].copy()
    
    if len(team_df) == 0:
        print(f"Error: No data found for '{team_name}'")
        return
    
    # Load historical data
    historical_file = f"{team_name.replace(' ', '_').lower()}_historical_data.csv"
    
    try:
        historical_df = pd.read_csv(historical_file)
        print(f"✓ Loaded historical data: {historical_file}")
        
        team_df = team_df.merge(
            historical_df[['SEASON', 'Head_Coach', 'General_Manager', 'Playoff_Result', 'Playoff_Depth']], 
            on='SEASON', 
            how='left'
        )
        print(f"✓ Merged historical data")
        
    except FileNotFoundError:
        print(f"⚠ No historical data found: {historical_file}")
        return
    
    team_df = team_df.sort_values('SEASON').reset_index(drop=True)
    
    print(f"\n{'='*100}")
    print(f"CREATING ANIMATION: {team_name}")
    print(f"{'='*100}")
    print(f"Total seasons: {len(team_df)}")
    
    # Ask for format
    print("\nChoose format:")
    print("1. GIF (works everywhere, larger file size)")
    print("2. MP4 (smaller file, requires ffmpeg)")
    choice = input("Enter choice (1 or 2, default=1): ").strip()
    
    # Ask for speed
    print("\nChoose animation speed:")
    print("1. Slow (1 season per 2 seconds)")
    print("2. Medium (1 season per second) - Recommended")
    print("3. Fast (2 seasons per second)")
    speed_choice = input("Enter choice (1-3, default=2): ").strip()
    
    fps_map = {'1': 0.5, '2': 1, '3': 2}
    fps = fps_map.get(speed_choice, 1)
    
    if choice == '2':
        create_video_mp4(team_df, df, team_name, fps)
    else:
        create_franchise_animation(team_df, df, team_name, fps)
    
    print("\n" + "=" * 100)
    print("✓ ANIMATION CREATED!")
    print("=" * 100)
    print("\nYou can now:")
    print("• Open the file in any image viewer")
    print("• Embed in PowerPoint presentations")
    print("• Share on social media")
    print("• Include in your portfolio website")


if __name__ == "__main__":
    main()
