import time
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import shotchartdetail

# 1. FETCH RAW DATA
def fetch_shot_data(season="2025-26", player_id=0):
    """
    Pulls shot chart data from the NBA stats API.
    player_id=0 fetches league-wide data, or pass a specific player ID.
    """
    print(f"Fetching shot data for season {season}...")
    shot_chart = shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=player_id,
        season_type_all_star="Regular Season",
        season_nullable=season,
        context_measure_simple="FGA"
    )
    # The first data frame contains the individual shot attempts
    df = shot_chart.get_data_frames()[0]
    return df

#Over 70 unique action types in the NBA shot data, so we can group them into broader categories for modeling purposes.
def group_action_types(action):
    action = str(action).lower()
    if "dunk" in action or "alley oop" in action:
        return "Dunk"
    elif "layup" in action or "finger roll" in action:
        return "Layup"
    elif "hook" in action:
        return "Hook"
    elif "fadeaway" in action or "step back" in action or "turnaround" in action:
        return "Fadeaway/StepBack"
    elif "floating" in action or "floater" in action:
        return "Floater"
    elif "pullup" in action:
        return "Pullup"
    elif "tip" in action:
        return "Putback"
    else:
        return "Standard Jump Shot"

# 2. FEATURE ENGINEERING (SPATIAL GEOMETRY)
def engineer_shot_features(df):
    """
    Extracts spatial distance, shot angle, and shot value.
    """
    # Calculate Euclidean distance in feet (NBA API coordinates are in tenths of a foot)
    df["CALCULATED_DIST"] = np.sqrt(df["LOC_X"]**2 + df["LOC_Y"]**2) / 10
    
    # Calculate shot angle in radians relative to the hoop center
    df["SHOT_ANGLE"] = np.arctan2(df["LOC_X"], df["LOC_Y"])
    
    # Binary target variable: 1 = Make, 0 = Miss
    df["TARGET"] = df["SHOT_MADE_FLAG"].astype(int)
    
    # Value of shot attempted (2-pointer vs 3-pointer)
    df["SHOT_VALUE"] = df["SHOT_TYPE"].apply(lambda x: 3 if "3PT" in x else 2)
    df["ACTION_GROUP"] = df["ACTION_TYPE"].apply(group_action_types)
    return df
# 3. DATA CLEANING & FILTERING
def clean_shot_data(df):
    """
    Removes edge cases, heaves, and missing values.
    """
    # Select only the relevant modeling columns
    columns_to_keep = [
        "GAME_ID", "GAME_EVENT_ID", "PLAYER_NAME", "TEAM_NAME", "PERIOD", "MINUTES_REMAINING", 
        "SECONDS_REMAINING", "ACTION_TYPE", "SHOT_TYPE", "SHOT_ZONE_BASIC", 
        "LOC_X", "LOC_Y", "SHOT_DISTANCE", "CALCULATED_DIST", 
        "SHOT_ANGLE", "SHOT_VALUE", "TARGET", "ACTION_GROUP"
    ]
    df_clean = df[columns_to_keep].dropna().copy()
    
    # Filter out backcourt heaves (>40 feet) that distort expected probability
    df_clean = df_clean[df_clean["SHOT_DISTANCE"] <= 40]
    
    return df_clean


# 4. MAIN PIPELINE EXECUTION
def main():
    raw_df = fetch_shot_data(season="2025-26", player_id=0) # 0 for all players, or test with Steph Curry: 201939
    clean_df = engineer_shot_features(raw_df)
    final_df = clean_shot_data(clean_df)
    
    # Save output to disk
    final_df.to_csv("shots_cleaned.csv", index=False)
    print(f"✅ Phase 1 complete! Saved {len(final_df)} cleaned shots to shots_cleaned.csv")

if __name__ == "__main__":
    main()