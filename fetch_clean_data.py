import argparse
import os
import re
import time
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import teams

SEASON = "2025-26"

# A single league-wide ShotChartDetail request stopped at exactly this many rows
# (data ended on 29 Jan 2026, ~47% of the season), so we request one team at a time.
API_ROW_CAP = 102400
FULL_SEASON_GAMES = 1230  # 30 teams x 82 games / 2

ACTION_GROUPS = [
    "Putback", "Dunk", "Layup", "Hook", "Fadeaway/StepBack", "Floater", "Pullup", "Standard Jump Shot"
]


# 1. FETCH RAW DATA
def check_not_truncated(df, label):
    """Guards against the silent row cap: a response of exactly API_ROW_CAP rows is cut off."""
    if len(df) >= API_ROW_CAP:
        raise RuntimeError(
            f"{label} returned {len(df)} rows, which matches the API row cap. "
            "The data is truncated, so request a smaller slice (e.g. a single team)."
        )


def fetch_shot_data(season=SEASON, player_id=0, team_id=0, retries=3):
    """
    Pulls shot chart data for one request, retrying on failures.
    NOTE: team_id=0 and player_id=0 (league-wide) hits the row cap, so use fetch_all_shots().
    """
    for attempt in range(retries):
        try:
            shot_chart = shotchartdetail.ShotChartDetail(
                team_id=team_id,
                player_id=player_id,
                season_type_all_star="Regular Season",
                season_nullable=season,
                context_measure_simple="FGA",
                timeout=60,
            )
            # The first data frame contains the individual shot attempts
            df = shot_chart.get_data_frames()[0]
            check_not_truncated(df, f"team_id={team_id}, player_id={player_id}")
            return df
        except RuntimeError:
            raise  # truncation is not a transient error, do not retry
        except Exception as e:
            wait = 3 * (attempt + 1)
            print(f"  attempt {attempt + 1} failed ({type(e).__name__}: {e}); retrying in {wait}s")
            time.sleep(wait)
    raise ConnectionError(f"All {retries} attempts failed for team_id={team_id}, player_id={player_id}")


def fetch_all_shots(season=SEASON):
    """
    Pulls every regular-season shot by requesting each of the 30 teams separately.
    Raises instead of silently skipping a team, so we never save a partial dataset.
    """
    frames, failed = [], []
    all_teams = teams.get_teams()
    for i, team in enumerate(all_teams, start=1):
        print(f"[{i}/{len(all_teams)}] {team['full_name']}...")
        try:
            df = fetch_shot_data(season=season, player_id=0, team_id=team["id"])
            print(f"  {len(df)} shots")
            frames.append(df)
        except ConnectionError as e:
            print(f"  FAILED: {e}")
            failed.append(team["full_name"])
        time.sleep(1)  # be polite to the API

    if failed:
        raise RuntimeError(f"No data saved. Teams that failed: {failed}. Re-run the script.")

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID"])
    print(f"Combined {before} rows, removed {before - len(combined)} duplicates")
    return combined


def group_action_types(action):
    """
    Rule-based grouping of the raw ACTION_TYPE labels into 8 archetypes.
    Rules are checked in order and the first match wins, so order matters.
    Putbacks are checked first: labels like "Tip Layup Shot" or "Putback Dunk Shot"
    would otherwise be swallowed by the Layup and Dunk rules.
    """
    # Strip hyphens so spellings like "Pull-Up" still match "pullup"
    action = str(action).lower().replace("-", "")
    if re.search(r"\b(tip|putback)\b", action):
        return "Putback"
    elif "dunk" in action or "alley oop" in action:
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
    else:
        return "Standard Jump Shot"


# 2. FEATURE ENGINEERING (SPATIAL GEOMETRY)
def engineer_shot_features(df):
    """
    Extracts spatial distance, shot angle, shot value and action group.
    """
    df = df.copy()

    # Euclidean distance in feet (NBA API coordinates are in tenths of a foot)
    df["CALCULATED_DIST"] = np.sqrt(df["LOC_X"]**2 + df["LOC_Y"]**2) / 10

    # Shot angle in radians relative to the hoop centre.
    # arctan2(x, y): 0 = straight on, +/- pi/2 = along the baseline, sign = left/right.
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
    columns_to_keep = [
        "GAME_ID", "GAME_EVENT_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_NAME",
        "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING", "ACTION_TYPE", "SHOT_TYPE",
        "SHOT_ZONE_BASIC", "LOC_X", "LOC_Y", "SHOT_DISTANCE", "CALCULATED_DIST",
        "SHOT_ANGLE", "SHOT_VALUE", "TARGET", "ACTION_GROUP"
    ]
    n_start = len(df)
    df_clean = df[columns_to_keep].dropna().copy()
    n_after_na = len(df_clean)

    # Filter out backcourt heaves (>40 feet) that distort expected probability
    df_clean = df_clean[df_clean["SHOT_DISTANCE"] <= 40]

    print(f"Rows: {n_start} raw -> {n_after_na} after dropna -> {len(df_clean)} after removing shots > 40 ft")
    return df_clean


# 4. DIAGNOSTICS
def print_diagnostics(raw_df, final_df):
    """
    Numbers worth knowing (and quoting) about the dataset.
    """
    games = raw_df["GAME_ID"].nunique()
    print("\n--- Data diagnostics ---")
    print(f"Raw rows: {len(raw_df)} | games: {games} (full season = {FULL_SEASON_GAMES}) | "
          f"players: {raw_df['PLAYER_ID'].nunique()}")
    print(f"Average shots per game: {len(raw_df) / games:.1f}")
    print(f"Date range: {raw_df['GAME_DATE'].min()} to {raw_df['GAME_DATE'].max()}")
    print("Shots per month:")
    print(raw_df.groupby(raw_df["GAME_DATE"].astype(str).str[:6]).size().to_string())
    print(f"Raw action types: {raw_df['ACTION_TYPE'].nunique()}")
    print(f"Make rate (cleaned): {final_df['TARGET'].mean():.3f}")
    diff = (final_df["CALCULATED_DIST"] - final_df["SHOT_DISTANCE"]).abs()
    print(f"Mean |CALCULATED_DIST - SHOT_DISTANCE|: {diff.mean():.2f} ft")
    print("\nTop 5 players by attempts:")
    print(final_df.groupby("PLAYER_NAME").size().sort_values(ascending=False).head(5).to_string())
    print("\nShots per ACTION_GROUP:")
    print(final_df["ACTION_GROUP"].value_counts().to_string())
    empty = [g for g in ACTION_GROUPS if g not in set(final_df["ACTION_GROUP"])]
    if empty:
        print(f"WARNING: groups with zero rows: {empty}")
    print(f"\nRaw ACTION_TYPE -> ACTION_GROUP ({final_df['ACTION_TYPE'].nunique()} types):")
    print(final_df.groupby(["ACTION_GROUP", "ACTION_TYPE"]).size().to_string())
    print("------------------------\n")


# 5. MAIN PIPELINE EXECUTION
def main(refresh=False):
    if os.path.exists("raw_shots.csv") and not refresh:
        print("Using cached raw_shots.csv (run with --refresh to re-pull from the API)")
        raw_df = pd.read_csv("raw_shots.csv", dtype={"GAME_ID": str})
    else:
        raw_df = fetch_all_shots(season=SEASON)
        # Keep the untouched pull so the API never needs re-hitting just to debug
        raw_df.to_csv("raw_shots.csv", index=False)

    featured_df = engineer_shot_features(raw_df)
    final_df = clean_shot_data(featured_df)
    print_diagnostics(raw_df, final_df)

    final_df.to_csv("shots_cleaned.csv", index=False)
    print(f"✅ Saved {len(final_df)} cleaned shots to shots_cleaned.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-pull raw data from the NBA API")
    main(refresh=parser.parse_args().refresh)