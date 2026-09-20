import pandas as pd

raw = pd.read_csv("raw_shots.csv", dtype={"GAME_ID": str})

games = raw["GAME_ID"].nunique()
print(f"Rows: {len(raw)} | games: {games} | avg shots per game: {len(raw) / games:.1f}")
print(f"Date range: {raw['GAME_DATE'].min()} to {raw['GAME_DATE'].max()}")

print("\nShots per month (gaps or a sudden stop show up here):")
print(raw.groupby(raw["GAME_DATE"].astype(str).str[:6]).size().to_string())

print("\nTeams present per game (2 = both teams, 1 = one team only):")
print(raw.groupby("GAME_ID")["TEAM_ID"].nunique().value_counts().to_string())

print(f"\nTeams in data: {raw['TEAM_ID'].nunique()}")
print(raw.groupby("TEAM_NAME").size().describe().to_string())

print("\nTop 5 players by attempts:")
print(raw.groupby("PLAYER_NAME").size().sort_values(ascending=False).head(5).to_string())