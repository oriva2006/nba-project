import pandas as pd
from nba_api.stats.static import players

# Fetch NBA player directory
all_players = players.get_players()
df = pd.DataFrame(all_players)

print(f"✅ Environment working! Loaded {len(df)} NBA players.")
print(df.head(3))