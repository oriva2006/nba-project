import streamlit as st
import textwrap
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from nba_api.stats.static import players as nba_players

# 1. STREAMLIT APP CONFIG & DARK THEME HUD
st.set_page_config(
    page_title="NBA 2K Scouting & Shot Quality",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0b0e14;
        color: #e6edf3;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        font-weight: 700;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.7rem;
        font-weight: 900;
        color: #58a6ff;
    }
    .player-banner {
        display: flex;
        align-items: center;
        background: linear-gradient(90deg, #161b22 0%, #1f2937 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
    }
    .player-photo {
        width: 90px;
        height: 90px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid #58a6ff;
        background-color: #0d1117;
        margin-right: 20px;
    }
    .player-title h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
    }
    .player-title p {
        margin: 4px 0 0 0;
        color: #8b949e;
        font-size: 0.95rem;
    }
    .badge-hot {
        background-color: rgba(235, 87, 87, 0.2);
        color: #ff6b6b;
        border: 1px solid #eb5757;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .badge-cold {
        background-color: rgba(47, 128, 237, 0.2);
        color: #5dade2;
        border: 1px solid #2f80ed;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .badge-neutral {
        background-color: rgba(139, 148, 158, 0.2);
        color: #8b949e;
        border: 1px solid #8b949e;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

st.caption("xFG% uses shot location, shot type and game clock only. It has no defender or shot-clock data.")
# 2. MODEL & DATA LOADERS
@st.cache_resource
def load_model():
    return joblib.load("shot_success_model.pkl")

shot_success_model = load_model()

@st.cache_data
def get_processed_data():
    df = pd.read_csv("shots_cleaned.csv", dtype={"GAME_ID": str, "GAME_EVENT_ID": int})
    
    numerical_cols = [
        "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING",
        "SHOT_DISTANCE", "CALCULATED_DIST", "SHOT_ANGLE", "SHOT_VALUE"
    ]
    categorical_cols = ["SHOT_ZONE_BASIC", "SHOT_TYPE", "ACTION_GROUP"]
    X = df[numerical_cols + categorical_cols]
    
    df["XFG"] = shot_success_model.predict_proba(X)[:, 1]
    return df 

df = get_processed_data()

# 3. PLAYER HEADSHOT HELPER
@st.cache_data
def get_player_headshot_url(player_name: str) -> str:
    matches = nba_players.find_players_by_full_name(player_name)
    if matches:
        return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{matches[0]['id']}.png"
    return "https://cdn.nba.com/headshots/nba/latest/1040x760/fallback.png"

# 4. MATHEMATICALLY SCALED NBA COURT
def draw_court(fig):
    # Rim & Net
    fig.add_shape(type="circle", x0=-7.5, y0=-7.5, x1=7.5, y1=7.5, 
                  line=dict(color="#f39c12", width=2.5), fillcolor="rgba(243, 156, 18, 0.2)")
    
    # Backboard
    fig.add_shape(type="line", x0=-30, y0=-7.5, x1=30, y1=-7.5, 
                  line=dict(color="#ffffff", width=3))
    
    # Paint (Key)
    fig.add_shape(type="rect", x0=-80, y0=-47.5, x1=80, y1=142.5, 
                  line=dict(color="#484f58", width=1.8), fillcolor="rgba(255, 255, 255, 0.02)")
    
    # Free Throw Circle
    fig.add_shape(type="circle", x0=-60, y0=82.5, x1=60, y1=202.5, 
                  line=dict(color="#484f58", width=1.5))
    
    # Restricted Area Arc (4ft)
    theta = np.linspace(0, np.pi, 50)
    fig.add_trace(go.Scatter(
        x=40 * np.cos(theta), y=40 * np.sin(theta),
        mode="lines", line=dict(color="#30363d", width=1.5, dash="dot"),
        showlegend=False, hoverinfo="skip"
    ))

    # Corner 3-Point Lines
    fig.add_shape(type="line", x0=-220, y0=-47.5, x1=-220, y1=92.5, line=dict(color="#6e7681", width=1.8))
    fig.add_shape(type="line", x0=220, y0=-47.5, x1=220, y1=92.5, line=dict(color="#6e7681", width=1.8))
    
    # 3-Point Arc
    arc_angle = np.arcsin(92.5 / 237.5)
    angles = np.linspace(arc_angle, np.pi - arc_angle, 100)
    fig.add_trace(go.Scatter(
        x=237.5 * np.cos(angles), y=237.5 * np.sin(angles),
        mode="lines", line=dict(color="#6e7681", width=1.8),
        showlegend=False, hoverinfo="skip"
    ))

    # Outer Boundary
    fig.add_shape(type="rect", x0=-250, y0=-47.5, x1=250, y1=422.5, line=dict(color="#484f58", width=2.5))

    # LOCKED 1:1 ASPECT RATIO
    fig.update_layout(
        xaxis=dict(range=[-260, 260], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-55, 435], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, scaleanchor="x", scaleratio=1),
        plot_bgcolor="#12161f",
        paper_bgcolor="#12161f",
        height=580,
        margin=dict(l=5, r=5, t=5, b=5),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color="#e6edf3", size=12)
        )
    )
    return fig

# 5. METRICS CALCULATION
def calculate_player_metrics(df: pd.DataFrame, player_name: str) -> dict[str, float]:
    p_df = df[df["PLAYER_NAME"] == player_name]
    total = len(p_df)
    if total == 0:
        return {"total": 0, "actual_fg": 0.0, "exp_fg": 0.0, "delta": 0.0, "avg_xpts": 0.0}
        
    actual_fg = (p_df["TARGET"].sum() / total) * 100
    exp_fg = p_df["XFG"].mean() * 100
    delta = actual_fg - exp_fg
    avg_xpts = (p_df["XFG"] * p_df["SHOT_VALUE"]).mean()

    return {
        "total": total,
        "actual_fg": round(actual_fg, 1),
        "exp_fg": round(exp_fg, 1),
        "delta": round(delta, 1),
        "avg_xpts": round(avg_xpts, 2)
    }

# 6. SIDEBAR CONTROLS
st.sidebar.markdown("### ⚙️ Player Selection & Filters")
players = sorted(df["PLAYER_NAME"].unique())
selected_player = st.sidebar.selectbox("Select Player", players, index=0)

zone_filter = st.sidebar.multiselect(
    "Filter by Shot Zone",
    options=sorted(df["SHOT_ZONE_BASIC"].unique()),
    default=sorted(df["SHOT_ZONE_BASIC"].unique())
)

player_df = df[
    (df["PLAYER_NAME"] == selected_player) & 
    (df["SHOT_ZONE_BASIC"].isin(zone_filter))
].copy().reset_index(drop=True)
player_df["ROW_ID"] = player_df.index

metrics = calculate_player_metrics(df, selected_player)
headshot_url = get_player_headshot_url(selected_player)

# Status Badge
MIN_FGA = 200
if metrics["total"] < MIN_FGA:
    badge_html = '<span class="badge-neutral">📉 Small sample</span>'
elif metrics['delta'] >= 2.0:
    badge_html = f'<span class="badge-hot">🔥 Hot Shot Maker (+{metrics["delta"]} pts)</span>'
elif metrics['delta'] <= -2.0:
    badge_html = f'<span class="badge-cold">❄️ Below Expectation ({metrics["delta"]} pts)</span>'
else:
    badge_html = f'<span class="badge-neutral">⚖️ Neutral Baseline ({metrics["delta"]:+.1f} pts)</span>'

team_name = player_df["TEAM_NAME"].iloc[0] if len(player_df) > 0 else "NBA"

# 7. TOP PLAYER BANNER
st.markdown(f"""
<div class="player-banner">
    <img src="{headshot_url}" class="player-photo" onerror="this.style.display='none'">
    <div class="player-title">
        <h1>{selected_player}</h1>
        <p>{team_name} | 2025-26 NBA Shot Quality Scouting Report</p>
        <div style="margin-top: 8px;">{badge_html}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 8. METRIC ROW
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total FGA", f"{metrics['total']}")
m2.metric("Actual FG%", f"{metrics['actual_fg']}%")
m3.metric("Expected xFG%", f"{metrics['exp_fg']}%")
m4.metric("Shooting Delta", f"{metrics['delta']:+.1f} pts")
m5.metric("Avg Point Yield", f"{metrics['avg_xpts']} pts")

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# 9. TWO-COLUMN SCOUTING LAYOUT
col_court, col_film = st.columns([3, 2], gap="large")

with col_court:
    st.markdown("### 🏟️ Spatial Shot Chart")
    fig = go.Figure()
    draw_court(fig)

    if len(player_df) > 0:
        # Missed Shots (Coral X's)
        misses = player_df[player_df["TARGET"] == 0]
        if len(misses) > 0:
            fig.add_trace(go.Scatter(
                x=misses["LOC_X"], y=misses["LOC_Y"],
                mode="markers", name="Missed Shot",
                marker=dict(color="#ff3366", size=8, symbol="x", opacity=0.8),
                customdata=np.stack((
                    misses["ACTION_GROUP"], misses["SHOT_DISTANCE"], 
                    (misses["XFG"] * 100).round(1), misses["ROW_ID"]
                ), axis=-1),
                hovertemplate=(
                    "<b>❌ MISS</b><br>" +
                    "<b>Type:</b> %{customdata[0]}<br>" +
                    "<b>Distance:</b> %{customdata[1]} ft<br>" +
                    "<b>Model xFG%:</b> %{customdata[2]}%<extra></extra>"
                )
            ))

        # Made Shots (Emerald Circles)
        makes = player_df[player_df["TARGET"] == 1]
        if len(makes) > 0:
            fig.add_trace(go.Scatter(
                x=makes["LOC_X"], y=makes["LOC_Y"],
                mode="markers", name="Made Shot",
                marker=dict(color="#00e676", size=9, symbol="circle", opacity=0.9, line=dict(width=1, color="#000000")),
                customdata=np.stack((
                    makes["ACTION_GROUP"], makes["SHOT_DISTANCE"], 
                    (makes["XFG"] * 100).round(1), makes["ROW_ID"]
                ), axis=-1),
                hovertemplate=(
                    "<b>✅ MAKE</b><br>" +
                    "<b>Type:</b> %{customdata[0]}<br>" +
                    "<b>Distance:</b> %{customdata[1]} ft<br>" +
                    "<b>Model xFG%:</b> %{customdata[2]}%<extra></extra>"
                )
            ))


    # 2K Zone Hot/Cold Breakdown
    if len(player_df) > 0:
        st.markdown("#### 🎯 2K Zone Efficiency Ratings")
        zone_summary = player_df.groupby("SHOT_ZONE_BASIC").agg(
            FGA=("TARGET", "count"),
            Actual_FG=("TARGET", lambda x: round(x.mean() * 100, 1)),
            Expected_xFG=("XFG", lambda x: round(x.mean() * 100, 1))
        ).reset_index()
        zone_summary["Delta"] = (zone_summary["Actual_FG"] - zone_summary["Expected_xFG"]).round(1)
        zone_summary["2K Rating"] = zone_summary["Delta"].apply(
            lambda d: "🔥 Hot" if d >= 2.0 else ("❄️ Cold" if d <= -2.0 else "⚖️ Neutral")
        )
        st.dataframe(zone_summary, use_container_width=True, hide_index=True)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
        on_select="rerun",
        selection_mode="points",
        key="shot_chart"
    )
with col_film:
    st.markdown("### 📋 Shot Quality & Value Breakdown")

    if len(player_df) > 0:
        player_df["SHOT_LABEL"] = player_df.apply(
            lambda r: f"#{r['GAME_EVENT_ID']} | Q{r['PERIOD']} {r['MINUTES_REMAINING']}:{str(r['SECONDS_REMAINING']).zfill(2)} - {r['ACTION_GROUP']} ({r['SHOT_DISTANCE']}ft) - {'MAKE' if r['TARGET'] == 1 else 'MISS'}",
            axis=1
        )
        label_map = dict(zip(player_df["ROW_ID"], player_df["SHOT_LABEL"]))

        # Read the chart click, if any
        clicked_id = None
        sel = st.session_state.get("shot_chart")
        points = sel.selection["points"] if sel else []
        if points:
            clicked_id = int(points[0]["customdata"][3])

        # Key includes player and filter so options never go stale
        select_key = f"shot_select_{selected_player}_{'-'.join(zone_filter)}"

        # Push a NEW click into the dropdown, so the dropdown stays usable afterwards
        click_token = (selected_player, clicked_id)
        if clicked_id is None:
            st.session_state["last_click"] = None
        elif clicked_id in label_map and st.session_state.get("last_click") != click_token:
            st.session_state["last_click"] = click_token
            st.session_state[select_key] = clicked_id

        chosen_id = st.selectbox(
            "Select Attempt to Analyze",
            player_df["ROW_ID"].tolist(),
            format_func=lambda i: label_map[i],
            key=select_key,
        )
        shot_row = player_df.loc[chosen_id]

        # Shot quality tier, based on xFG% only
        xfg_pct = shot_row["XFG"] * 100
        if xfg_pct >= 60.0:
            diff_title = "🟢 High-Efficiency Look"
            diff_bar_color = "#00e676"
        elif xfg_pct >= 40.0:
            diff_title = "🟡 Average Look"
            diff_bar_color = "#f39c12"
        else:
            diff_title = "🔴 Low Quality Look"
            diff_bar_color = "#ff3366"

        outcome_color = "#00e676" if shot_row['TARGET'] == 1 else "#ff3366"
        outcome_text = "Made" if shot_row['TARGET'] == 1 else "Missed"

        hud_html = textwrap.dedent(f"""
        <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 18px; margin-bottom: 16px;">
            <p style="margin: 0 0 4px 0; font-size: 0.8rem; color: #8b949e; text-transform: uppercase; font-weight: 700;">Shot Quality (xFG%)</p>
            <p style="margin: 0 0 8px 0; font-weight: bold; color: {diff_bar_color};">{diff_title}</p>
            <div style="background-color: #30363d; border-radius: 6px; height: 10px; width: 100%; overflow: hidden; margin-bottom: 14px;">
                <div style="background-color: {diff_bar_color}; width: {xfg_pct:.1f}%; height: 100%;"></div>
            </div>
            <p style="margin: 0 0 6px 0;"><strong>🎯 Mechanic:</strong> {shot_row['ACTION_GROUP']} <span style="color: #8b949e;">({shot_row['SHOT_TYPE']})</span></p>
            <p style="margin: 0 0 6px 0;"><strong>📏 Coordinates:</strong> {shot_row['SHOT_DISTANCE']} ft | {np.degrees(shot_row['SHOT_ANGLE']):.1f}° angle</p>
            <p style="margin: 0 0 6px 0;"><strong>🤖 Expected Conversion (xFG%):</strong> <span style="color: #58a6ff; font-weight: 800;">{xfg_pct:.1f}%</span></p>
            <p style="margin: 0 0 6px 0;"><strong>✨ Expected Value (xPTS):</strong> <span style="color: #58a6ff; font-weight: 800;">{shot_row['XFG'] * shot_row['SHOT_VALUE']:.2f} pts</span></p>
            <p style="margin: 0;"><strong>🏁 Actual Outcome:</strong> <span style="color: {outcome_color}; font-weight: 800;">{outcome_text}</span></p>
        </div>
        """)
        st.markdown(hud_html, unsafe_allow_html=True)

        st.markdown("#### 📊 Shot Diet & Expected Efficiency")
        player_df["xPTS"] = player_df["XFG"] * player_df["SHOT_VALUE"]

        shot_diet = player_df.groupby("ACTION_GROUP").agg(
            Attempts=("TARGET", "count"),
            Actual_FG=("TARGET", lambda x: f"{(x.mean() * 100):.1f}%"),
            Expected_xFG=("XFG", lambda x: f"{(x.mean() * 100):.1f}%"),
            Avg_xPTS=("xPTS", lambda x: f"{x.mean():.2f}")
        ).reset_index()

        st.dataframe(shot_diet, use_container_width=True, hide_index=True)
    else:
        st.warning("No shot attempts match your active filter criteria.")