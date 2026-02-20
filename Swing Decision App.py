import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ----------------------------
# Config / constants
# ----------------------------
# TEAM_FILTER_VALUE was removed to allow dynamic selection

# Strike-zone bounds (matches your Shiny server section)
X_MIN, X_MAX = -0.95, 0.95
Z_MIN, Z_MAX = 1.5, 3.5  # note: your earlier preprocessing used 1.5–3.5; the Shiny plot uses 1.645–3.355

X_BREAKS = np.linspace(X_MIN, X_MAX, 4)  # 3 columns
Z_BREAKS = np.linspace(Z_MIN, Z_MAX, 4)  # 3 rows

ALL_COUNTS = [
    "0-0", "0-1", "0-2",
    "1-0", "1-1", "1-2",
    "2-0", "2-1", "2-2",
    "3-0", "3-1", "3-2",
]

HITTER_COUNTS = {"2-0", "3-0", "3-1", "1-0"}
PITCHER_COUNTS = {"0-2", "1-2", "2-2"}

GOOD_PITCH_RADIUS = 0.875

# ----------------------------
# Data prep (port of your R logic)
# ----------------------------
def compute_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Port of your Shiny server preprocessing:
    - swing from PitchCall
    - in_zone + dist_from_center
    - count string, count_type
    - good/bad pitch and swing_points
    - filter NA swing & plate locations
    """
    df = df.copy()

    # Basic validation (fail early with friendly messages)
    required_cols = [
        "BatterTeam", "Batter", "PitchCall",
        "PlateLocSide", "PlateLocHeight",
        "Balls", "Strikes",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # (Team filtering is now handled in the UI, so we process whatever is passed in)

    # swing flag
    swing_map_1 = {"StrikeSwinging", "Foul", "InPlay"}
    swing_map_0 = {"BallCalled", "StrikeCalled"}
    df["swing"] = np.where(df["PitchCall"].isin(swing_map_1), 1,
                    np.where(df["PitchCall"].isin(swing_map_0), 0, np.nan))

    # in_zone + distance
    df["in_zone"] = (
        (df["PlateLocSide"] >= X_MIN) & (df["PlateLocSide"] <= X_MAX) &
        (df["PlateLocHeight"] >= Z_MIN) & (df["PlateLocHeight"] <= Z_MAX)
    )
    df["dist_from_center"] = np.sqrt((df["PlateLocSide"] ** 2) + ((df["PlateLocHeight"] - 2.5) ** 2))

    # count + type
    df["count"] = df["Balls"].astype(int).astype(str) + "-" + df["Strikes"].astype(int).astype(str)

    def count_type(c: str) -> str:
        if c in HITTER_COUNTS:
            return "hitter"
        if c in PITCHER_COUNTS:
            return "pitcher"
        return "neutral"

    df["count_type"] = df["count"].apply(count_type)

    # good/bad pitch (your Shiny server uses only good_pitch and bad_pitch)
    df["good_pitch"] = df["in_zone"] & (df["dist_from_center"] <= GOOD_PITCH_RADIUS)
    df["bad_pitch"] = ~df["in_zone"]

    # swing_points (matches Shiny server section where "swing == 0 & bad_pitch ~ 2")
    def points(row) -> float:
        ct = row["count_type"]
        swing = row["swing"]
        good = row["good_pitch"]
        bad = row["bad_pitch"]

        if pd.isna(swing):
            return 0.0

        # Hitter's count
        if ct == "hitter" and swing == 1 and good:
            return 3
        if ct == "hitter" and swing == 0 and good:
            return -2
        if ct == "hitter" and swing == 1 and bad:
            return -3

        # Pitcher's count
        if ct == "pitcher" and swing == 1 and good:
            return 2
        if ct == "pitcher" and swing == 0 and good:
            return -1
        if ct == "pitcher" and swing == 1 and bad:
            return -1

        # Neutral count
        if ct == "neutral" and swing == 1 and good:
            return 2
        if ct == "neutral" and swing == 0 and good:
            return -1
        if ct == "neutral" and swing == 1 and bad:
            return -2

        # Reward for taking bad pitches
        if swing == 0 and bad:
            # Uncompetitive pitches (way out of zone, e.g. > 1.5 ft from center) get 1 point
            # "Competitive" bad pitches (close) get 2 points
            if row["dist_from_center"] > 1.5:
                return 1
            return 2

        return 0

    df["swing_points"] = df.apply(points, axis=1)

    # Filter NAs like your R code
    df = df.dropna(subset=["swing", "PlateLocSide", "PlateLocHeight"]).copy()

    return df


def zone_label_positions(x_breaks: np.ndarray, z_breaks: np.ndarray) -> pd.DataFrame:
    # 1..9 labels arranged top-left to bottom-right (same as R: arrange(desc(zone_y), zone_x))
    xs = (x_breaks[:-1] + x_breaks[1:]) / 2
    zs = (z_breaks[:-1] + z_breaks[1:]) / 2
    zs = zs[::-1]  # reverse (top row first)
    labels = []
    k = 1
    for z in zs:
        for x in xs:
            labels.append((k, x, z))
            k += 1
    return pd.DataFrame(labels, columns=["label", "x", "y"])


def build_zone_plot(player_df: pd.DataFrame, pov: str = "Pitcher") -> go.Figure:
    labels_df = zone_label_positions(X_BREAKS, Z_BREAKS)

    # Scatter points colored by swing (0 take, 1 swing)
    color_map = {0: "red", 1: "blue"}
    player_df = player_df.copy()
    
    # Adjust for POV
    if pov == "Batter":
        player_df["PlateLocSide"] = player_df["PlateLocSide"] * -1
        
    player_df["decision"] = player_df["swing"].astype(int)

    fig = go.Figure()

    # Draw Home Plate
    # 17 inches = 1.4167 ft.
    half_plate = 17 / 12 / 2
    # Define a visual "depth" for the plate on the ground (e.g. 0.25 ft)
    plate_depth = 0.25
    # The sides go back 8.5 inches (half of total depth 17 inches)
    slant_start = plate_depth * (8.5 / 17)

    # Define plate coordinates based on POV
    if pov == "Batter":
        # Batter/Catcher View: Point is closest (y=0), Flat is furthest (y=plate_depth)
        # However, standard plots usually just invert the shape vertically if "looking from behind"?
        # Actually, "Point" of home plate points to Catcher.
        # If Catcher View means Catcher is at bottom of screen:
        # Point should be at bottom (y=0). Flat at top (y=depth).
        
        # Invert the vertical logic: 
        # But wait, y=0 is the "front" of the strike zone (closest to pitcher).
        # The plate sits on the ground.
        # If we are Catcher, looking at Pitcher:
        # The point is closest to us. The flat is further away (towards pitcher).
        # In a 2D projection where y is height, and we look "into" the zone:
        # Objects further away are usually obscured or just placed relative to ground Z.
        # Let's keep the plate shape logic consistent with "Point is Back" for Pitcher, "Point is Front" for Batter?
        # Standard MLB Gameday (Catcher View) shows Point at Bottom.
        
        # Let's try to draw Point at Y=0.
        pass
        # Actually, visually on a 2D plot, the "Standard" plate graphic (Pentagon pointing down) is iconic.
        # Pitcher View (Pentagon pointing Up/Away).
        
        # Batter/Catcher View -> Point Down (at y=0 or even negative?)
        # Let's keep it simple and just mirror the shape vertically if needed.
        # But y needs to stay near 0.
        
        # Let's go with "Point Close" (y=0) for Batter View. 
        # x: [0, half, half, -half, -half, 0]
        # y: [0, slant, depth, depth, slant, 0]
        plate_x = [0, half_plate, half_plate, -half_plate, -half_plate, 0]
        plate_y = [0, slant_start, plate_depth, plate_depth, slant_start, 0]

    else:
        # Pitcher Perspective (Original)
        # Flat close (y=0), Point away.
        plate_x = [-half_plate, half_plate, half_plate, 0, -half_plate, -half_plate]
        plate_y = [0, 0, slant_start, plate_depth, slant_start, 0]

    fig.add_trace(go.Scatter(
        x=plate_x,
        y=plate_y,
        mode="lines",
        fill="toself",
        fillcolor="lightgray",
        line=dict(color="black", width=2),
        name="Home Plate",
        hoverinfo="skip",
        showlegend=False
    ))

    for dec_val, name in [(0, "Take"), (1, "Swing")]:
        sub = player_df[player_df["decision"] == dec_val]
        
        # Clean up PitchCall for tooltip
        calls = sub["PitchCall"].fillna("Unknown")
        
        # If InPlay, try to use PlayResult if available
        if "PlayResult" in sub.columns:
            # Mask for InPlay records
            in_play_mask = calls == "InPlay"
            # distinct fallback for missing PlayResult
            calls = np.where(in_play_mask, sub["PlayResult"].fillna("In Play"), calls)
            
        calls = pd.Series(calls).replace({
            "StrikeCalled": "Called Strike",
            "BallCalled": "Ball",
            "StrikeSwinging": "Whiff", 
            "Foul": "Foul",
            "InPlay": "In Play" # fallback if not replaced above or PlayResult missing
        })
        
        # Stack count and call for customdata [[count, call], [count, call]...]
        custom_data = np.stack((sub["count"], calls), axis=-1)

        fig.add_trace(
            go.Scatter(
                x=sub["PlateLocSide"],
                y=sub["PlateLocHeight"],
                mode="markers",
                name=name,
                marker=dict(size=7, opacity=0.6, color=color_map[dec_val]),
                customdata=custom_data,
                # customdata[0] is count, customdata[1] is call result
                hovertemplate=f"<b>{name}</b><br>Side=%{{x:.2f}}<br>Height=%{{y:.2f}}<br>Count=%{{customdata[0]}}<br>Result=%{{customdata[1]}}<extra></extra>",
            )
        )

    # Grid lines
    for x in X_BREAKS:
        fig.add_shape(type="line", x0=x, x1=x, y0=Z_MIN, y1=Z_MAX, line=dict(color="gray", width=2))
    for y in Z_BREAKS:
        fig.add_shape(type="line", x0=X_MIN, x1=X_MAX, y0=y, y1=y, line=dict(color="gray", width=2))

    # Outer strike zone rectangle
    fig.add_shape(type="rect", x0=X_MIN, x1=X_MAX, y0=Z_MIN, y1=Z_MAX, line=dict(color="black", width=3), fillcolor="rgba(0,0,0,0)")

    # Zone labels 1..9
    fig.add_trace(
        go.Scatter(
            x=labels_df["x"],
            y=labels_df["y"],
            mode="text",
            text=labels_df["label"].astype(str),
            textfont=dict(size=18, color="darkred"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Match your coord_fixed + xlim/ylim behavior
    fig.update_xaxes(range=[-2.5, 2.5], title_text="Plate Side")
    fig.update_yaxes(range=[0, 5], title_text="Plate Height", scaleanchor="x", scaleratio=1)

    fig.update_layout(
        template="simple_white",
        height=650,
        margin=dict(l=40, r=40, t=60, b=40),
        legend_title_text="Decision",
        legend=dict(
            font=dict(size=16),
            title=dict(font=dict(size=18)),
            itemsizing='constant'
        )
    )

    return fig


def score_by_count_table(player_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        player_df.groupby("count", as_index=False)
        .agg(total_points=("swing_points", "sum"), pitch_count=("swing_points", "size"))
    )
    # ensure all counts exist
    all_df = pd.DataFrame({"count": ALL_COUNTS})
    out = all_df.merge(agg, on="count", how="left").fillna({"total_points": 0, "pitch_count": 0})
    out["total_points"] = out["total_points"].astype(float)
    out["pitch_count"] = out["pitch_count"].astype(int)
    out["average_points"] = np.where(out["pitch_count"] > 0, (out["total_points"] / out["pitch_count"]).round(2), 0.0)
    out = out.sort_values("average_points", ascending=False)

    out = out.rename(columns={
        "count": "Count",
        "total_points": "Total Points",
        "pitch_count": "Pitch Count",
        "average_points": "Average Points",
    })
    # Reorder columns as requested
    out = out[["Count", "Average Points", "Pitch Count", "Total Points"]]
    return out


def leaderboard_table(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("Batter", as_index=False)
        .agg(TotalPoints=("swing_points", "sum"), PitchCount=("swing_points", "size"))
    )
    out["Average Points"] = (out["TotalPoints"] / out["PitchCount"]).round(2)
    out = out.sort_values("Average Points", ascending=False)
    
    # Rename and reorder columns
    out = out.rename(columns={
        "PitchCount": "Pitches Seen",
        "TotalPoints": "Total Points"
    })
    out = out[["Batter", "Average Points", "Pitches Seen", "Total Points"]]
    return out


def calculate_random_baseline(df: pd.DataFrame, n_simulations: int = 5) -> float:
    """
    Simulates random 50/50 swing decisions on the dataset to determine
    a 'coin-flip' baseline score.
    """
    # Create a copy to manipulate
    sim_df = df.copy()
    
    avg_scores = []
    
    # We need access to the inner 'points' logic, but 'points' is defined inside compute_fields.
    # To keep it simple, we'll replicate the core scoring logic vector-wise or reuse compute_fields logic?
    # Reuse isn't easy because compute_fields does a lot.
    # Let's vectorize the scoring logic for speed and re-use.
    
    # Pre-calculate fields that don't change
    # We need: count_type, good_pitch, bad_pitch, dist_from_center
    # These are already in 'df' coming from compute_fields.
    
    # Vectorized scoring function
    def run_sim():
        # Always Swing (Aggressive Mode)
        sim_swing = np.ones(len(sim_df), dtype=int)
        
        # Conditions
        ct_hitter = sim_df["count_type"] == "hitter"
        ct_pitcher = sim_df["count_type"] == "pitcher"
        ct_neutral = sim_df["count_type"] == "neutral"
        
        good = sim_df["good_pitch"]
        bad = sim_df["bad_pitch"]
        dist = sim_df["dist_from_center"]
        
        scores = np.zeros(len(sim_df))
        
        # Hitter Count
        scores[ct_hitter & (sim_swing == 1) & good] = 3
        scores[ct_hitter & (sim_swing == 0) & good] = -2
        scores[ct_hitter & (sim_swing == 1) & bad] = -3
        
        # Pitcher Count
        scores[ct_pitcher & (sim_swing == 1) & good] = 2
        scores[ct_pitcher & (sim_swing == 0) & good] = -1
        scores[ct_pitcher & (sim_swing == 1) & bad] = -1
        
        # Neutral Count
        scores[ct_neutral & (sim_swing == 1) & good] = 2
        scores[ct_neutral & (sim_swing == 0) & good] = -1
        scores[ct_neutral & (sim_swing == 1) & bad] = -2
        
        # Takes on bad pitches
        take_bad_mask = (sim_swing == 0) & bad
        # Apply competitive/uncompetitive logic
        # If dist > 1.5, score 1, else score 2
        uncomp_mask = take_bad_mask & (dist > 1.5)
        comp_mask = take_bad_mask & (dist <= 1.5)
        
        scores[uncomp_mask] = 1
        scores[comp_mask] = 2
        
        return scores.mean()

    for _ in range(n_simulations):
        avg_scores.append(run_sim())
        
    return float(np.mean(avg_scores))


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Swing Decision UI", layout="wide")

st.title("Swing Decisions")

# Defines layout containers for Sidebar
with st.sidebar:
    st.header("Controls")
    team_container = st.empty()
    nof_players_container = st.empty()
    batter_container = st.empty()
    
    st.divider()
    # Interpretation goes here
    interpret_container = st.container()

    st.divider()
    go_leaderboard = st.button("Show Leaderboard")
    
    st.divider()
    # PITCH VIEW (POV)
    view_pov = st.radio("View Perspective", ["Pitcher", "Batter"], horizontal=True)

    st.divider()
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV (Yakkertech export / master file)", type=["csv"])
    default_path = st.text_input("...or local CSV path (IDE use)", value="UNF Master File 25-26.csv")


# Load data
raw_df: Optional[pd.DataFrame] = None
load_error: Optional[str] = None

try:
    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
    else:
        # For now, try local path; later this can become env var / deployment storage
        raw_df = pd.read_csv(default_path)
except Exception as e:
    load_error = str(e)

if load_error:
    st.error("Upload a CSV file to begin. Must be in the format of Yakkertech exports.")
    # st.code(load_error)
    st.stop()
    
# Populate Interpretation with Placeholder
with interpret_container:
    with st.expander("How to interpret Score?"):
        # Plot visual number line (Vertical)
        fig_scale = go.Figure()
        
        # Scale Points
        points = [
            (1.00,  "Elite", "green"),
            (0.50,  "Adequate", "lightgreen"),
            (0.35,  "Poor / Always Take", "yellow"),
            (0.00,  "Truly Random", "orange"),
            (-0.35, "Always Swing", "red")
        ]
        
        for val, desc, color in points:
            fig_scale.add_trace(go.Scatter(
                x=[0], y=[val],
                mode='markers+text',
                marker=dict(size=15, color=color, line=dict(width=1, color='black')),
                text=[f"<b>{val}</b>: {desc}"],
                textposition="middle right",
                hoverinfo='skip',
                showlegend=False
            ))

        # Main vertical line
        fig_scale.add_shape(type="line", x0=0, x1=0, y0=-0.45, y1=1.1, line=dict(color="gray", width=3))
        
        fig_scale.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(
                range=[-0.1, 1], # Space for text on the right
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                fixedrange=True
            ),
            yaxis=dict(
                range=[-0.45, 1.1], 
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                fixedrange=True
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig_scale, use_container_width=True, config={'displayModeBar': False})

        st.markdown("""
        **Benchmark Descriptions:**
        
        *   **1.00 (Elite):** Takes uncompetitive, Swings at Heart, 50/50 on Edge/Chase.
        *   **0.5 - 1.0:** Taking uncompetitive pitches, decision made on competitive pitches. Skill in making this decision determines the score in this range.
        *   **0.50 (Adequate):** Takes uncompetitive, 50/50 on competitive pitches.
        *   **0.35 (Pure Passive):** Taking every single pitch. This is unrealistic, but used as a baseline.
        *   **0.00 (Random):** Pure 50/50 coin flip on every single pitch as to whether to swing or take.
        *   **-0.35 (Pure Aggressive):** Swinging at every single pitch. This is unrealistic, but used as a baseline.
        """)

# Team Selection (Dynamic)
if "BatterTeam" in raw_df.columns:
    all_teams = sorted(raw_df["BatterTeam"].dropna().unique().tolist())
    # Default to "NOF_OSP" if present, otherwise first available
    default_ix = 0
    if "NOF_OSP" in all_teams:
        default_ix = all_teams.index("NOF_OSP")

    with team_container:
        selected_team = st.selectbox(
            "Filter by Team:", 
            options=all_teams, 
            index=default_ix
        )
    # Filter the raw data before processing
    raw_df = raw_df[raw_df["BatterTeam"] == selected_team].copy()

    # Extra guardrail for UNF team labeling:
    # when viewing NOF_OSP, allow removing players that should not be included.
    if selected_team == "NOF_OSP" and "Batter" in raw_df.columns:
        nof_players = sorted(raw_df["Batter"].dropna().unique().tolist())
        with nof_players_container:
            selected_nof_players = st.multiselect(
                "UNF Players to include:",
                options=nof_players,
                default=nof_players,
                help="Unselect any incorrectly labeled non-UNF players to exclude them from all views."
            )

        if not selected_nof_players:
            st.warning("No UNF players selected. Choose at least one player to continue.")
            st.stop()

        raw_df = raw_df[raw_df["Batter"].isin(selected_nof_players)].copy()

# Compute fields
try:
    df = compute_fields(raw_df)
except Exception as e:
    st.error("CSV loaded, but preprocessing failed (likely missing column names expected by the R app).")
    st.code(str(e))
    st.stop()

# Batter selection
batters = sorted(df["Batter"].dropna().unique().tolist())
if not batters:
    st.warning("No batters available after current team/player filters.")
    st.stop()

with batter_container:
    player = st.selectbox("Select Batter:", options=batters)

# Tabs
tab_names = ["Swing vs Take by Zone", "Swing Decision Score by Count", "Leaderboard"]

# Manage tab selection state
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = tab_names[0]

# If button clicked, jump to leaderboard
if go_leaderboard:
    st.session_state["active_tab"] = "Leaderboard"

# Use radio button as fake tabs to allow programmatic switching
selected_tab = st.radio(
    "Navigation", 
    tab_names, 
    horizontal=True, 
    key="active_tab",
    label_visibility="collapsed"
)

# Tab 1
if selected_tab == tab_names[0]:
    st.subheader(f"Swing vs Take by Zone: {player}")
    player_df = df[df["Batter"] == player].copy()

    overall_avg = float(np.round(player_df["swing_points"].mean(), 2)) if len(player_df) else 0.0
    st.markdown(
        f"<div style='text-align:center;font-size:20px;font-weight:700;'>Swing Decision Score: {overall_avg}</div>",
        unsafe_allow_html=True,
    )

    fig = build_zone_plot(player_df, pov=view_pov)
    st.plotly_chart(fig, use_container_width=True)

# Tab 2
elif selected_tab == tab_names[1]:
    st.subheader("Swing Decision Score by Count")
    player_df = df[df["Batter"] == player].copy()
    
    overall_avg = float(np.round(player_df["swing_points"].mean(), 2)) if len(player_df) else 0.0
    st.markdown(
        f"<div style='text-align:center;font-size:20px;font-weight:700;margin-bottom:15px;'>Swing Decision Score: {overall_avg}</div>",
        unsafe_allow_html=True,
    )

    table_df = score_by_count_table(player_df)
    styled_table = (
        table_df.style
        .background_gradient(subset=["Average Points"], cmap="RdYlGn", vmin=-3, vmax=3)
        .format({"Average Points": "{:.2f}", "Total Points": "{:.0f}", "Pitch Count": "{:.0f}"})
    )
    
    st.dataframe(
        styled_table, 
        use_container_width=True, 
        hide_index=True,
        height=(len(table_df) + 1) * 35 + 3
    )

# Tab 3
elif selected_tab == tab_names[2]:
    st.subheader("Leaderboard")
    
    team_avg = float(np.round(df["swing_points"].mean(), 2)) if len(df) else 0.0
    st.markdown(
        f"<div style='text-align:center;font-size:20px;font-weight:700;margin-bottom:15px;'>Team Average Swing Decision Score: {team_avg}</div>",
        unsafe_allow_html=True,
    )

    lb_df = leaderboard_table(df)
    
    # Calculate height based on rows (approx 35px per row + 35px header + buffer)
    # Cap at some reasonable max height if desired (e.g. 1000px), or let it grow indefinitely
    table_height = (len(lb_df) + 1) * 35 + 3
    
    st.dataframe(
        lb_df, 
        use_container_width=True, 
        hide_index=True, 
        height=table_height
    )



