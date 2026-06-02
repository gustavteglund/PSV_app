import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from mplsoccer import Pitch

DATA_FOLDER = "Data/2024"
STREAMLIT_FOLDER = f"{DATA_FOLDER}/streamlit"
PAGE_TITLE = "Pass Suppression Value"
ACCENT = "#22c55e"
ACCENT_ALT = "#38bdf8"
INK = "#0f172a"
MUTED = "#475569"
SURFACE = "#f8fafc"
CARD = "#ffffff"
GRID = "#cbd5e1"
PITCH_GREEN = "#124d36"

PLOTTING_COLUMNS = [
    "player_id",
    "PSV",
    "PSVrel",
    "ProbsTot",
    "PSVraw",
    "PSVrelraw",
    "ProbsTotraw",
    "Probs",
    "Probs_cf",
    "xT",
    "match_id",
    "Frame",
    "Receiver_ids",
    "Pass_id",
]

METRIC_COLUMN_MAP = {
    "Pass Suppression Value": "PSVraw",
    "Relative Pass Suppression Value": "PSVrelraw",
    "Pass Suppression": "ProbsTotraw",
    "Probabilities": "Probs",
    "Counterfactual Probabilities": "Probs_cf",
    "Expected Threat (xT)": "xT",
}

st.set_page_config(
    page_title=PAGE_TITLE,
    layout="wide",
    initial_sidebar_state="collapsed",
)


def apply_app_theme():
    st.markdown(
        f"""
        <style>
            .stApp {{
                background:
                    radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 28%),
                    radial-gradient(circle at top right, rgba(34, 197, 94, 0.18), transparent 24%),
                    linear-gradient(180deg, #f8fbff 0%, #f4f7fb 42%, #eef3f9 100%);
            }}
            .block-container {{
                padding-top: 1.8rem;
                padding-bottom: 2.5rem;
                max-width: 1380px;
            }}
            h1, h2, h3 {{
                color: {INK};
                letter-spacing: -0.02em;
            }}
            div[data-testid="stTabs"] button {{
                font-weight: 600;
            }}
            div[data-testid="stMetric"] {{
                background: rgba(255, 255, 255, 0.86);
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 18px;
                padding: 0.7rem 0.9rem;
                box-shadow: 0 18px 50px rgba(15, 23, 42, 0.05);
            }}
            div[data-testid="stDataFrame"] {{
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 18px;
                padding: 0.35rem;
                box-shadow: 0 18px 50px rgba(15, 23, 42, 0.05);
            }}
            .hero-card {{
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.94));
                color: white;
                border-radius: 24px;
                padding: 1.4rem 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 20px 65px rgba(15, 23, 42, 0.18);
            }}
            .hero-card p {{
                color: rgba(226, 232, 240, 0.9);
                margin: 0.35rem 0 0;
            }}
            .section-card {{
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 22px;
                padding: 1rem 1.1rem 0.9rem;
                margin: 0.55rem 0 1rem;
                box-shadow: 0 18px 50px rgba(15, 23, 42, 0.05);
                backdrop-filter: blur(10px);
            }}
            .section-card h3 {{
                margin-bottom: 0.2rem;
            }}
            .section-card p {{
                color: {MUTED};
                margin-bottom: 0;
            }}
            .small-note {{
                color: {MUTED};
                font-size: 0.92rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_intro(title, subtitle):
    st.markdown(
        f"""
        <div class="section-card">
            <h3>{title}</h3>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_percentile(value):
    return None if pd.isna(value) else f"{value:.0f}th percentile"


def prepare_table(df, decimals_map=None):
    table = df.copy()
    if decimals_map:
        for col, decimals in decimals_map.items():
            if col in table.columns:
                table[col] = table[col].round(decimals)
    return table


# -----------------------------
# Cached subset data loaders
# -----------------------------
@st.cache_data(ttl=10 * 60)
def get_selector_df():
    return pd.read_parquet(f"{STREAMLIT_FOLDER}/selector_df.parquet")


@st.cache_data(ttl=10 * 60)
def get_plotting_data():
    data = pd.read_parquet(
        f"{STREAMLIT_FOLDER}/plotting_data.parquet",
        columns=PLOTTING_COLUMNS,
    ).reset_index(drop=True)
    data["_subset_row"] = np.arange(len(data))
    return data


@st.cache_data(ttl=10 * 60)
def get_tracking_subset():
    return pd.read_parquet(
        f"{STREAMLIT_FOLDER}/tracking_subset.parquet",
        columns=["tracking_frame"],
    ).reset_index(drop=True)


@st.cache_data(ttl=10 * 60)
def get_meta(match_id):
    with open(f"{DATA_FOLDER}/meta/{match_id}.json") as f:
        return json.load(f)


@st.cache_data(ttl=10 * 60)
def get_match_resources(match_id):
    meta = get_meta(match_id)

    player_lookup = {
        p["id"]: {
            "number": p["number"],
            "team_id": p["team_id"],
            "short_name": p.get("short_name") or f"Player {p['id']}",
        }
        for p in meta["players"]
    }

    jersey_colors = {
        meta["home_team"]["id"]: meta["home_team_kit"]["jersey_color"],
        meta["away_team"]["id"]: meta["away_team_kit"]["jersey_color"],
    }

    number_colors = {
        meta["home_team"]["id"]: meta["home_team_kit"]["number_color"],
        meta["away_team"]["id"]: meta["away_team_kit"]["number_color"],
    }

    team_names = [meta["home_team"]["name"], meta["away_team"]["name"]]

    return player_lookup, jersey_colors, number_colors, team_names


@st.cache_data(ttl=10 * 60)
def get_pass_options():
    data = get_plotting_data()
    pass_options = (
        data.groupby(["match_id", "Pass_id"], as_index=False)
        .agg(
            PSV=("PSV", "mean"),
            PSVrel=("PSVrel", "mean"),
            ProbsTot=("ProbsTot", "mean"),
            frames=("Frame", "nunique"),
            centre_backs=("player_id", "nunique"),
        )
        .sort_values(["match_id", "Pass_id"])
        .reset_index(drop=True)
    )
    pass_options.insert(0, "example", np.arange(1, len(pass_options) + 1))
    return pass_options


def format_player_option(player_id, player_lookup):
    info = player_lookup.get(int(player_id), {})
    name = info.get("short_name", f"Player {player_id}")
    number = info.get("number")
    return f"{name} (#{number})" if number is not None else name


# -----------------------------
# Plotting
# -----------------------------
def plot_radar_updated(players_df, metrics):
    colors = ["#22c55e", "#0ea5e9", "#f97316", "#8b5cf6", "#ef4444", "#f59e0b"]
    labels = [
        "Pass Suppression Value",
        "Pass Suppression",
        "Relative Pass Suppression Value",
        "Pair Dominance",
    ]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(9.6, 7.6), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f3f7fb")

    for i, (_, row) in enumerate(players_df.iterrows()):
        values = row[metrics].values.astype(float)
        values = np.concatenate([values, [values[0]]])
        color = colors[i % len(colors)]

        ax.plot(angles, values, color=color, linewidth=2.8, label=row["Player"])
        ax.fill(angles, values, color=color, alpha=0.16)
        ax.scatter(angles[:-1], row[metrics].values.astype(float), color=color, s=42, zorder=3)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, color=INK)

    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        angle_deg = np.degrees(angle) % 360
        if angle_deg == 90:
            label.set_horizontalalignment("left")
        elif angle_deg == 270:
            label.set_horizontalalignment("right")

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels([f"{tick}" for tick in [20, 40, 60, 80, 100]], fontsize=9, color=MUTED)
    ax.grid(color=GRID, alpha=0.7, linewidth=0.9)
    ax.spines["polar"].set_visible(False)
    ax.set_title(
        "Positioning Profile",
        fontsize=18,
        fontweight="bold",
        color=INK,
        pad=24,
    )

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=min(3, len(players_df)),
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_color(INK)

    return fig, ax


def plot_frame(
    metric_value,
    receiver_pids,
    tracking_frame,
    player_lookup,
    jersey_colors,
    number_colors,
    team_names,
    selected_id,
    metric_label,
):
    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=105,
        pitch_width=68,
        pitch_color=PITCH_GREEN,
        line_color="#ecfeff",
        linewidth=1.4,
    )

    fig, ax = pitch.draw(figsize=(12.5, 8.2))
    fig.set_dpi(300)
    fig.patch.set_facecolor("#ecfdf5")
    ax.set_facecolor(PITCH_GREEN)
    receiver_idx = {pid: i for i, pid in enumerate(receiver_pids)}

    for p in tracking_frame["player_data"]:
        player_id = p["player_id"]
        info = player_lookup.get(player_id)
        if info is None:
            continue

        team_id = info["team_id"]
        kit_number = info["number"]
        x, y = p["x"], p["y"]
        jersey = jersey_colors[team_id]
        number_color = number_colors[team_id]

        if player_id == selected_id:
            pitch.scatter(
                x,
                y,
                s=820,
                color="#fde047",
                ax=ax,
                zorder=3,
                edgecolors="#0f172a",
                linewidths=1.8,
                alpha=0.96,
            )

        pitch.scatter(
            x,
            y,
            s=430,
            color=jersey,
            ax=ax,
            zorder=4,
            edgecolors="#0f172a",
            linewidths=1.2,
        )

        pitch.annotate(
            str(kit_number),
            (x, y),
            color=number_color,
            weight="bold",
            fontsize=10,
            ha="center",
            va="center",
            ax=ax,
            zorder=5,
        )

        if player_id in receiver_idx:
            node_idx = receiver_idx[player_id]
            if node_idx < len(metric_value):
                pass_prob = metric_value[node_idx] * 100
                pitch.annotate(
                    f"{pass_prob:.1f}",
                    (x, y + 2.4),
                    color=INK,
                    weight="bold",
                    fontsize=9.5,
                    ha="center",
                    va="center",
                    ax=ax,
                    zorder=7,
                    bbox=dict(
                        boxstyle="round,pad=0.24",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.88,
                    ),
                )

    ball = tracking_frame["ball_data"]
    pitch.scatter(
        ball["x"],
        ball["y"],
        s=120,
        color="white",
        edgecolors="#0f172a",
        ax=ax,
        zorder=6,
        linewidths=1.5,
    )

    ax.set_title(
        f"{team_names[0]} vs {team_names[1]}\n{metric_label} at {tracking_frame['timestamp']}",
        fontsize=16,
        color=INK,
        pad=16,
        fontweight="bold",
    )

    return fig, ax


apply_app_theme()

st.markdown(
    f"""
    <div class="hero-card">
        <h1 style="margin:0;">{PAGE_TITLE}</h1>
        <p>The Pass Suppression Value (PSV) metric quantifies how effectively a defender suppresses the passing options of the ball carrier. Explore defender profiles and inspect a curated set of pass-level suppression snapshots.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["Player Lookup", "Pass Examples"])


# -----------------------------
# Tab 1
# -----------------------------
with tab1:
    selector_df = get_selector_df()
    metrics_pct = [
        "PSV pct",
        "Pass Suppression pct",
        "Relative PSV pct",
        "Pair dominance pct",
    ]

    section_intro(
        "Player lookup",
        "Select one or more defenders to compare positioning profiles.",
    )

    selected_players_state = st.dataframe(
        prepare_table(
            selector_df,
            {
                "Minutes": 0,
                "PSV pct": 0,
                "Relative PSV pct": 0,
                "Pass Suppression pct": 0,
                "Pair dominance pct": 0,
            },
        ),
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        use_container_width=True,
        height=360,
    )

    selected_rows = selected_players_state["selection"]["rows"]
    selected_players = selector_df.iloc[selected_rows]

    if len(selected_players) == 0:
        st.info("Choose at least one defender from the table to show the profile view.")
    else:
        overview_cols = st.columns([1.3, 1.3, 1.1, 1.1])
        overview_cols[0].metric("Selected defenders", f"{len(selected_players)}")
        overview_cols[1].metric("Average minutes", f"{selected_players['Minutes'].mean():,.0f}")
        overview_cols[2].metric(
            "Top PSV percentile",
            safe_percentile(selected_players["PSV pct"].max()) or "-",
        )
        overview_cols[3].metric(
            "Top dominance percentile",
            safe_percentile(selected_players["Pair dominance pct"].max()) or "-",
        )

        plot_col, summary_col = st.columns([1.6, 1.0])
        with plot_col:
            fig, ax = plot_radar_updated(selected_players, metrics_pct)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with summary_col:
            section_intro(
                "Submetrics explanation",
                """
                <strong>Pass Suppression Value:</strong> Average value of reducing the likelihood of an opponent being targeted by the ball carrier.<br><br>
                <strong>Relative Pass Suppression Value:</strong> The defender's suppression of a passing option is relative to the likelihood of the pass.<br><br>
                <strong>Pass Suppression:</strong> Average reduction in the overall likelihood of an opponent being targeted by the ball carrier without considering how dangerous the passing option is.<br><br>
                <strong>Pair dominance:</strong> The average difference in PSV compared to the defender's centre back partners.
                """,
            )


# -----------------------------
# Tab 2
# -----------------------------
with tab2:
    plotting_data = get_plotting_data()
    tracking_subset = get_tracking_subset()

    section_intro(
        "Pass examples",
        "Choose one of the ten curated passes, select the centre back to evaluate and the metric to overlay, then scrub through the available frames.",
    )

    if len(plotting_data) != len(tracking_subset):
        st.error(
            "The plotting and tracking subset files do not have the same number of rows. "
            "Recreate the subset files before using the explorer."
        )
    else:
        pass_options = get_pass_options()
        pass_display = pass_options[
            ["example", "match_id", "Pass_id", "PSV", "PSVrel", "ProbsTot", "frames", "centre_backs"]
        ].rename(
            columns={
                "example": "Example",
                "match_id": "Match",
                "Pass_id": "Pass",
                "PSVrel": "Relative PSV",
                "ProbsTot": "Suppression",
                "frames": "Frames",
                "centre_backs": "Centre backs",
            }
        )

        selected_pass_state = st.dataframe(
            prepare_table(
                pass_display,
                {
                    "PSV": 3,
                    "Relative PSV": 3,
                    "Suppression": 3,
                    "Frames": 0,
                    "Centre backs": 0,
                },
            ),
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            use_container_width=True,
            height=360,
        )

        selected_pass_rows = selected_pass_state["selection"]["rows"]
        if len(selected_pass_rows) == 0:
            st.info("Select one pass from the table to show the pitch view.")
        else:
            selected_pass = pass_options.iloc[selected_pass_rows].iloc[0]
            selected_match_id = int(selected_pass["match_id"])
            selected_pass_id = int(selected_pass["Pass_id"])
            pass_frames = plotting_data.loc[
                plotting_data["match_id"].eq(selected_match_id)
                & plotting_data["Pass_id"].eq(selected_pass_id)
            ]

            player_lookup, jersey_colors, number_colors, team_names = get_match_resources(selected_match_id)
            centre_back_ids = sorted(pass_frames["player_id"].unique().tolist())

            control_cols = st.columns([1.0, 2.1])
            with control_cols[0]:
                selected_id = st.selectbox(
                    "Centre back",
                    centre_back_ids,
                    format_func=lambda player_id: format_player_option(player_id, player_lookup),
                )
            with control_cols[1]:
                metric_choice = st.radio(
                    "Displayed metric",
                    list(METRIC_COLUMN_MAP),
                    horizontal=True,
                )

            selected_frames = pass_frames.loc[pass_frames["player_id"].eq(selected_id)].sort_values("Frame")
            frames = sorted(selected_frames["Frame"].unique().tolist())
            if len(frames) == 1:
                frame_nr = frames[0]
                st.caption(f"Only one frame is available for this pass: {frame_nr}")
            else:
                frame_nr = st.select_slider(
                    "Frame",
                    options=frames,
                    value=frames[0],
                    key="example_frame_slider",
                )

            selected_frame_row = selected_frames.loc[selected_frames["Frame"].eq(frame_nr)].iloc[0]
            receiver_pids = selected_frame_row["Receiver_ids"]
            metric_value = selected_frame_row[METRIC_COLUMN_MAP[metric_choice]]
            tracking_frame = tracking_subset.iloc[int(selected_frame_row["_subset_row"])]["tracking_frame"]

            top_cols = st.columns(4)
            top_cols[0].metric("Match", f"{selected_match_id}")
            top_cols[1].metric("Pass ID", f"{selected_pass_id}")
            top_cols[2].metric("Frames in pass", f"{len(frames)}")
            top_cols[3].metric("Current frame", f"{frame_nr}")

            st.markdown(
                '<p class="small-note">The labels above potential receivers show the currently selected metric (multiplied by 100) for that frame. The highlighted player is the selected centre back.</p>',
                unsafe_allow_html=True,
            )

            fig, ax = plot_frame(
                metric_value=metric_value,
                receiver_pids=receiver_pids,
                tracking_frame=tracking_frame,
                player_lookup=player_lookup,
                jersey_colors=jersey_colors,
                number_colors=number_colors,
                team_names=team_names,
                selected_id=selected_id,
                metric_label=metric_choice,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
