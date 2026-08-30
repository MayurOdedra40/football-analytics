"""
Stage 6: reusable snapshot/trajectory extraction for the visualisation
notebook (notebooks/02_pass_visualisation.ipynb).

pass_candidates.parquet only keeps the CANDIDATES (a passer's own
teammates) plus derived features -- it never stored the opponents' raw
positions/velocities, since Stage 3 only needed opponent-derived aggregate
features (dist_nearest_opponent, etc.), not their raw coordinates. Drawing
"all 22 players" for a single pass needs those raw per-player positions
back, so this module re-runs the exact same synchronisation + attack-
normalisation machinery from src/build_dataset.py (MatchPositions,
sync_nearest/refined, attack_flip) for one pass at a time -- what gets
drawn is the same "pass moment" Stage 3/4/5 trained and evaluated on, not a
separately re-derived approximation of it.
"""

import numpy as np
import pandas as pd

from src.build_dataset import MatchPositions, attack_flip, SYNC_METHOD

OWN_COLOR = "#d62728"    # passer's team
OPP_COLOR = "#1f77b4"    # opponents
PASSER_COLOR = "gold"
RECIPIENT_OUTLINE = "lime"


def get_pass_snapshot(pass_row, match_pos, team_of_in_match, second_half_cutoff, team_left_by_match_half,
                       sync_method=SYNC_METHOD):
    """Full on-pitch snapshot for one pass (typically ~22 players: up to 11
    on the passer's team incl. the passer, up to 11 opponents), each with
    attack-normalised (x, y) and (vx, vy) -- same sync method and rotation
    Stage 3 used for its features. Returns (snapshot_df, synced_ts, flip)."""
    match_id = pass_row["match_id"]
    passer_id = pass_row["player_id"]
    team_id = pass_row["team_id"]
    event_ts_naive = pass_row["timestamp"].tz_convert("UTC").tz_localize(None)

    if sync_method == "nearest":
        synced_ts = match_pos.sync_nearest(event_ts_naive)
    else:
        synced_ts = match_pos.sync_refined(event_ts_naive, passer_id)

    flip = attack_flip(match_id, pass_row["timestamp"], team_id, second_half_cutoff, team_left_by_match_half)

    rows = []
    for pid, t in team_of_in_match.items():
        if not match_pos.on_pitch(pid, synced_ts):
            continue
        pv = match_pos.smoothed_pos_vel(pid, synced_ts)
        if pv is None:
            continue
        x, y, vx, vy = pv
        rows.append({
            "person_id": pid,
            "is_own_team": t == team_id,
            "is_passer": pid == passer_id,
            "is_recipient": pid == pass_row["recipient_id"],
            "x": x * flip, "y": y * flip,
            "vx": vx * flip, "vy": vy * flip,
        })
    return pd.DataFrame(rows), synced_ts, flip


def get_trajectories(person_ids, match_pos, synced_ts, flip, window_s=3.0):
    """Raw (unsmoothed -- this is for plotting a trail, not a model input)
    trajectory for each person_id, from (synced_ts - window_s) up to and
    including synced_ts. We still never draw anything after the pass
    moment, matching the modelling pipeline's own leakage rule, even
    though nothing here feeds back into training."""
    trails = {}
    target = np.datetime64(synced_ts)
    lo = target - np.timedelta64(int(window_s * 1000), "ms")
    for pid in person_ids:
        arr = match_pos.person_arr.get(pid)
        if arr is None:
            continue
        i0 = np.searchsorted(arr["t"], lo)
        i1 = np.searchsorted(arr["t"], target, side="right")
        trails[pid] = (arr["x"][i0:i1] * flip, arr["y"][i0:i1] * flip)
    return trails


# --- plotting helpers (kept here so the notebook cells stay short) --------

def draw_players(ax, snap, marker_size=220, show_velocity=True, velocity_scale=1.0):
    """Own team (red) / opponents (blue) dots, passer as a gold star, an
    optional velocity arrow per player. The building block every panel in
    the notebook starts from."""
    own = snap[snap["is_own_team"] & ~snap["is_passer"]]
    opp = snap[~snap["is_own_team"]]
    passer = snap[snap["is_passer"]].iloc[0]

    ax.scatter(own["x"], own["y"], s=marker_size, color=OWN_COLOR, edgecolor="black", zorder=4)
    ax.scatter(opp["x"], opp["y"], s=marker_size, color=OPP_COLOR, edgecolor="black", zorder=4)
    ax.scatter([passer["x"]], [passer["y"]], s=marker_size * 1.4, facecolor=PASSER_COLOR,
               edgecolor="black", marker="*", zorder=5)

    if show_velocity:
        for _, r in snap.iterrows():
            ax.annotate("", xy=(r["x"] + r["vx"] * velocity_scale, r["y"] + r["vy"] * velocity_scale),
                        xytext=(r["x"], r["y"]),
                        arrowprops=dict(arrowstyle="-|>", color="green", lw=1, alpha=0.7), zorder=3)
    return passer


def draw_pass_arrow(ax, snap):
    passer = snap[snap["is_passer"]].iloc[0]
    recipient = snap[snap["is_recipient"]].iloc[0]
    ax.annotate("", xy=(recipient["x"], recipient["y"]), xytext=(passer["x"], passer["y"]),
                arrowprops=dict(arrowstyle="->", color="black", lw=2), zorder=6)


def draw_trails(ax, snap, trails, alpha=0.35, linewidth=2):
    for _, r in snap.iterrows():
        tx, ty = trails.get(r["person_id"], (np.array([]), np.array([])))
        color = OWN_COLOR if r["is_own_team"] else OPP_COLOR
        ax.plot(tx, ty, color=color, alpha=alpha, linewidth=linewidth, zorder=2)


def draw_prediction_overlay(ax, snap, pred_df, marker_size_scale=2000, marker_size_base=60):
    """snap: this pass's full on-pitch snapshot (from get_pass_snapshot).
    pred_df: this pass's rows from a predict_with_model() output (pass_id,
    candidate_id, prob, is_recipient). Own-team non-passer players are
    coloured/sized by predicted probability; the true recipient gets a
    lime outline regardless of whether the model got it right."""
    opp = snap[~snap["is_own_team"]]
    passer = snap[snap["is_passer"]].iloc[0]
    ax.scatter(opp["x"], opp["y"], s=180, color=OPP_COLOR, edgecolor="black", alpha=0.6, zorder=3)
    ax.scatter([passer["x"]], [passer["y"]], s=320, facecolor=PASSER_COLOR, edgecolor="black",
               marker="*", zorder=5)

    candidates = snap[snap["is_own_team"] & ~snap["is_passer"]].drop(columns=["is_recipient"]).merge(
        pred_df, left_on="person_id", right_on="candidate_id")
    sc = ax.scatter(candidates["x"], candidates["y"], s=candidates["prob"] * marker_size_scale + marker_size_base,
                     c=candidates["prob"], cmap="Reds", vmin=0, vmax=1, edgecolor="black", zorder=4)
    true_recip = candidates[candidates["is_recipient"] == 1]
    ax.scatter(true_recip["x"], true_recip["y"], s=true_recip["prob"] * marker_size_scale + marker_size_base + 150,
               facecolor="none", edgecolor=RECIPIENT_OUTLINE, linewidth=3, zorder=6)
    return sc
