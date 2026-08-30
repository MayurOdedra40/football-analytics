"""
Stage 3: build the pass-receiver-prediction modelling dataset.

Reads data/processed/{events,players}.csv and positions/<match>.parquet
(Stage 1 outputs), and writes data/processed/pass_candidates.parquet: one
row per (pass, candidate teammate).

*** A finding that changes the spec's stated design, verified before writing
any of this (see README decisions table): `recipient_id` is only a valid
same-team "intent" label on COMPLETED passes. Checked all 5,903 passes with
a tagged recipient -- every unsuccessful one has an OPPONENT as recipient
(the interceptor), every successful one has a teammate. There is no intent
label for failed passes anywhere in this data, so -- confirmed with the
user -- this dataset is completed passes only, not "both" as the spec
originally asked for. ***

Run: `python -m src.build_dataset` (from repo root, venv active).
"""

import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import savgol_filter

from src.parse import CONFIG, OUT_DIR

SYNC_METHOD = CONFIG["sync_method"]  # "nearest" or "refined"
SEED = CONFIG["seed"]

# Position frames are stored ~8.33Hz apart (Stage 1's KEEP_EVERY=3 of 25Hz).
# A 5-frame causal window is therefore ~480ms of history -- short enough to
# reflect the player's *current* movement, long enough for savgol_filter's
# polyorder=2 to fit through actual noise rather than 2 points exactly.
VELOCITY_WINDOW = 5

# refined sync: search +/-3s around the tagged EventTime (per spec), scoring
# each candidate ball frame by ball-to-passer distance plus a time-offset
# penalty. LAMBDA (m/s) converts the offset into the same units as the
# spatial term -- picked as a plausible "cost of drifting away from the
# tagged time" (roughly a brisk jogging speed) and checked for coarse
# sensitivity (2-5 -> mean sync error moved by <1m on two test matches, so
# it's not a fragile choice), not exhaustively tuned against ground truth we
# don't have.
REFINED_WINDOW_S = 3.0
REFINED_LAMBDA = 2.0

RNG = np.random.default_rng(SEED)


# --- loading + the pass-selection funnel ---------------------------------

def load_base_tables():
    events = pd.read_csv(os.path.join(OUT_DIR, "events.csv"), low_memory=False)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, format="ISO8601")
    players = pd.read_csv(os.path.join(OUT_DIR, "players.csv"))
    matches = pd.read_csv(os.path.join(OUT_DIR, "matches.csv"))
    return events, players, matches


def select_passes(events, players):
    """The filter funnel from the spec, applied in order, logging what each
    step drops. Returns the surviving passes plus a dict of drop counts."""
    log = {}

    is_pass = events["event_type_path"].str.endswith(("_Pass", "_Cross"))
    step = events[is_pass].copy()
    log["1_pass_or_cross_events"] = len(step)

    step = step[step["recipient_id"].notna()].copy()
    log["2_has_recipient_id"] = len(step)

    # team lookup: (match_id, person_id) -> team_id
    team_of = players.set_index(["match_id", "person_id"])["team_id"]
    step["recipient_team"] = [team_of.get((m, r)) for m, r in zip(step["match_id"], step["recipient_id"])]
    same_team = step["team_id"] == step["recipient_team"]
    n_dropped_cross_team = (~same_team).sum()
    step = step[same_team].copy()
    log["3_same_team (== completed passes, see module docstring)"] = len(step)
    log["  -- dropped as cross-team (recipient = interceptor, not intent)"] = int(n_dropped_cross_team)

    return step, log


def add_on_pitch_filter(passes, match_ids):
    """Drops passes where the passer or recipient isn't covered by tracked
    position data at the event's tagged timestamp (Stage 1 confirmed a
    player's tracked span runs from kickoff/sub-on to full-time/sub-off/red
    card, so "covered" == "on the pitch")."""
    spans = {}
    for m in match_ids:
        tbl = pq.read_table(os.path.join(OUT_DIR, "positions", f"{m}.parquet"),
                             columns=["person_id", "timestamp"]).to_pandas()
        tbl["timestamp"] = pd.to_datetime(tbl["timestamp"], utc=True, format="ISO8601")
        spans[m] = tbl.groupby("person_id")["timestamp"].agg(["min", "max"])

    def on_pitch(row, col):
        g = spans[row["match_id"]]
        pid = row[col]
        if pid not in g.index:
            return False
        return g.loc[pid, "min"] <= row["timestamp"] <= g.loc[pid, "max"]

    passer_ok = passes.apply(lambda r: on_pitch(r, "player_id"), axis=1)
    recipient_ok = passes.apply(lambda r: on_pitch(r, "recipient_id"), axis=1)
    kept = passes[passer_ok & recipient_ok].copy()
    return kept, len(passes) - len(kept)


# --- per-match position lookup structures ---------------------------------

class MatchPositions:
    """Sorted-numpy-array lookups for one match's downsampled position
    data, built once per match and reused across every pass in it --
    avoids repeatedly filtering the ~150k-row parquet per pass."""

    def __init__(self, match_id):
        tbl = pq.read_table(os.path.join(OUT_DIR, "positions", f"{match_id}.parquet")).to_pandas()
        tbl["timestamp"] = pd.to_datetime(tbl["timestamp"], utc=True, format="ISO8601").dt.tz_localize(None)

        self.person_arr = {}
        for pid, g in tbl.groupby("person_id"):
            g = g.sort_values("timestamp")
            self.person_arr[pid] = {
                "t": g["timestamp"].values.astype("datetime64[ns]"),
                "x": g["x"].values,
                "y": g["y"].values,
            }
        # See build_dataset docstring / Stage 3 notes: the ball's PersonId is
        # a real DFL id, not literally "BALL" -- team_id is the only
        # reliable marker, so re-key it for a plain lookup by "BALL".
        ball_pid = tbl.loc[tbl["team_id"] == "BALL", "person_id"].iloc[0]
        self.person_arr["BALL"] = self.person_arr[ball_pid]

        self.spans = tbl.groupby("person_id")["timestamp"].agg(["min", "max"])

    def on_pitch(self, person_id, ts):
        if person_id not in self.spans.index:
            return False
        lo, hi = self.spans.loc[person_id, "min"], self.spans.loc[person_id, "max"]
        return lo <= ts <= hi

    @staticmethod
    def _nearest_idx(t_array, target_ns):
        idx = np.searchsorted(t_array, target_ns)
        if idx == 0:
            return 0
        if idx == len(t_array):
            return len(t_array) - 1
        before, after = t_array[idx - 1], t_array[idx]
        return idx - 1 if (target_ns - before) <= (after - target_ns) else idx

    def sync_nearest(self, event_ts_naive):
        ball = self.person_arr["BALL"]
        target = np.datetime64(event_ts_naive)
        i = self._nearest_idx(ball["t"], target)
        return ball["t"][i]

    def sync_refined(self, event_ts_naive, passer_id):
        """Within +/-REFINED_WINDOW_S of the tagged time, pick the ball
        frame minimising (distance from ball to passer) + LAMBDA * (time
        offset from the tag) -- our own simplified version of the paper's
        cost-function synchronisation (Kwiatkowski & Clark / Van Roy et al.
        style); see module-level constants for the exact weighting."""
        ball = self.person_arr["BALL"]
        target = np.datetime64(event_ts_naive)
        lo = target - np.timedelta64(int(REFINED_WINDOW_S * 1000), "ms")
        hi = target + np.timedelta64(int(REFINED_WINDOW_S * 1000), "ms")
        i0, i1 = np.searchsorted(ball["t"], lo), np.searchsorted(ball["t"], hi)
        passer = self.person_arr.get(passer_id)
        if i1 <= i0 or passer is None:
            return self.sync_nearest(event_ts_naive)

        best_cost, best_t = np.inf, None
        for k in range(i0, i1):
            j = self._nearest_idx(passer["t"], ball["t"][k])
            d_ball_passer = np.hypot(ball["x"][k] - passer["x"][j], ball["y"][k] - passer["y"][j])
            time_offset_s = abs((ball["t"][k] - target) / np.timedelta64(1, "s"))
            cost = d_ball_passer + REFINED_LAMBDA * time_offset_s
            if cost < best_cost:
                best_cost, best_t = cost, ball["t"][k]
        return best_t

    def smoothed_pos_vel(self, person_id, synced_ts):
        """Position + velocity for one player at (or just before) synced_ts.
        LEAKAGE RULE: only frames at or before synced_ts are ever touched --
        `idx` below is found via a right-side search that excludes anything
        after synced_ts, and the window then only looks backward from idx.
        Position returned is the actual last tracked point (not smoothed);
        velocity is finite-differenced from a savgol-smoothed causal window,
        which the paper recommends over raw frame-to-frame deltas at high
        speed. See VELOCITY_WINDOW's comment for why the window is causal,
        not centred: a centred window would need frames after the pass."""
        arr = self.person_arr.get(person_id)
        if arr is None:
            return None
        idx = np.searchsorted(arr["t"], synced_ts, side="right") - 1
        if idx < 0:
            return None  # not yet tracked at this point in the match

        start = max(0, idx - VELOCITY_WINDOW + 1)
        tt, xx, yy = arr["t"][start:idx + 1], arr["x"][start:idx + 1], arr["y"][start:idx + 1]
        n = len(tt)
        x_now, y_now = xx[-1], yy[-1]

        if n == 1:
            return x_now, y_now, 0.0, 0.0
        if n < 5:
            # Too little history for a degree-2 savgol fit -- fall back to
            # a plain backward difference between the last two points.
            dt = (tt[-1] - tt[-2]) / np.timedelta64(1, "s")
            vx, vy = (xx[-1] - xx[-2]) / dt, (yy[-1] - yy[-2]) / dt
            return x_now, y_now, vx, vy

        xs = savgol_filter(xx, window_length=n, polyorder=2)
        ys = savgol_filter(yy, window_length=n, polyorder=2)
        dt = (tt[-1] - tt[-2]) / np.timedelta64(1, "s")
        vx, vy = (xs[-1] - xs[-2]) / dt, (ys[-1] - ys[-2]) / dt
        return x_now, y_now, vx, vy


# --- attack-direction normalisation (same convention as Stage 2) ---------

def build_kickoff_lookup(events):
    kickoffs = events.loc[events["kickoff_gamesection"].notna(),
                           ["match_id", "kickoff_gamesection", "kickoff_teamleft", "timestamp"]]
    kickoffs = kickoffs.rename(columns={"kickoff_gamesection": "half", "kickoff_teamleft": "team_left"})
    second_half_cutoff = kickoffs.loc[kickoffs["half"] == "secondHalf"].set_index("match_id")["timestamp"]
    team_left_by_match_half = kickoffs.set_index(["match_id", "half"])["team_left"].to_dict()
    return second_half_cutoff, team_left_by_match_half


def attack_flip(match_id, ts, team_id, second_half_cutoff, team_left_by_match_half):
    half = "firstHalf" if ts < second_half_cutoff[match_id] else "secondHalf"
    team_left = team_left_by_match_half[(match_id, half)]
    return 1.0 if team_id == team_left else -1.0


# --- geometry helpers ------------------------------------------------------

def perp_dist_and_projection(px, py, cx, cy, ox, oy):
    """Perpendicular distance from opponent (ox,oy) to the segment passer
    (px,py) -> candidate (cx,cy), and the fraction along that segment where
    the opponent's projection falls (0=at passer, 1=at candidate, outside
    [0,1] means the projection lands beyond one of the two endpoints)."""
    seg_x, seg_y = cx - px, cy - py
    seg_len_sq = seg_x ** 2 + seg_y ** 2
    if seg_len_sq < 1e-9:  # passer and candidate at the same point (shouldn't happen)
        return np.hypot(ox - px, oy - py), 0.0
    t = ((ox - px) * seg_x + (oy - py) * seg_y) / seg_len_sq
    proj_x, proj_y = px + t * seg_x, py + t * seg_y
    perp = np.hypot(ox - proj_x, oy - proj_y)
    return perp, t


GOAL_X = 52.5  # attack-normalised: opponent goal always at +52.5, own goal at -52.5


# --- one pass -> its candidate rows ---------------------------------------

def build_candidate_rows(pass_row, match_pos, team_of_in_match, second_half_cutoff, team_left_by_match_half,
                          sync_method):
    """Returns a list of feature dicts (one per candidate teammate) for one
    pass, or None if the pass has to be skipped (logged by the caller)."""
    match_id = pass_row["match_id"]
    passer_id = pass_row["player_id"]
    team_id = pass_row["team_id"]
    event_ts_naive = pass_row["timestamp"].tz_convert("UTC").tz_localize(None)

    if sync_method == "nearest":
        synced_ts = match_pos.sync_nearest(event_ts_naive)
    elif sync_method == "refined":
        synced_ts = match_pos.sync_refined(event_ts_naive, passer_id)
    else:
        raise ValueError(f"unknown SYNC_METHOD: {sync_method}")

    flip = attack_flip(match_id, pass_row["timestamp"], team_id, second_half_cutoff, team_left_by_match_half)

    passer_pv = match_pos.smoothed_pos_vel(passer_id, synced_ts)
    if passer_pv is None:
        return None
    px, py = passer_pv[0] * flip, passer_pv[1] * flip

    # Teammates (candidates, excluding the passer) and opponents on the
    # pitch at the SYNCED moment -- not the raw tagged moment, since a few
    # hundred ms of drift could in principle cross a substitution boundary.
    teammates = [pid for pid, t in team_of_in_match.items() if t == team_id and pid != passer_id
                 and match_pos.on_pitch(pid, synced_ts)]
    opponents = [pid for pid, t in team_of_in_match.items() if t != team_id and t is not None
                 and match_pos.on_pitch(pid, synced_ts)]

    opp_positions = []
    for opp_id in opponents:
        pv = match_pos.smoothed_pos_vel(opp_id, synced_ts)
        if pv is not None:
            opp_positions.append((pv[0] * flip, pv[1] * flip))

    if opp_positions:
        passer_pressure = min(np.hypot(px - ox, py - oy) for ox, oy in opp_positions)
    else:
        passer_pressure = np.nan  # no opponent data at this moment -- rare, logged separately

    rows = []
    for cand_id in teammates:
        pv = match_pos.smoothed_pos_vel(cand_id, synced_ts)
        if pv is None:
            continue
        cx, cy = pv[0] * flip, pv[1] * flip
        cvx, cvy = pv[2] * flip, pv[3] * flip  # velocity flips the same way as position under a 180 deg rotation

        dx, dy = cx - px, cy - py
        dist_to_passer = np.hypot(dx, dy)

        # perp_dists_in_lane: perpendicular distance for every opponent whose
        # projection falls between passer and candidate (0<=frac<=1),
        # regardless of how close -- used for min_perp_dist_in_lane below.
        # n_opponents_in_lane is the stricter spec definition: those same
        # opponents, but only within 2m of the direct passing line.
        perp_dists_in_lane = []
        n_opponents_in_lane = 0
        for ox, oy in opp_positions:
            perp, frac = perp_dist_and_projection(px, py, cx, cy, ox, oy)
            if 0.0 <= frac <= 1.0:
                perp_dists_in_lane.append(perp)
                if perp < 2.0:
                    n_opponents_in_lane += 1

        rows.append({
            "pass_id": pass_row["event_id"],
            "match_id": match_id,
            "team_id": team_id,
            "passer_id": passer_id,
            "candidate_id": cand_id,
            "is_recipient": int(cand_id == pass_row["recipient_id"]),
            "dx": dx, "dy": dy,
            "dist_to_passer": dist_to_passer,
            "angle_to_passer": np.degrees(np.arctan2(dy, dx)),
            "cand_x": cx, "cand_y": cy,
            "cand_vx": cvx, "cand_vy": cvy,
            "cand_speed": np.hypot(cvx, cvy),
            "dist_to_opp_goal": np.hypot(cx - GOAL_X, cy),
            "dist_to_own_goal": np.hypot(cx + GOAL_X, cy),
            "dist_nearest_opponent": (min(np.hypot(cx - ox, cy - oy) for ox, oy in opp_positions)
                                       if opp_positions else np.nan),
            "n_opponents_within_5m": sum(1 for ox, oy in opp_positions if np.hypot(cx - ox, cy - oy) < 5.0),
            "n_opponents_in_lane": n_opponents_in_lane,
            "min_perp_dist_in_lane": min(perp_dists_in_lane) if perp_dists_in_lane else np.nan,
            "is_progressive": int(cx > px),
            "passer_pressure": passer_pressure,
        })

    if not rows:
        return None
    if not any(r["is_recipient"] for r in rows):
        return None  # recipient wasn't on-pitch at the synced moment -- drop, log separately

    return rows


# --- orchestration ---------------------------------------------------------

def compute_sync_quality(passes, match_positions):
    """Ball-to-event distance under both sync methods, for the Stage 3
    report -- NOT used to pick features, purely a diagnostic printed to
    compare against the paper's 9.37m (unsynced) / 2.61m (their refined)."""
    out = {"nearest": [], "refined": []}
    for _, row in passes.iterrows():
        mp = match_positions[row["match_id"]]
        event_ts = row["timestamp"].tz_convert("UTC").tz_localize(None)
        x_ev, y_ev = row["x_centred"], row["y_centred"]

        t_near = mp.sync_nearest(event_ts)
        ball = mp.person_arr["BALL"]
        i = np.searchsorted(ball["t"], t_near)
        i = min(i, len(ball["t"]) - 1)
        out["nearest"].append(np.hypot(ball["x"][i] - x_ev, ball["y"][i] - y_ev))

        t_ref = mp.sync_refined(event_ts, row["player_id"])
        j = np.searchsorted(ball["t"], t_ref)
        j = min(j, len(ball["t"]) - 1)
        out["refined"].append(np.hypot(ball["x"][j] - x_ev, ball["y"][j] - y_ev))

    return pd.DataFrame(out)


def main():
    print(f"SYNC_METHOD (from config.yaml) = {SYNC_METHOD}")
    events, players, matches = load_base_tables()
    match_ids = list(matches["match_id"])

    passes, funnel_log = select_passes(events, players)
    passes, n_dropped_not_on_pitch = add_on_pitch_filter(passes, match_ids)
    funnel_log["4_passer_and_recipient_on_pitch_at_event_time"] = len(passes)
    funnel_log["  -- dropped as not on pitch (sub timing / red card edge cases)"] = n_dropped_not_on_pitch

    print("\nPass selection funnel:")
    for k, v in funnel_log.items():
        print(f"  {k}: {v}")

    print(f"\nBuilding per-match position lookups for {len(match_ids)} matches...")
    match_positions = {m: MatchPositions(m) for m in match_ids}

    print("\nSynchronisation quality (ball-to-event distance, both methods, all", len(passes), "passes):")
    sync_quality = compute_sync_quality(passes, match_positions)
    print(sync_quality.describe())
    print(f"  paper's reported figures for reference: unsynced mean=9.37m, their refined mean=2.61m")
    print(f"  ours: nearest mean={sync_quality['nearest'].mean():.2f}m, "
          f"refined mean={sync_quality['refined'].mean():.2f}m")

    second_half_cutoff, team_left_by_match_half = build_kickoff_lookup(events)

    all_rows = []
    n_dropped_no_recipient_on_pitch = 0
    for match_id in match_ids:
        team_of_in_match = players.loc[players["match_id"] == match_id].set_index("person_id")["team_id"].to_dict()
        match_pass_rows = passes[passes["match_id"] == match_id]
        for _, prow in match_pass_rows.iterrows():
            result = build_candidate_rows(prow, match_positions[match_id], team_of_in_match,
                                           second_half_cutoff, team_left_by_match_half, SYNC_METHOD)
            if result is None:
                n_dropped_no_recipient_on_pitch += 1
                continue
            all_rows.extend(result)

    df = pd.DataFrame(all_rows)
    # dist_rank: computed after the fact, over the whole table grouped by
    # pass -- rank 1 = closest candidate to the passer for that pass.
    df["dist_rank"] = df.groupby("pass_id")["dist_to_passer"].rank(method="first").astype(int)

    print(f"\n5_final (dropped {n_dropped_no_recipient_on_pitch} more: recipient not on pitch at synced moment)")
    print(f"\nFinal: {df['pass_id'].nunique()} passes -> {len(df)} candidate rows "
          f"({len(df) / df['pass_id'].nunique():.1f} candidates/pass on average)")

    out_path = os.path.join(OUT_DIR, "pass_candidates.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")

    print("\n" + "=" * 70)
    print("STAGE 3 SANITY CHECKS")
    print("=" * 70)

    # Exactly one recipient per pass.
    recipients_per_pass = df.groupby("pass_id")["is_recipient"].sum()
    bad = (recipients_per_pass != 1).sum()
    print(f"[{'OK' if bad == 0 else 'FAIL'}] exactly one is_recipient=1 per pass: "
          f"{(recipients_per_pass == 1).sum()}/{len(recipients_per_pass)}")

    # Candidate count sanity: should be <= 10 almost always (11 - passer),
    # occasionally less (red card / late substitution gaps).
    cand_counts = df.groupby("pass_id").size()
    print(f"[INFO] candidates per pass: min={cand_counts.min()}, max={cand_counts.max()}, "
          f"mean={cand_counts.mean():.1f}")
    print(f"[{'OK' if cand_counts.max() <= 10 else 'FAIL'}] max candidates per pass <= 10")

    # Leakage tripwire: no single feature should near-perfectly separate
    # is_recipient (that would mean the feature secretly encodes the
    # answer, e.g. a post-pass ball position slipping in by mistake).
    print("\n[Check] feature correlation with is_recipient (tripwire for leaked features):")
    feature_cols = ["dx", "dy", "dist_to_passer", "angle_to_passer", "cand_x", "cand_y",
                     "cand_vx", "cand_vy", "cand_speed", "dist_to_opp_goal", "dist_to_own_goal",
                     "dist_nearest_opponent", "n_opponents_within_5m", "n_opponents_in_lane",
                     "min_perp_dist_in_lane", "is_progressive", "dist_rank", "passer_pressure"]
    any_suspicious = False
    for col in feature_cols:
        corr = df[col].corr(df["is_recipient"])
        flag = ""
        if abs(corr) > 0.9:
            flag = "  <-- SUSPICIOUSLY HIGH, investigate before modelling"
            any_suspicious = True
        print(f"  {col}: {corr:+.3f}{flag}")
    print(f"[{'OK' if not any_suspicious else 'FAIL'}] no feature exceeds |corr|=0.9 with is_recipient")

    print(f"\n[INFO] NaN counts (expected for passer_pressure/dist_nearest_opponent/min_perp_dist_in_lane "
          f"on the rare pass with no opponents tracked, or no opponent in the lane):")
    print(df[feature_cols].isna().sum()[lambda s: s > 0])


if __name__ == "__main__":
    main()
