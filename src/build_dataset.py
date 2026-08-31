import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.parse import CONFIG, OUT_DIR

SYNC_METHOD = CONFIG["sync_method"]  # "nearest" or "refined"
SEED = CONFIG["seed"]

REFINED_WINDOW_S = 3.0
REFINED_LAMBDA = 2.0

RNG = np.random.default_rng(SEED)


# ---pass selection ---------------------------------

def load_base_tables():
    events = pd.read_csv(os.path.join(OUT_DIR, "events.csv"), low_memory=False)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, format="ISO8601")
    players = pd.read_csv(os.path.join(OUT_DIR, "players.csv"))
    matches = pd.read_csv(os.path.join(OUT_DIR, "matches.csv"))
    return events, players, matches


def select_passes(events, players):
    """Selects passes from the raw events table, dropping any that don't have a
    valid recipient_id"""
    log = {}

    is_pass = events["event_type_path"].str.endswith(("_Pass", "_Cross"))
    step = events[is_pass].copy()
    log["pass_or_cross_events"] = len(step)

    step = step[step["recipient_id"].notna()].copy()
    log["has_recipient_id"] = len(step)

    # team lookup: (match_id, person_id) -> team_id
    team_of = players.set_index(["match_id", "person_id"])["team_id"]
    step["recipient_team"] = [team_of.get((m, r)) for m, r in zip(step["match_id"], step["recipient_id"])]
    same_team = step["team_id"] == step["recipient_team"]
    n_dropped_cross_team = (~same_team).sum()
    step = step[same_team].copy()
    log["same_team"] = len(step)
    log[" dropped "] = int(n_dropped_cross_team)

    return step, log


def add_on_pitch_filter(passes, match_ids):
    """Drops passes where the passer or recipient isn't covered by tracked
    position data at the event's tagged timestamp."""
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


# --- per-match position lookup ---------------------------------

class MatchPositions:
    """Sorted-numpy-array lookups for one match position data, 
    built it per match and reused across every pass"""

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
        """distance + lamda + |time offset|."""
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

    def get_position(self, person_id, synced_ts):
        """(x, y) for one player """
        arr = self.person_arr.get(person_id)
        if arr is None:
            return None
        idx = np.searchsorted(arr["t"], synced_ts, side="right") - 1
        if idx < 0:
            return None  # not yet tracked at this point in the match
        return arr["x"][idx], arr["y"][idx]


# --- attack-direction ----------------------------------

def build_kickoff_lookup(events):
    """Returns two dicts for all matches: 
    give match halfs and then helps in flipping, so always left to right"""
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


GOAL_X = 52.5  # opponent goal always at +52.5, own goal at -52.5


# --- one pass per its candidate rows ---------------------------------------

def build_candidate_rows(pass_row, match_pos, team_of_in_match, second_half_cutoff, team_left_by_match_half,
                          sync_method):
    """Returns a list of feature dicts (one per candidate teammate) for one
    pass, or None if the pass has to be skipped."""
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

    passer_pos = match_pos.get_position(passer_id, synced_ts)
    if passer_pos is None:
        return None
    px, py = passer_pos[0] * flip, passer_pos[1] * flip

    teammates = [pid for pid, t in team_of_in_match.items() if t == team_id and pid != passer_id
                 and match_pos.on_pitch(pid, synced_ts)]
    opponents = [pid for pid, t in team_of_in_match.items() if t != team_id and t is not None
                 and match_pos.on_pitch(pid, synced_ts)]

    opp_positions = []
    for opp_id in opponents:
        pos = match_pos.get_position(opp_id, synced_ts)
        if pos is not None:
            opp_positions.append((pos[0] * flip, pos[1] * flip))

    rows = []
    for cand_id in teammates:
        pos = match_pos.get_position(cand_id, synced_ts)
        if pos is None:
            continue
        cx, cy = pos[0] * flip, pos[1] * flip

        dx, dy = cx - px, cy - py
        dist_to_passer = np.hypot(dx, dy)

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
            "dist_to_opp_goal": np.hypot(cx - GOAL_X, cy),
            "dist_to_own_goal": np.hypot(cx + GOAL_X, cy),
            "dist_nearest_opponent": (min(np.hypot(cx - ox, cy - oy) for ox, oy in opp_positions)
                                       if opp_positions else np.nan),
            "n_opponents_within_5m": sum(1 for ox, oy in opp_positions if np.hypot(cx - ox, cy - oy) < 5.0),
        })

    if not rows:
        return None
    if not any(r["is_recipient"] for r in rows):
        return None 

    return rows


#  ---------------------------------------------------------

def compute_sync_quality(passes, match_positions):
    """Computes the ball-to-event distance for each pass, using both the nearest and refined method"""
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
    events, players, matches = load_base_tables()
    match_ids = list(matches["match_id"])

    passes, funnel_log = select_passes(events, players)
    passes, n_dropped_not_on_pitch = add_on_pitch_filter(passes, match_ids)
    funnel_log["passer_and_recipient_on_pitch_at_event_time"] = len(passes)
    funnel_log["dropped as not on pitch"] = n_dropped_not_on_pitch

    print("\nPass selection funnel:")
    for k, v in funnel_log.items():
        print(f"  {k}: {v}")

    print(f"\nBuilding per-match position lookups for {len(match_ids)} matches...")
    match_positions = {m: MatchPositions(m) for m in match_ids}

    print("\nSynchronisation (ball-to-event distance, both methods, all", len(passes), "passes):")
    sync_quality = compute_sync_quality(passes, match_positions)
    print(sync_quality.describe())

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

    print(f"\nfinal (dropped {n_dropped_no_recipient_on_pitch} )")
    print(f"\nFinal: {df['pass_id'].nunique()} passes -> {len(df)} candidate rows "
          f"({len(df) / df['pass_id'].nunique():.1f} candidates/pass on average)")

    out_path = os.path.join(OUT_DIR, "pass_candidates.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
