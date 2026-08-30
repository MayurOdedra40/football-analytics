"""
Sanity tests over the already-built data/processed/ outputs.

These are regression checks, not the pipeline itself -- they assume
`python -m src.parse` and `python -m src.build_dataset` have already been
run (both scripts print their own, more detailed sanity checks at the end
of a run; this file re-asserts the three checks the project spec calls out
by name, so a later code change can't silently break them unnoticed).

Run: `pytest tests/` (or `pytest tests/test_sanity.py -v`) from the repo
root, with the venv active.
"""

import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.parse import OUT_DIR


@pytest.fixture(scope="module")
def matches():
    return pd.read_csv(os.path.join(OUT_DIR, "matches.csv"))


@pytest.fixture(scope="module")
def players():
    return pd.read_csv(os.path.join(OUT_DIR, "players.csv"))


@pytest.fixture(scope="module")
def events():
    df = pd.read_csv(os.path.join(OUT_DIR, "events.csv"), low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    return df


@pytest.fixture(scope="module")
def pass_candidates():
    path = os.path.join(OUT_DIR, "pass_candidates.parquet")
    if not os.path.exists(path):
        pytest.skip("pass_candidates.parquet not built yet -- run `python -m src.build_dataset` first")
    return pd.read_parquet(path)


# --- 1. coordinate transform ------------------------------------------

def test_coordinate_transform_formula(events):
    """x_centred/y_centred must equal x_event-52.5 / y_event-34.0 exactly
    (the transform itself, not tagger timing -- see Stage 1's distinction
    in README) for every row where the source coordinate is present."""
    has_coord = events["x_event"].notna()
    assert has_coord.sum() > 0
    np.testing.assert_allclose(events.loc[has_coord, "x_centred"], events.loc[has_coord, "x_event"] - 52.5)
    np.testing.assert_allclose(events.loc[has_coord, "y_centred"], events.loc[has_coord, "y_event"] - 34.0)


def test_kickoff_lands_near_ball_position(events):
    """At least one match's kickoff event (a moment taggers can pin down
    precisely -- the whistle) must land within 1m of the ball's tracked
    position after the transform. Confirms the transform formula is right,
    independent of tagging-timestamp jitter on other matches (Stage 1
    found only 2/7 matches land this close; the other 5 are timing noise,
    not a transform bug -- see README)."""
    kickoffs = events[events["top_level_type"] == "KickOff"]
    distances = []
    for match_id, group in kickoffs.groupby("match_id"):
        ko = group.iloc[0]
        if pd.isna(ko["x_centred"]):
            continue
        pq_path = os.path.join(OUT_DIR, "positions", f"{match_id}.parquet")
        ball = pq.read_table(pq_path, filters=[("team_id", "=", "BALL")]).to_pandas()
        ball["timestamp"] = pd.to_datetime(ball["timestamp"], utc=True, format="ISO8601")
        idx = (ball["timestamp"] - ko["timestamp"]).abs().idxmin()
        nearest = ball.loc[idx]
        d = np.hypot(nearest["x"] - ko["x_centred"], nearest["y"] - ko["y_centred"])
        distances.append(d)
    assert distances, "no KickOff events with usable coordinates found"
    assert min(distances) <= 1.0, f"no match's kickoff landed within 1m of the ball (best was {min(distances):.2f}m)"


# --- 2. distance covered -------------------------------------------------

def test_distance_covered_in_range_for_full_match_starters(events, players):
    """Full-match (never subbed, not sent off), non-keeper starters must
    cover 9-13km -- Stage 1's headline sanity check. Uses the same
    full-resolution D-sum approach as src/parse.py (not the downsampled
    parquet, which was shown in Stage 1 to undercount by ~3x)."""
    from src.parse import compute_distance_covered_km

    n_checked = 0
    n_in_range = 0
    for match_id in matches_from(players):
        starters = players[(players["match_id"] == match_id) & (players["starting"])]
        keepers = set(players[(players["match_id"] == match_id) & (players["playing_position"] == "TW")]["person_id"])
        subs_df = events[(events["match_id"] == match_id) & (events["top_level_type"] == "Substitution")]
        subbed = set(subs_df["substitution_playerout"].dropna()) | set(subs_df["substitution_playerin"].dropna())
        cautions_df = events[(events["match_id"] == match_id) & (events["top_level_type"] == "Caution")]
        sent_off = set(cautions_df[cautions_df["caution_cardcolor"] == "red"]["caution_player"].dropna())

        dist_km = compute_distance_covered_km(match_id)
        for person_id, km in dist_km.items():
            if (person_id not in set(starters["person_id"]) or person_id in keepers
                    or person_id in subbed or person_id in sent_off):
                continue
            n_checked += 1
            if 9 <= km <= 13:
                n_in_range += 1

    assert n_checked > 0
    # A small number of legitimate edge cases sit just under 9km in one
    # low-tempo, high-foul match (see README) -- require the large majority
    # in range rather than 100%, but catch a real unit/duplication bug
    # (which would put most or all players far outside the band).
    assert n_in_range / n_checked >= 0.9, (
        f"only {n_in_range}/{n_checked} full-match starters in the 9-13km band -- "
        f"check for a unit or half-duplication bug"
    )


def matches_from(players):
    return sorted(players["match_id"].unique())


# --- 3. leakage tripwire ---------------------------------------------------

def test_no_feature_leaks_the_answer(pass_candidates):
    """No single candidate feature should near-perfectly separate
    is_recipient -- that would mean a feature secretly encodes the answer
    (e.g. a post-pass ball position slipping in by mistake). Stage 3's own
    run measured a max |corr| of ~0.26 (dist_rank); this is a generous
    threshold to catch a real leak, not to reproduce that exact number."""
    feature_cols = [
        "dx", "dy", "dist_to_passer", "angle_to_passer", "cand_x", "cand_y",
        "cand_vx", "cand_vy", "cand_speed", "dist_to_opp_goal", "dist_to_own_goal",
        "dist_nearest_opponent", "n_opponents_within_5m", "n_opponents_in_lane",
        "min_perp_dist_in_lane", "is_progressive", "dist_rank", "passer_pressure",
    ]
    offenders = {}
    for col in feature_cols:
        corr = pass_candidates[col].corr(pass_candidates["is_recipient"])
        if abs(corr) > 0.9:
            offenders[col] = corr
    assert not offenders, f"suspiciously high correlation with is_recipient (possible leakage): {offenders}"


def test_exactly_one_recipient_per_pass(pass_candidates):
    """A basic structural check alongside the leakage tripwire: every pass
    group must have exactly one candidate flagged as the true recipient."""
    counts = pass_candidates.groupby("pass_id")["is_recipient"].sum()
    assert (counts == 1).all(), f"{(counts != 1).sum()} passes don't have exactly one is_recipient=1 row"
