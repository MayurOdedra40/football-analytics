"""
EXPERIMENTAL: does attention-over-candidates or an LSTM-over-trajectory
change the results? Not part of the core Stage 1-6 pipeline -- a direct
answer to a direct question, using the exact same leave-one-match-out CV,
metrics, and (for the attention variant) features as Stage 5's MLP, so the
comparison is apples-to-apples.

Run: `python -m src.experiment_architectures` (~2-3 min on CPU, mostly the
one-time trajectory extraction pass over all 7 matches' position data).
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.baselines import leave_one_match_out_cv, evaluate_predictions
from src.build_dataset import MatchPositions, SYNC_METHOD, load_base_tables
from src.model import CandidateScorer
from src.model_variants import CandidateScorerAttention, CandidateScorerLSTM
from src.parse import CONFIG, OUT_DIR
from src.train import (
    NN_FEATURE_COLS, MAX_EPOCHS, PATIENCE, BATCH_SIZE, LR, WEIGHT_DECAY,
    INTERNAL_VAL_FRACTION, passes_to_tensors, masked_scores, tensors_to_long_df, set_all_seeds,
)

SEED = CONFIG["seed"]
TRAJ_LEN = 10  # trailing frames, ~1.2s at the position data's 8.33Hz


# --- one-time trajectory extraction across all matches ---------------------

def build_pass_context(df, events):
    """pass_id -> (match_id, synced_ts, flip). Same sync method + rotation
    Stage 3 used, so the trajectory data lines up with the same moment the
    static features were computed for."""
    from src.build_dataset import build_kickoff_lookup, attack_flip

    second_half_cutoff, team_left_by_match_half = build_kickoff_lookup(events)
    match_ids = sorted(df["match_id"].unique())
    match_positions = {m: MatchPositions(m) for m in match_ids}

    events_by_id = events.set_index("event_id")
    context = {}
    for pass_id, match_id in df[["pass_id", "match_id"]].drop_duplicates().itertuples(index=False):
        row = events_by_id.loc[pass_id]
        event_ts_naive = row["timestamp"].tz_convert("UTC").tz_localize(None)
        mp = match_positions[match_id]
        if SYNC_METHOD == "nearest":
            synced_ts = mp.sync_nearest(event_ts_naive)
        else:
            synced_ts = mp.sync_refined(event_ts_naive, row["player_id"])
        flip = attack_flip(match_id, row["timestamp"], row["team_id"], second_half_cutoff, team_left_by_match_half)
        context[pass_id] = (match_id, synced_ts, flip)
    return context, match_positions


def get_candidate_trajectory(match_pos, candidate_id, synced_ts, flip, traj_len=TRAJ_LEN):
    """Last traj_len frames up to and including synced_ts, as (dx, dy)
    relative to the synced-moment position (translation-invariant -- only
    the run's shape/dynamics matters; absolute position is already a
    separate static feature). Left-pads by repeating the earliest available
    frame for a player with less than traj_len frames of history (e.g. just
    subbed on)."""
    arr = match_pos.person_arr.get(candidate_id)
    if arr is None:
        return np.zeros((traj_len, 2), dtype=np.float32)
    idx = np.searchsorted(arr["t"], synced_ts, side="right") - 1
    if idx < 0:
        return np.zeros((traj_len, 2), dtype=np.float32)
    start = max(0, idx - traj_len + 1)
    xs = arr["x"][start:idx + 1] * flip
    ys = arr["y"][start:idx + 1] * flip
    n = len(xs)
    if n < traj_len:
        xs = np.concatenate([np.full(traj_len - n, xs[0]), xs])
        ys = np.concatenate([np.full(traj_len - n, ys[0]), ys])
    now_x, now_y = xs[-1], ys[-1]
    return np.stack([xs - now_x, ys - now_y], axis=1).astype(np.float32)


def build_trajectory_tensor(pass_ids, candidate_ids, pass_context, match_positions, traj_len=TRAJ_LEN):
    """Mirrors passes_to_tensors' (pass, candidate) slot ordering exactly --
    driven by the SAME pass_ids/candidate_ids arrays passes_to_tensors
    already produced, so the two tensors line up slot-for-slot."""
    max_candidates = len(candidate_ids[0])
    traj = np.zeros((len(pass_ids), max_candidates, traj_len, 2), dtype=np.float32)
    for i, pid in enumerate(pass_ids):
        match_id, synced_ts, flip = pass_context[pid]
        mp = match_positions[match_id]
        for j, cand_id in enumerate(candidate_ids[i]):
            if cand_id is None:
                continue
            traj[i, j] = get_candidate_trajectory(mp, cand_id, synced_ts, flip, traj_len)
    return torch.from_numpy(traj)


# --- train/predict closures, same predict_fn(train_df, test_df) contract ---

def make_attention_predict_fn(max_candidates):
    def train_predict(train_df, test_df):
        rng = np.random.default_rng(SEED)
        train_pass_ids = train_df["pass_id"].unique()
        rng.shuffle(train_pass_ids)
        n_val = max(1, int(len(train_pass_ids) * INTERNAL_VAL_FRACTION))
        val_ids, fit_ids = set(train_pass_ids[:n_val]), set(train_pass_ids[n_val:])
        fit_df = train_df[train_df["pass_id"].isin(fit_ids)]
        val_df = train_df[train_df["pass_id"].isin(val_ids)]

        fit_t = passes_to_tensors(fit_df, max_candidates)
        val_t = passes_to_tensors(val_df, max_candidates, fit_t["feature_means"], fit_t["feature_stds"])
        test_t = passes_to_tensors(test_df, max_candidates, fit_t["feature_means"], fit_t["feature_stds"])

        model = CandidateScorerAttention(n_features=len(NN_FEATURE_COLS))
        optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.CrossEntropyLoss()
        n_fit = fit_t["X"].shape[0]
        best_val_loss, best_state, epochs_since_improve = float("inf"), None, 0

        for epoch in range(MAX_EPOCHS):
            model.train()
            perm = torch.randperm(n_fit)
            for start in range(0, n_fit, BATCH_SIZE):
                idx = perm[start:start + BATCH_SIZE]
                optimiser.zero_grad()
                scores = model(fit_t["X"][idx], fit_t["mask"][idx]).masked_fill(~fit_t["mask"][idx], float("-inf"))
                loss = loss_fn(scores, fit_t["target"][idx])
                loss.backward()
                optimiser.step()
            model.eval()
            with torch.no_grad():
                val_scores = model(val_t["X"], val_t["mask"]).masked_fill(~val_t["mask"], float("-inf"))
                val_loss = loss_fn(val_scores, val_t["target"]).item()
            if val_loss < best_val_loss - 1e-4:
                best_val_loss, best_state, epochs_since_improve = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= PATIENCE:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_scores = model(test_t["X"], test_t["mask"]).masked_fill(~test_t["mask"], float("-inf"))
            test_probs = torch.softmax(test_scores, dim=1)
        pred_df = tensors_to_long_df(test_t["pass_ids"], test_t["candidate_ids"], test_probs, test_t["mask"])
        pred_df = pred_df.merge(test_df[["pass_id", "candidate_id", "is_recipient"]],
                                 on=["pass_id", "candidate_id"], how="left")
        return pred_df
    return train_predict


def make_lstm_predict_fn(max_candidates, pass_context, match_positions):
    def train_predict(train_df, test_df):
        rng = np.random.default_rng(SEED)
        train_pass_ids = train_df["pass_id"].unique()
        rng.shuffle(train_pass_ids)
        n_val = max(1, int(len(train_pass_ids) * INTERNAL_VAL_FRACTION))
        val_ids, fit_ids = set(train_pass_ids[:n_val]), set(train_pass_ids[n_val:])
        fit_df = train_df[train_df["pass_id"].isin(fit_ids)]
        val_df = train_df[train_df["pass_id"].isin(val_ids)]

        fit_t = passes_to_tensors(fit_df, max_candidates)
        val_t = passes_to_tensors(val_df, max_candidates, fit_t["feature_means"], fit_t["feature_stds"])
        test_t = passes_to_tensors(test_df, max_candidates, fit_t["feature_means"], fit_t["feature_stds"])

        fit_traj = build_trajectory_tensor(fit_t["pass_ids"], fit_t["candidate_ids"], pass_context, match_positions)
        val_traj = build_trajectory_tensor(val_t["pass_ids"], val_t["candidate_ids"], pass_context, match_positions)
        test_traj = build_trajectory_tensor(test_t["pass_ids"], test_t["candidate_ids"], pass_context, match_positions)

        model = CandidateScorerLSTM(n_static_features=len(NN_FEATURE_COLS))
        optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.CrossEntropyLoss()
        n_fit = fit_t["X"].shape[0]
        best_val_loss, best_state, epochs_since_improve = float("inf"), None, 0

        for epoch in range(MAX_EPOCHS):
            model.train()
            perm = torch.randperm(n_fit)
            for start in range(0, n_fit, BATCH_SIZE):
                idx = perm[start:start + BATCH_SIZE]
                optimiser.zero_grad()
                scores = model(fit_t["X"][idx], fit_traj[idx]).masked_fill(~fit_t["mask"][idx], float("-inf"))
                loss = loss_fn(scores, fit_t["target"][idx])
                loss.backward()
                optimiser.step()
            model.eval()
            with torch.no_grad():
                val_scores = model(val_t["X"], val_traj).masked_fill(~val_t["mask"], float("-inf"))
                val_loss = loss_fn(val_scores, val_t["target"]).item()
            if val_loss < best_val_loss - 1e-4:
                best_val_loss, best_state, epochs_since_improve = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= PATIENCE:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_scores = model(test_t["X"], test_traj).masked_fill(~test_t["mask"], float("-inf"))
            test_probs = torch.softmax(test_scores, dim=1)
        pred_df = tensors_to_long_df(test_t["pass_ids"], test_t["candidate_ids"], test_probs, test_t["mask"])
        pred_df = pred_df.merge(test_df[["pass_id", "candidate_id", "is_recipient"]],
                                 on=["pass_id", "candidate_id"], how="left")
        return pred_df
    return train_predict


def make_mlp_predict_fn(max_candidates):
    """Stage 5's baseline, re-run here for a same-process reference point."""
    from src.train import train_model, predict_with_model

    def train_predict(train_df, test_df):
        model, means, stds = train_model(train_df, max_candidates)
        return predict_with_model(model, test_df, max_candidates, means, stds)
    return train_predict


def main():
    print(f"seed={SEED}")
    set_all_seeds(SEED)

    df = pd.read_parquet(f"{OUT_DIR}/pass_candidates.parquet")
    events, players, matches = load_base_tables()
    max_candidates = df.groupby("pass_id").size().max()

    print("Building trajectory context (sync + rotation) for all passes...")
    t0 = time.time()
    pass_context, match_positions = build_pass_context(df, events)
    print(f"  done in {time.time() - t0:.1f}s")

    variants = {
        "mlp (Stage 5 baseline)": make_mlp_predict_fn(max_candidates),
        "attention_over_candidates": make_attention_predict_fn(max_candidates),
        "lstm_trajectory": make_lstm_predict_fn(max_candidates, pass_context, match_positions),
    }

    all_folds = []
    for name, fn in variants.items():
        print(f"\nRunning leave-one-match-out CV: {name}...")
        t0 = time.time()
        set_all_seeds(SEED)
        fold_df = leave_one_match_out_cv(df, fn, name)
        print(f"  {time.time() - t0:.1f}s")
        all_folds.append(fold_df)

    all_folds_df = pd.concat(all_folds, ignore_index=True)
    out_path = f"{OUT_DIR}/architecture_experiment_results.csv"
    all_folds_df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("ARCHITECTURE EXPERIMENT RESULTS (mean +/- std across 7 folds)")
    print("=" * 70)
    summary = all_folds_df.groupby("model")[["top1_acc", "top3_acc", "mrr", "cross_entropy"]].agg(["mean", "std"])
    for name in variants:
        row = summary.loc[name]
        print(f"\n{name}:")
        for metric in ["top1_acc", "top3_acc", "mrr", "cross_entropy"]:
            print(f"  {metric}: {row[(metric, 'mean')]:.3f} +/- {row[(metric, 'std')]:.3f}")

    print(f"\nWrote per-fold results to {out_path}")


if __name__ == "__main__":
    main()
