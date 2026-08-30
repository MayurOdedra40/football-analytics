"""
Stage 5: train the shared-weight candidate scorer (src/model.py) and
evaluate it the same way as Stage 4's baselines -- leave-one-match-out CV,
same four metrics, via the exact same `evaluate_predictions` /
`leave_one_match_out_cv` helpers imported from src/baselines.py, so the
final comparison table is apples-to-apples rather than separately
reimplemented and possibly subtly different.

Config flag TRAIN_SCOPE (config.yaml): "all_matches" (default, all 7
matches) or "single_team" (Fortuna Düsseldorf's 5 matches only) -- run both
and report the gap, a concrete data-volume result.

Everything is small on purpose (tiny MLP, ~4-5k passes) and trains in well
under a minute on CPU per TRAIN_SCOPE.

Run: `python -m src.train` (from repo root, venv active).
"""

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.baselines import evaluate_predictions, leave_one_match_out_cv, ALL_FEATURE_COLS
from src.model import CandidateScorer
from src.parse import CONFIG, OUT_DIR

SEED = CONFIG["seed"]
TRAIN_SCOPE = CONFIG["train_scope"]  # "all_matches" or "single_team"

# NaN in min_perp_dist_in_lane means "no opponent's projection falls in the
# passing lane" (Stage 3), which a neural net can't consume directly the
# way LightGBM does. Concrete choice, deferred from Stage 3's decisions
# table: impute with a fixed sentinel well past any real in-lane distance
# (an unmarked passing lane is functionally "unlimited space"), PLUS an
# explicit missing-indicator feature so the network isn't just told a
# suspiciously large number and left to guess why.
LANE_MISSING_SENTINEL = 50.0
NN_FEATURE_COLS = ALL_FEATURE_COLS + ["lane_missing_flag"]

MAX_EPOCHS = 200
PATIENCE = 15
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
INTERNAL_VAL_FRACTION = 0.15  # carved out of the 6 TRAINING matches only, for early stopping


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --- data: long dataframe -> padded/masked tensors ------------------------

def passes_to_tensors(df, max_candidates, feature_means=None, feature_stds=None):
    """One row per (pass, candidate) -> per-pass padded tensors.
    If feature_means/stds are None, they're computed from this df (used for
    the training split) and returned so the same scaling can be applied to
    validation/test data -- standardising with test-set statistics would
    itself be a (mild) form of leakage."""
    df = df.copy()
    df["lane_missing_flag"] = df["min_perp_dist_in_lane"].isna().astype(float)
    df["min_perp_dist_in_lane"] = df["min_perp_dist_in_lane"].fillna(LANE_MISSING_SENTINEL)

    if feature_means is None:
        feature_means = df[NN_FEATURE_COLS].mean()
        feature_stds = df[NN_FEATURE_COLS].std().replace(0, 1.0)
    df_norm = df.copy()
    df_norm[NN_FEATURE_COLS] = (df[NN_FEATURE_COLS] - feature_means) / feature_stds

    pass_ids = df["pass_id"].unique()
    n_features = len(NN_FEATURE_COLS)
    X = np.zeros((len(pass_ids), max_candidates, n_features), dtype=np.float32)
    mask = np.zeros((len(pass_ids), max_candidates), dtype=bool)
    target = np.zeros(len(pass_ids), dtype=np.int64)
    candidate_ids = []  # candidate_ids[i][j] = candidate_id for pass i, slot j

    for i, pid in enumerate(pass_ids):
        group = df_norm[df_norm["pass_id"] == pid].reset_index(drop=True)
        n = len(group)
        X[i, :n, :] = group[NN_FEATURE_COLS].values
        mask[i, :n] = True
        target[i] = group.index[group["is_recipient"] == 1][0]
        candidate_ids.append(list(group["candidate_id"]) + [None] * (max_candidates - n))

    return {
        "X": torch.from_numpy(X), "mask": torch.from_numpy(mask), "target": torch.from_numpy(target),
        "pass_ids": pass_ids, "candidate_ids": candidate_ids,
        "feature_means": feature_means, "feature_stds": feature_stds,
    }


def masked_scores(model, X, mask):
    scores = model(X)
    return scores.masked_fill(~mask, float("-inf"))


def tensors_to_long_df(pass_ids, candidate_ids, probs, mask):
    """Inverse of passes_to_tensors, for the predicted probabilities --
    reconstructs the same long format evaluate_predictions expects."""
    rows = []
    probs_np = probs.detach().numpy()
    mask_np = mask.numpy()
    for i, pid in enumerate(pass_ids):
        for j in range(mask_np.shape[1]):
            if not mask_np[i, j]:
                continue
            rows.append({"pass_id": pid, "candidate_id": candidate_ids[i][j], "prob": probs_np[i, j]})
    return pd.DataFrame(rows)


# --- one leave-one-match-out fold: train + predict -------------------------
#
# Split into train_model (returns the actual trained model + the scaling
# stats used on it) and predict_with_model (scores an arbitrary df with a
# trained model) so Stage 6's visualisation notebook can run inference on
# specific hand-picked passes -- not just get a metrics table back the way
# make_train_predict_fn's combined version (below) does for Stage 5's CV.

def train_model(train_df, max_candidates):
    """Trains one CandidateScorer with early stopping on an internal
    train/val split (see module docstring). Returns (model, feature_means,
    feature_stds) -- the means/stds are part of the trained artifact, since
    predict_with_model must normalise new data the same way."""
    # Internal train/val split, BY PASS, from the training matches only --
    # a normal model-selection split (for early stopping), not the leave-
    # one-match-out test fold, so a random split at pass level is fine here
    # (unlike the outer CV split, which must never be random -- see README).
    rng = np.random.default_rng(SEED)
    train_pass_ids = train_df["pass_id"].unique()
    rng.shuffle(train_pass_ids)
    n_val = max(1, int(len(train_pass_ids) * INTERNAL_VAL_FRACTION))
    val_ids, fit_ids = set(train_pass_ids[:n_val]), set(train_pass_ids[n_val:])
    fit_df = train_df[train_df["pass_id"].isin(fit_ids)]
    val_df = train_df[train_df["pass_id"].isin(val_ids)]

    fit_t = passes_to_tensors(fit_df, max_candidates)
    val_t = passes_to_tensors(val_df, max_candidates, fit_t["feature_means"], fit_t["feature_stds"])

    model = CandidateScorer(n_features=len(NN_FEATURE_COLS))
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
            scores = masked_scores(model, fit_t["X"][idx], fit_t["mask"][idx])
            loss = loss_fn(scores, fit_t["target"][idx])
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_scores = masked_scores(model, val_t["X"], val_t["mask"])
            val_loss = loss_fn(val_scores, val_t["target"]).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss, best_state, epochs_since_improve = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, fit_t["feature_means"], fit_t["feature_stds"]


def predict_with_model(model, df, max_candidates, feature_means, feature_stds):
    """Scores an arbitrary df (any subset of pass_candidates.parquet's
    columns/rows) with an already-trained model. Returns the same long
    format (pass_id, candidate_id, prob, is_recipient) as the baselines."""
    t = passes_to_tensors(df, max_candidates, feature_means, feature_stds)
    model.eval()
    with torch.no_grad():
        scores = masked_scores(model, t["X"], t["mask"])
        probs = torch.softmax(scores, dim=1)
    pred_df = tensors_to_long_df(t["pass_ids"], t["candidate_ids"], probs, t["mask"])
    pred_df = pred_df.merge(df[["pass_id", "candidate_id", "is_recipient"]],
                             on=["pass_id", "candidate_id"], how="left")
    return pred_df


def make_train_predict_fn(max_candidates):
    """Returns a predict_fn(train_df, test_df) -> df with 'prob', matching
    the signature src/baselines.leave_one_match_out_cv expects -- lets us
    reuse that exact CV runner instead of writing a second one here."""

    def train_predict(train_df, test_df):
        model, means, stds = train_model(train_df, max_candidates)
        return predict_with_model(model, test_df, max_candidates, means, stds)

    return train_predict


def main():
    print(f"seed={SEED}, TRAIN_SCOPE={TRAIN_SCOPE}")
    set_all_seeds(SEED)

    df = pd.read_parquet(os.path.join(OUT_DIR, "pass_candidates.parquet"))

    if TRAIN_SCOPE == "single_team":
        players = pd.read_csv(os.path.join(OUT_DIR, "players.csv"))
        team_id = players.loc[players["team_name"] == CONFIG["team_name"], "team_id"].iloc[0]
        df = df[df["team_id"] == team_id].copy()
    elif TRAIN_SCOPE != "all_matches":
        raise ValueError(f"unknown TRAIN_SCOPE: {TRAIN_SCOPE}")

    max_candidates = df.groupby("pass_id").size().max()
    n_matches = df["match_id"].nunique()
    print(f"{df['pass_id'].nunique()} passes, {len(df)} candidate rows, {n_matches} matches "
          f"(-> {n_matches}-fold leave-one-match-out), max {max_candidates} candidates/pass")

    predict_fn = make_train_predict_fn(max_candidates)
    fold_df = leave_one_match_out_cv(df, predict_fn, model_name=f"neural_net_{TRAIN_SCOPE}")
    print("\nPer-fold results:")
    print(fold_df[["match_id", "top1_acc", "top3_acc", "mrr", "cross_entropy"]].to_string(index=False))

    summary = fold_df[["top1_acc", "top3_acc", "mrr", "cross_entropy"]].agg(["mean", "std"])
    print(f"\n{'=' * 70}\nSTAGE 5 RESULTS: neural_net ({TRAIN_SCOPE}), mean +/- std across {n_matches} folds\n{'=' * 70}")
    for metric in ["top1_acc", "top3_acc", "mrr", "cross_entropy"]:
        print(f"  {metric}: {summary.loc['mean', metric]:.3f} +/- {summary.loc['std', metric]:.3f}")

    out_path = os.path.join(OUT_DIR, f"nn_fold_results_{TRAIN_SCOPE}.csv")
    fold_df.to_csv(out_path, index=False)
    print(f"\nWrote per-fold results to {out_path}")

    return fold_df


if __name__ == "__main__":
    main()
