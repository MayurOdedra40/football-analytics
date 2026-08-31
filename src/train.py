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

NN_FEATURE_COLS = ALL_FEATURE_COLS

MAX_EPOCHS = 200
PATIENCE = 15
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
INTERNAL_VAL_FRACTION = 0.15  # validation fraction


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --- data: padded/masked tensors ------------------------

def passes_to_tensors(df, max_candidates, feature_means=None, feature_stds=None):
    """Converts a long-format DataFrame of pass candidates into padded/masked tensors."""
    df = df.copy()

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
    candidate_ids = [] 

    for i, pid in enumerate(pass_ids):
        group = df_norm[df_norm["pass_id"] == pid].reset_index(drop=True)
        n = len(group)
        X[i, :n, :] = group[NN_FEATURE_COLS].values
        mask[i, :n] = True
        target[i] = group.index[group["is_recipient"] == 1][0]
        candidate_ids.append(list(group["candidate_id"]) + [None] * (max_candidates - n))

    return {
        "X": torch.from_numpy(X), 
        "mask": torch.from_numpy(mask), 
        "target": torch.from_numpy(target),
        "pass_ids": pass_ids, 
        "candidate_ids": candidate_ids,
        "feature_means": feature_means, 
        "feature_stds": feature_stds,
    }


def masked_scores(model, X, mask):
    scores = model(X)
    return scores.masked_fill(~mask, float("-inf"))


def tensors_to_long_df(pass_ids, candidate_ids, probs, mask):
    """Converts the padded/masked tensors back into a long-format DataFrame of pass candidates with probabilities"""
    rows = []
    probs_np = probs.detach().numpy()
    mask_np = mask.numpy()
    for i, pid in enumerate(pass_ids):
        for j in range(mask_np.shape[1]):
            if not mask_np[i, j]:
                continue
            rows.append({"pass_id": pid, "candidate_id": candidate_ids[i][j], "prob": probs_np[i, j]})
    return pd.DataFrame(rows)


# --- train + predict -------------------------

def train_model(train_df, max_candidates):
    """Trains a CandidateScorer on the given training fold, returns the
    trained model and the feature means/stds used for scaling."""
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
    """Scores with an already-trained model. """
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
    """Returns a function that trains a model on a training fold and predicts on a test fold"""

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
    print(fold_df[["match_id", "top1_acc", "top3_acc", "cross_entropy"]].to_string(index=False))

    summary = fold_df[["top1_acc", "top3_acc", "cross_entropy"]].agg(["mean", "std"])
    print(f"\n\nRESULTS: neural_net ({TRAIN_SCOPE}), mean +/- std across {n_matches} folds")
    for metric in ["top1_acc", "top3_acc", "cross_entropy"]:
        print(f"  {metric}: {summary.loc['mean', metric]:.3f} +/- {summary.loc['std', metric]:.3f}")

    out_path = os.path.join(OUT_DIR, f"nn_fold_results_{TRAIN_SCOPE}.csv")
    fold_df.to_csv(out_path, index=False)
    print(f"\nWrote per-fold results to {out_path}")

    return fold_df


if __name__ == "__main__":
    main()
