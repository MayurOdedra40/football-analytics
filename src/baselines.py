"""
Stage 4: baselines, established before any neural network.

    1. Nearest teammate     -- predict argmin(dist_to_passer). No training.
    2. Logistic regression  -- on dist_to_passer alone, softmaxed within pass.
    3. Gradient boosting    -- LightGBM on the full feature set, softmaxed within pass.

All three (and Stage 5's neural net, which imports `evaluate_predictions` and
`leave_one_match_out_cv` from here) are scored the same way: top-1 accuracy,
top-3 accuracy, mean reciprocal rank, and cross-entropy, under leave-one-
match-out CV (7 folds -- never a random pass-level split, which would leak
team identity and adjacent-possession passes across train/test; see README).

Run: `python -m src.baselines` (from repo root, venv active).
"""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.parse import CONFIG, OUT_DIR

SEED = CONFIG["seed"]

# The full engineered feature set from Stage 3, in the order build_dataset.py
# writes them (dist_rank included: it's derived from dist_to_passer within
# the pass's own candidate set, which is legitimately known before the pass
# happens -- not future information, so not a leak, just a correlated
# feature; harmless for a tree model like LightGBM).
ALL_FEATURE_COLS = [
    "dx", "dy", "dist_to_passer", "angle_to_passer", "cand_x", "cand_y",
    "cand_vx", "cand_vy", "cand_speed", "dist_to_opp_goal", "dist_to_own_goal",
    "dist_nearest_opponent", "n_opponents_within_5m", "n_opponents_in_lane",
    "min_perp_dist_in_lane", "is_progressive", "passer_pressure", "dist_rank",
]
DIST_ONLY_FEATURE_COLS = ["dist_to_passer"]

CROSS_ENTROPY_EPS = 1e-12  # clip so a confident wrong prediction gives a large but finite loss, not inf


# --- metrics, shared with Stage 5 -----------------------------------------

def softmax_within_pass(df, score_col, out_col="prob"):
    """Turns a raw per-candidate score into a probability distribution that
    sums to 1 within each pass_id group -- the same normalisation Stage 5's
    neural net applies over its own per-candidate scores, so every model in
    the comparison table is evaluated on a genuinely comparable quantity."""
    df = df.copy()
    df[out_col] = df.groupby("pass_id")[score_col].transform(lambda s: np.exp(s - s.max()))
    df[out_col] = df[out_col] / df.groupby("pass_id")[out_col].transform("sum")
    return df


def evaluate_predictions(df, prob_col="prob"):
    """df must have pass_id, is_recipient, and a probability column that
    sums to 1 within each pass_id group. Returns the four headline metrics."""
    g = df.groupby("pass_id")

    def per_pass(group):
        group = group.sort_values(prob_col, ascending=False).reset_index(drop=True)
        true_rank = group.index[group["is_recipient"] == 1][0] + 1  # 1-indexed
        p_true = group.loc[group["is_recipient"] == 1, prob_col].iloc[0]
        return pd.Series({
            "top1": int(true_rank == 1),
            "top3": int(true_rank <= 3),
            "reciprocal_rank": 1.0 / true_rank,
            "cross_entropy": -np.log(np.clip(p_true, CROSS_ENTROPY_EPS, 1.0)),
        })

    per_pass_metrics = g.apply(per_pass, include_groups=False)
    return {
        "top1_acc": per_pass_metrics["top1"].mean(),
        "top3_acc": per_pass_metrics["top3"].mean(),
        "mrr": per_pass_metrics["reciprocal_rank"].mean(),
        "cross_entropy": per_pass_metrics["cross_entropy"].mean(),
    }


# --- the three baseline models --------------------------------------------

def predict_nearest_teammate(test_df):
    """No training: score = -dist_to_passer, softmaxed within the pass.
    top1_acc from this is EXACTLY the spec's "predict argmin(dist_to_passer)"
    rule, since softmax preserves ranking -- the max-probability candidate
    is always the nearest one. top3_acc/MRR need a full ranking, not just
    the single argmin pick, so scoring every candidate by raw (negative)
    distance rather than one-hot-encoding only the winner is what makes
    those numbers meaningful instead of an arbitrary tie-break among a pile
    of zero-probability candidates. Cross-entropy then falls out of the
    same softmax everything else in this module uses, at a fixed "1 logit
    unit per metre" scale -- unlike logistic_regression below, nothing here
    is fit to data, which is the actual point of comparing the two."""
    df = test_df.copy()
    df["score"] = -df["dist_to_passer"]
    return softmax_within_pass(df, "score")


def fit_predict_logreg(train_df, test_df):
    model = LogisticRegression(random_state=SEED)
    model.fit(train_df[DIST_ONLY_FEATURE_COLS], train_df["is_recipient"])
    # decision_function is the raw logit -- softmax that within each pass,
    # not model.predict_proba (which is P(recipient) vs P(not recipient) for
    # each row independently and does not sum to 1 across a pass's
    # candidates).
    test_df = test_df.copy()
    test_df["score"] = model.decision_function(test_df[DIST_ONLY_FEATURE_COLS])
    return softmax_within_pass(test_df, "score")


def fit_predict_lightgbm(train_df, test_df):
    train_set = lgb.Dataset(train_df[ALL_FEATURE_COLS], label=train_df["is_recipient"])
    params = {
        "objective": "binary",
        "num_leaves": 15,       # small on purpose: ~40k training rows across 6 matches, not a huge dataset
        "min_data_in_leaf": 30,
        "learning_rate": 0.05,
        "verbose": -1,
        "seed": SEED,
        "deterministic": True,
    }
    booster = lgb.train(params, train_set, num_boost_round=200)
    test_df = test_df.copy()
    test_df["score"] = booster.predict(test_df[ALL_FEATURE_COLS], raw_score=True)
    return softmax_within_pass(test_df, "score")


# --- leave-one-match-out CV, shared with Stage 5 ---------------------------

def leave_one_match_out_cv(df, predict_fn, model_name):
    """predict_fn(train_df, test_df) -> test_df with a 'prob' column
    summing to 1 per pass_id. Returns a DataFrame of per-fold metrics
    (one row per held-out match) plus mean/std, for the results table."""
    match_ids = sorted(df["match_id"].unique())
    fold_rows = []
    for held_out in match_ids:
        train_df = df[df["match_id"] != held_out]
        test_df = df[df["match_id"] == held_out]
        pred_df = predict_fn(train_df, test_df)
        metrics = evaluate_predictions(pred_df)
        metrics["match_id"] = held_out
        metrics["model"] = model_name
        fold_rows.append(metrics)
    return pd.DataFrame(fold_rows)


def summarise_folds(fold_df):
    metric_cols = ["top1_acc", "top3_acc", "mrr", "cross_entropy"]
    summary = fold_df.groupby("model")[metric_cols].agg(["mean", "std"])
    return summary


def main():
    df = pd.read_parquet(os.path.join(OUT_DIR, "pass_candidates.parquet"))
    print(f"{df['pass_id'].nunique()} passes, {len(df)} candidate rows, "
          f"{df['match_id'].nunique()} matches (leave-one-match-out folds)")

    models = {
        "nearest_teammate": lambda train_df, test_df: predict_nearest_teammate(test_df),
        "logistic_regression": fit_predict_logreg,
        "lightgbm": fit_predict_lightgbm,
    }

    all_folds = []
    for name, fn in models.items():
        print(f"\nRunning leave-one-match-out CV: {name}...")
        fold_df = leave_one_match_out_cv(df, fn, name)
        print(fold_df[["match_id", "top1_acc", "top3_acc", "mrr", "cross_entropy"]].to_string(index=False))
        all_folds.append(fold_df)

    all_folds_df = pd.concat(all_folds, ignore_index=True)
    out_path = os.path.join(OUT_DIR, "baseline_fold_results.csv")
    all_folds_df.to_csv(out_path, index=False)
    print(f"\nWrote per-fold results to {out_path}")

    print("\n" + "=" * 70)
    print("STAGE 4 RESULTS (mean +/- std across 7 leave-one-match-out folds)")
    print("=" * 70)
    summary = summarise_folds(all_folds_df)
    for model in ["nearest_teammate", "logistic_regression", "lightgbm"]:
        row = summary.loc[model]
        print(f"\n{model}:")
        for metric in ["top1_acc", "top3_acc", "mrr", "cross_entropy"]:
            print(f"  {metric}: {row[(metric, 'mean')]:.3f} +/- {row[(metric, 'std')]:.3f}")

    print("\n" + "=" * 70)
    print("STAGE 4 SANITY CHECKS")
    print("=" * 70)
    # nearest_teammate and logistic_regression are both distance-only rules
    # -- they should be close (logreg is a smoothed version of the same
    # signal), and lightgbm (full feature set) should beat both, or Stage 4
    # would be telling us the extra features aren't earning their keep.
    nt_top1 = summary.loc["nearest_teammate", ("top1_acc", "mean")]
    lr_top1 = summary.loc["logistic_regression", ("top1_acc", "mean")]
    gbm_top1 = summary.loc["lightgbm", ("top1_acc", "mean")]
    print(f"[INFO] top1_acc: nearest_teammate={nt_top1:.3f}, logistic_regression={lr_top1:.3f}, "
          f"lightgbm={gbm_top1:.3f}")
    print(f"[{'OK' if abs(nt_top1 - lr_top1) < 0.05 else 'NOTE'}] nearest_teammate and logistic_regression "
          f"within 5pp of each other (both are distance-only rules, expected to be close)")
    print(f"[{'OK' if gbm_top1 >= nt_top1 else 'NOTE'}] lightgbm (full features) >= nearest_teammate "
          f"(distance only) -- {'the extra features are earning their keep' if gbm_top1 >= nt_top1 else 'extra features are NOT helping -- worth investigating before Stage 5'}")


if __name__ == "__main__":
    main()
