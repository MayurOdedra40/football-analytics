import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.parse import CONFIG, OUT_DIR

SEED = CONFIG["seed"]

ALL_FEATURE_COLS = [
    "dx", "dy", "dist_to_passer", "angle_to_passer", "cand_x", "cand_y",
    "dist_to_opp_goal", "dist_to_own_goal", "dist_nearest_opponent", "n_opponents_within_5m",
]
DIST_ONLY_FEATURE_COLS = ["dist_to_passer"]

CROSS_ENTROPY = 1e-12 


# --- metrics -----------------------------------------

def softmax_within_pass(df, score_col, out_col="prob"):
    """Turns a raw per-candidate score into a probability distribution"""
    df = df.copy()
    df[out_col] = df.groupby("pass_id")[score_col].transform(lambda s: np.exp(s - s.max()))
    df[out_col] = df[out_col] / df.groupby("pass_id")[out_col].transform("sum")
    return df


def evaluate_predictions(df, prob_col="prob"):
    """returns top-1 accuracy, top-3 accuracy, cross-entropy."""
    g = df.groupby("pass_id")

    def per_pass(group):
        group = group.sort_values(prob_col, ascending=False).reset_index(drop=True)
        true_rank = group.index[group["is_recipient"] == 1][0] + 1  # 1-indexed
        p_true = group.loc[group["is_recipient"] == 1, prob_col].iloc[0]
        return pd.Series({
            "top1": int(true_rank == 1),
            "top3": int(true_rank <= 3),
            "cross_entropy": -np.log(np.clip(p_true, CROSS_ENTROPY, 1.0)),
        })

    per_pass_metrics = g.apply(per_pass, include_groups=False)
    return {
        "top1_acc": per_pass_metrics["top1"].mean(),
        "top3_acc": per_pass_metrics["top3"].mean(),
        "cross_entropy": per_pass_metrics["cross_entropy"].mean(),
    }


# --- the two baseline models ------------------------------------------------

def predict_nearest_teammate(test_df):
    """ Score = - dist_to_passer, softmaxed within the pass"""
    df = test_df.copy()
    df["score"] = -df["dist_to_passer"]
    return softmax_within_pass(df, "score")


def fit_predict_logreg(train_df, test_df):
    model = LogisticRegression(random_state=SEED)
    model.fit(train_df[DIST_ONLY_FEATURE_COLS], train_df["is_recipient"])
    test_df = test_df.copy()
    test_df["score"] = model.decision_function(test_df[DIST_ONLY_FEATURE_COLS])
    return softmax_within_pass(test_df, "score")


# --- leave-one-match-out CV ---------------------------

def leave_one_match_out_cv(df, predict_fn, model_name):
    """leave-one-match-out CV on the given DataFrame"""
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
    metric_cols = ["top1_acc", "top3_acc", "cross_entropy"]
    summary = fold_df.groupby("model")[metric_cols].agg(["mean", "std"])
    return summary


def main():
    df = pd.read_parquet(os.path.join(OUT_DIR, "pass_candidates.parquet"))
    print(f"{df['pass_id'].nunique()} passes, {len(df)} candidate rows, "
          f"{df['match_id'].nunique()} matches (leave-one-match-out folds)")

    models = {
        "nearest_teammate": lambda train_df, test_df: predict_nearest_teammate(test_df),
        "logistic_regression": fit_predict_logreg,
    }

    all_folds = []
    for name, fn in models.items():
        print(f"\nRunning leave-one-match-out CV: {name}...")
        fold_df = leave_one_match_out_cv(df, fn, name)
        print(fold_df[["match_id", "top1_acc", "top3_acc", "cross_entropy"]].to_string(index=False))
        all_folds.append(fold_df)

    all_folds_df = pd.concat(all_folds, ignore_index=True)
    out_path = os.path.join(OUT_DIR, "baseline_fold_results.csv")
    all_folds_df.to_csv(out_path, index=False)
    print(f"\nWrote per-fold results to {out_path}")

    print("\n" + "=" * 70)
    print("STAGE 4 RESULTS (mean +/- std across 7 leave-one-match-out folds)")
    print("=" * 70)
    summary = summarise_folds(all_folds_df)
    for model in ["nearest_teammate", "logistic_regression"]:
        row = summary.loc[model]
        print(f"\n{model}:")
        for metric in ["top1_acc", "top3_acc", "cross_entropy"]:
            print(f"  {metric}: {row[(metric, 'mean')]:.3f} +/- {row[(metric, 'std')]:.3f}")


if __name__ == "__main__":
    main()
