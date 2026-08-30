"""
Stage 5: the shared-weight candidate scorer.

    for each pass:
        for each candidate i:  score_i = MLP(features_i)   # same MLP, shared weights
        p = softmax([score_1 ... score_n])                  # over that pass's candidates
    loss = cross-entropy against the true recipient

Why this architecture, not a fixed-N classifier (a 10-way softmax head, one
output per "candidate slot"): a fixed classifier bakes in a fixed, ordered
player set, but the roster changes between matches and even within one
(substitutions), and "candidate slot 3" is a different person in every
match. Scoring each candidate through the SAME small MLP and only combining
scores at the softmax step is permutation-invariant (candidate order
doesn't matter, matching reality -- there's no natural order to "the other
10 players on the pitch") and naturally handles a variable candidate count
via the padding+mask mechanism in src/train.py. It's the DeepSets idea
(a shared per-element function, combined by a symmetric reduction --
here, softmax rather than DeepSets' usual sum/mean) applied to ranking.
"""

import torch
import torch.nn as nn


class CandidateScorer(nn.Module):
    """A small MLP applied independently to every candidate's feature
    vector -- the "shared weights" in the docstring above are literally
    just this one nn.Sequential, reused for every candidate slot."""

    def __init__(self, n_features, hidden=(128, 64), dropout=0.2):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: (batch, max_candidates, n_features) -> scores: (batch, max_candidates).
        Padded candidate slots are scored too (garbage in, garbage out) --
        train.py masks them to -inf before softmax/loss, so their score
        here never affects a prediction or a gradient."""
        batch, n_cand, n_feat = x.shape
        flat = x.reshape(batch * n_cand, n_feat)
        scores = self.net(flat).reshape(batch, n_cand)
        return scores
