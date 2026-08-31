import torch
import torch.nn as nn


class CandidateScorer(nn.Module):
    """A simple feedforward network that scores each candidate for a pass"""

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
        """x: (batch, max_candidates, n_features) -> scores: (batch, max_candidates)"""
        batch, n_cand, n_feat = x.shape
        flat = x.reshape(batch * n_cand, n_feat)
        scores = self.net(flat).reshape(batch, n_cand)
        return scores
