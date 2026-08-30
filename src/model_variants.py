"""
EXPERIMENTAL, not part of the core Stage 1-6 pipeline: two architecture
variants tried in response to a direct question ("does RNN/LSTM/Transformer
make a difference"), compared against Stage 5's CandidateScorer MLP under
the exact same leave-one-match-out CV / metrics via src/experiment_architectures.py.

1. CandidateScorerAttention -- same static per-candidate features as the
   MLP, but candidates attend to EACH OTHER (via nn.TransformerEncoderLayer)
   before scoring, instead of being scored in complete isolation. Tests
   whether cross-candidate context ("this player is open because that one
   is dragging defenders away") helps -- something the plain MLP
   structurally cannot represent, no matter how big it's made.

2. CandidateScorerLSTM -- replaces the single velocity feature with an
   LSTM encoding of each candidate's raw ~1.2s trailing trajectory (see
   src/experiment_architectures.py's build_trajectory_tensor), concatenated
   with the remaining static features. Tests whether trajectory dynamics
   (acceleration, a curving run) carry signal the single first-derivative
   velocity feature smooths over.
"""

import torch
import torch.nn as nn


class CandidateScorerAttention(nn.Module):
    def __init__(self, n_features, model_dim=32, n_heads=4, n_layers=1, dropout=0.2):
        super().__init__()
        self.embed = nn.Linear(n_features, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads, dim_feedforward=model_dim * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(model_dim, model_dim), nn.ReLU(), nn.Linear(model_dim, 1))

    def forward(self, x, mask):
        """x: (batch, max_candidates, n_features), mask: (batch, max_candidates) bool, True=real candidate."""
        h = self.embed(x)
        # nn.TransformerEncoder's key_padding_mask is True=IGNORE, opposite of ours.
        h = self.encoder(h, src_key_padding_mask=~mask)
        scores = self.head(h).squeeze(-1)
        return scores


class CandidateScorerLSTM(nn.Module):
    def __init__(self, n_static_features, traj_dim=2, lstm_hidden=32, mlp_hidden=(64,), dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=traj_dim, hidden_size=lstm_hidden, batch_first=True)
        layers = []
        in_dim = n_static_features + lstm_hidden
        for h in mlp_hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, static_x, traj_x):
        """static_x: (batch, max_candidates, n_static_features)
        traj_x: (batch, max_candidates, traj_len, 2)"""
        batch, n_cand, traj_len, traj_dim = traj_x.shape
        flat_traj = traj_x.reshape(batch * n_cand, traj_len, traj_dim)
        _, (h_n, _) = self.lstm(flat_traj)
        traj_embed = h_n[-1].reshape(batch, n_cand, -1)  # last layer's final hidden state
        combined = torch.cat([static_x, traj_embed], dim=-1)
        scores = self.head(combined).squeeze(-1)
        return scores
