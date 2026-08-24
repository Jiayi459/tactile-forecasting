"""Tactile-map -> F/CoP forecasters. Two per-frame encoders behind an IDENTICAL GRU + one-shot
head, so the encoder is the only variable in the flatten-vs-CNN comparison.

Input  x: (B, t_in, 2, 32, 32) normalized map history.
Output  : (B, H, 6) forecast of the next H steps of the 6-dim F/CoP target (normalized units).
"""
from __future__ import annotations

import torch
import torch.nn as nn

IN_CH, GRID = 2, 32
FLAT = IN_CH * GRID * GRID          # 2048


class FlattenEncoder(nn.Module):
    """Flatten each frame -> linear -> embedding (no spatial structure exploited)."""

    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(FLAT, d), nn.ReLU())

    def forward(self, x):                         # (B,t_in,2,32,32) -> (B,t_in,d)
        B, T = x.shape[:2]
        return self.proj(x.reshape(B * T, IN_CH, GRID, GRID)).reshape(B, T, -1)


class CNNEncoder(nn.Module):
    """Small conv stack per frame -> embedding (exploits spatial structure)."""

    def __init__(self, d: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(IN_CH, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),   # 32->16
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),   # 16->8
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, d), nn.ReLU())

    def forward(self, x):                         # (B,t_in,2,32,32) -> (B,t_in,d)
        B, T = x.shape[:2]
        return self.conv(x.reshape(B * T, IN_CH, GRID, GRID)).reshape(B, T, -1)


class AggEncoder(nn.Module):
    """Per-frame projection of the aggregate 6-dim F/CoP -> embedding. The neural (GRU) counterpart
    of the linear AR baseline: same raw 6-dim target, but input is the aggregate F/CoP history (no map)."""

    def __init__(self, d: int, n_in: int = 6):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_in, d), nn.ReLU())

    def forward(self, x):                         # (B,t_in,6) -> (B,t_in,d)
        return self.proj(x)


class Seq2Seq(nn.Module):
    """encoder -> GRU over t_in frames -> one-shot PROBABILISTIC head -> (mu, logvar), each (B,H,6).

    Predicts the RESIDUAL (change vs the last observed value) as a Gaussian per (step, channel):
    mean mu + log-variance lv. Trained with Gaussian NLL; lv clamped for stability."""

    def __init__(self, encoder: nn.Module, d: int, hidden: int, horizon: int, n_out: int = 6):
        super().__init__()
        self.encoder = encoder
        self.gru = nn.GRU(d, hidden, batch_first=True)
        self.mu = nn.Linear(hidden, horizon * n_out)
        self.lv = nn.Linear(hidden, horizon * n_out)
        self.H, self.n_out = horizon, n_out

    def forward(self, x):
        e = self.encoder(x)                       # (B,t_in,d)
        _, h = self.gru(e)                        # h: (1,B,hidden)
        last = h[-1]
        mu = self.mu(last).reshape(-1, self.H, self.n_out)
        lv = self.lv(last).clamp(-6, 4).reshape(-1, self.H, self.n_out)
        return mu, lv                             # (B,H,6), (B,H,6)


class ProbGRU(nn.Module):
    """encoder -> GRU -> action embedding -> AUTOREGRESSIVE decoder -> (mu, lv), each (B,H,C).

    The same backbone as src/opentouch/prob_gru.py, so the two sensors' probGRU arms are one
    model and their numbers can sit in one table: 8-dim action embedding, decoder seeded with
    the last observed target and fed its own mean, mu and log-variance over
    [decoder state ; embedding], logvar clamped to [-6, 4].

    Predicts the ABSOLUTE target, not the residual Seq2Seq predicts. That is deliberate and
    costs this arm something: a residual head can emit 0 and match persistence, a prior this
    one does not get. Giving one sensor's arm that prior and not the other's would make the
    comparison meaningless, which matters more (docs/model_comparability.md).

    `frame_encoder` is applied per frame before the encoder GRU, so flatten/cnn/aggregate
    differ in that module ALONE.
    """

    def __init__(self, encoder: nn.Module, d: int, hidden: int, horizon: int,
                 n_act: int, n_out: int = 6):
        super().__init__()
        self.frame_encoder = encoder
        self.emb = nn.Embedding(n_act, 8)
        self.enc = nn.GRU(d, hidden, batch_first=True)
        self.dec = nn.GRU(n_out, hidden, batch_first=True)
        self.mu = nn.Linear(hidden + 8, n_out)
        self.lv = nn.Linear(hidden + 8, n_out)
        self.H, self.n_out = horizon, n_out

    def forward(self, x, aid, y_last, t_out=None):
        _, h = self.enc(self.frame_encoder(x))
        e = self.emb(aid)
        inp = y_last.unsqueeze(1)
        mus, lvs = [], []
        for _ in range(t_out or self.H):
            o, h = self.dec(inp, h)
            oc = torch.cat([o[:, -1], e], -1)
            mu = self.mu(oc); lv = self.lv(oc).clamp(-6, 4)
            mus.append(mu); lvs.append(lv)
            inp = mu.unsqueeze(1)                   # autoregressive: feed the mean back
        return torch.stack(mus, 1), torch.stack(lvs, 1)


def build_model(encoder: str, horizon: int, d: int = 64, hidden: int = 64,
                backbone: str = "seq2seq", n_act: int = 1, n_out: int = 6):
    """encoder x backbone -> model. The encoder is the only thing that varies within a
    backbone, and the backbone is the only thing that varies across the two families."""
    enc = {"flatten": FlattenEncoder, "cnn": CNNEncoder, "aggregate": AggEncoder}[encoder](d)
    if backbone == "seq2seq":
        return Seq2Seq(enc, d, hidden, horizon, n_out)
    if backbone == "probgru":
        return ProbGRU(enc, d, hidden, horizon, n_act, n_out)
    raise ValueError(f"backbone must be seq2seq or probgru, got {backbone!r}")
