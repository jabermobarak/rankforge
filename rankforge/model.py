"""A small CPU-friendly two-tower model; pairwise implicit-feedback training."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TwoTower(nn.Module):
    def __init__(self, users: int, items: int, categories: int, dim: int = 24):
        super().__init__()
        self.user_embedding = nn.Embedding(users + 1, dim, padding_idx=0)
        self.item_embedding = nn.Embedding(items, dim)
        self.category_embedding = nn.Embedding(categories, dim)
        self.user_net = nn.Sequential(
            nn.Linear(dim + categories, dim), nn.Tanh(), nn.Linear(dim, dim)
        )
        self.item_net = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))

    def user_vectors(self, users, profiles):
        return F.normalize(
            self.user_net(torch.cat([self.user_embedding(users), profiles], -1)), dim=-1
        )

    def item_vectors(self, items, categories):
        return F.normalize(
            self.item_net(self.item_embedding(items) + self.category_embedding(categories)), dim=-1
        )

    def forward(self, users, profiles, items, categories):
        return (self.user_vectors(users, profiles) * self.item_vectors(items, categories)).sum(-1)


def pairwise_loss(positive, negative):
    return F.softplus(-(positive - negative) * 8).mean()


def mmr_order(scores: np.ndarray, vectors: np.ndarray, k: int, diversity: float):
    """Greedy MMR, stable ties. Input relevance must be scaled to [0,1]."""
    remaining = list(range(len(scores)))
    chosen = []
    while remaining and len(chosen) < k:
        novelty = (
            np.zeros(len(remaining))
            if not chosen
            else (vectors[remaining] @ vectors[chosen].T).max(axis=1)
        )
        objective = (1 - diversity) * scores[remaining] - diversity * novelty
        best = int(np.argmax(objective))
        chosen.append(remaining.pop(best))
    return chosen
