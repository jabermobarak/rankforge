"""Artifact loading, neural retrieval, profile ranking, and cold-start fallback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .model import TwoTower, mmr_order


class Engine:
    def __init__(self, artifact: dict):
        self.artifact = artifact
        self.version = artifact["version"]
        self.items = artifact["items"]
        self.item_map = {i["item_id"]: n for n, i in enumerate(self.items)}
        self.users = artifact["users"]
        self.user_map = {u: n + 1 for n, u in enumerate(self.users)}
        self.categories = artifact["categories"]
        self.cat_ids = np.array([self.categories.index(i["category"]) for i in self.items])
        self.histories = artifact["histories"]
        self.popularity = np.array(artifact["popularity"], dtype=float)
        self.alpha = artifact["alpha"]
        self.model = TwoTower(
            len(self.users), len(self.items), len(self.categories), artifact["dim"]
        )
        self.model.load_state_dict({k: torch.tensor(v) for k, v in artifact["weights"].items()})
        self.model.eval()
        with torch.inference_mode():
            self.vectors = self.model.item_vectors(
                torch.arange(len(self.items)), torch.tensor(self.cat_ids)
            ).numpy()

    @classmethod
    def load(cls, directory: Path):
        return cls(json.loads((directory / "model.json").read_text(encoding="utf-8")))

    def profile(self, user_id: str, recent: list[str] | None = None):
        history = self.histories.get(user_id, []) + (recent or [])
        indices = [self.item_map[i] for i in dict.fromkeys(history) if i in self.item_map]
        counts = np.bincount(self.cat_ids[indices], minlength=len(self.categories)).astype(float)
        return counts / max(counts.sum(), 1)

    def scores(self, user_id: str, recent: list[str] | None = None, strategy="neural"):
        profile = self.profile(user_id, recent)
        uid = self.user_map.get(user_id, 0)
        if strategy == "popular" or (uid == 0 and profile.sum() == 0):
            scores = self.popularity / max(self.popularity.max(), 1)
            return scores, "popularity_fallback" if uid == 0 else "popularity"
        if strategy == "category":
            return profile[self.cat_ids] + 0.01 * self.popularity / max(
                self.popularity.max(), 1
            ), "category"
        with torch.inference_mode():
            query = self.model.user_vectors(
                torch.tensor([uid]), torch.tensor(profile[None, :], dtype=torch.float32)
            ).numpy()[0]
        return self.vectors @ query, "two_tower"

    def recommend(
        self,
        user_id: str,
        k=10,
        diversity=0.0,
        recent=None,
        strategy="neural",
        now: int | None = None,
        extra_seen=None,
        alpha: float | None = None,
    ):
        scores, method = self.scores(user_id, recent, strategy)
        seen = set(self.histories.get(user_id, [])) | set(recent or []) | set(extra_seen or [])
        eligible = np.array(
            [
                n
                for n, i in enumerate(self.items)
                if i["item_id"] not in seen and (now is None or i["available_at"] <= now)
            ],
            dtype=int,
        )
        if not len(eligible):
            return [], method
        # Exact dot product retrieval; full-catalog evaluation uses the same candidate set size.
        candidates = eligible[np.argsort(-scores[eligible], kind="stable")[:80]]
        relevance = (scores[candidates] + 1) / 2 if method == "two_tower" else scores[candidates]
        if method == "two_tower":
            blend = self.alpha if alpha is None else alpha
            relevance = (1 - blend) * relevance + blend * self.profile(user_id, recent)[
                self.cat_ids[candidates]
            ]
        # MMR uses cosine similarity in learned space; score remains a ranking value, not probability.
        order = mmr_order(relevance, self.vectors[candidates], k, diversity)
        top_category = self.categories[int(np.argmax(self.profile(user_id, recent)))]
        rows = []
        for rank, idx in enumerate(order, 1):
            item = self.items[int(candidates[idx])]
            reason = (
                "Popular among training users"
                if method.startswith("popular")
                else f"Learned affinity; your history includes {top_category}"
            )
            rows.append(
                {**item, "rank": rank, "score": round(float(relevance[idx]), 6), "reason": reason}
            )
        return rows, method


def artifact_version(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
