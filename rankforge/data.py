"""Deterministic synthetic fixture and strict CSV contracts.

Catalog: item_id,title,category,available_at
Interactions: user_id,item_id,timestamp (positive implicit feedback, integer UTC seconds)
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CATEGORIES = ("Outdoors", "Audio", "Home", "Fitness", "Books", "Travel", "Art", "Tech")


@dataclass(frozen=True)
class Dataset:
    items: list[dict]
    events: list[dict]
    train_end: int
    validation_end: int
    source: str = "custom_csv"

    def split(self):
        train = [e for e in self.events if e["timestamp"] <= self.train_end]
        val = [e for e in self.events if self.train_end < e["timestamp"] <= self.validation_end]
        test = [e for e in self.events if e["timestamp"] > self.validation_end]
        return train, val, test

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self):
        return {
            "items": self.items,
            "events": self.events,
            "train_end": self.train_end,
            "validation_end": self.validation_end,
            "source": self.source,
        }


def validate(data: Dataset) -> Dataset:
    ids = [i["item_id"] for i in data.items]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Catalog must contain unique nonempty item IDs")
    catalog = {i["item_id"]: i for i in data.items}
    for item in data.items:
        if not all(
            isinstance(item[k], str) and 0 < len(item[k]) <= 160
            for k in ("item_id", "title", "category")
        ):
            raise ValueError("Catalog strings must have 1..160 characters")
    if data.train_end >= data.validation_end:
        raise ValueError("train_end must be before validation_end")
    seen = set()
    for e in data.events:
        if not isinstance(e["user_id"], str) or not 0 < len(e["user_id"]) <= 80:
            raise ValueError("user_id must have 1..80 characters")
        if e["item_id"] not in catalog:
            raise ValueError("Event references unknown item")
        if e["timestamp"] < catalog[e["item_id"]]["available_at"]:
            raise ValueError("Event precedes catalog availability")
        key = e["user_id"], e["item_id"]
        if key in seen:
            raise ValueError(
                "Duplicate user-item positives: aggregate repeated events before import"
            )
        seen.add(key)
    if any(not part for part in data.split()):
        raise ValueError("Global train, validation, and test windows must all be nonempty")
    train, _, _ = data.split()
    train_users = {e["user_id"] for e in train}
    eligible = {i["item_id"] for i in data.items if i["available_at"] <= data.train_end}
    if not train_users or len(eligible) < 3:
        raise ValueError("Training requires users and at least three available items")
    for u in train_users:
        if eligible <= {e["item_id"] for e in train if e["user_id"] == u}:
            raise ValueError("Every training user needs an unobserved eligible negative")
    return data


def synthetic(seed: int = 42, users: int = 400, items: int = 320) -> Dataset:
    if users < 8 or items < 80:
        raise ValueError("Synthetic fixture needs at least 8 users and 80 items")
    rng = np.random.default_rng(seed)
    start = 1_770_000_000
    catalog = [
        {
            "item_id": f"item-{i:04d}",
            "title": f"{CATEGORIES[i % 8]} / Edition {i // 8 + 1:02d}",
            "category": CATEGORIES[i % 8],
            "available_at": start - 1000,
        }
        for i in range(items)
    ]
    cats = np.arange(items) % 8
    item_style = rng.normal(size=(items, 6))
    popularity = rng.normal(scale=0.25, size=items)
    events = []
    for u in range(users):
        primary, secondary = rng.choice(8, size=2, replace=False)
        preference = np.zeros(8)
        preference[primary], preference[secondary] = 3.8, 2.0
        style = rng.normal(size=6)
        utility = preference[cats] + 0.3 * (item_style @ style) + popularity
        # Gumbel sampling without replacement: hidden preferences are never written to the dataset.
        order = np.argsort(-(utility + rng.gumbel(size=items)))[:40]
        for step, idx in enumerate(order):
            events.append(
                {
                    "user_id": f"user-{u:04d}",
                    "item_id": catalog[idx]["item_id"],
                    "timestamp": start + step * 86400 + u,
                }
            )
    events.sort(key=lambda e: (e["timestamp"], e["user_id"]))
    return validate(
        Dataset(
            catalog,
            events,
            start + 27 * 86400 + users,
            start + 33 * 86400 + users,
            "synthetic_demo",
        )
    )


def export_csv(data: Dataset, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("catalog.csv", data.items), ("interactions.csv", data.events)):
        with (directory / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def from_csv(directory: Path, train_end: int, validation_end: int) -> Dataset:
    with (directory / "catalog.csv").open(encoding="utf-8", newline="") as f:
        items = [
            {
                "item_id": r["item_id"],
                "title": r["title"],
                "category": r["category"],
                "available_at": int(r["available_at"]),
            }
            for r in csv.DictReader(f)
        ]
    with (directory / "interactions.csv").open(encoding="utf-8", newline="") as f:
        events = [
            {"user_id": r["user_id"], "item_id": r["item_id"], "timestamp": int(r["timestamp"])}
            for r in csv.DictReader(f)
        ]
    return validate(
        Dataset(items, sorted(events, key=lambda e: e["timestamp"]), train_end, validation_end)
    )
