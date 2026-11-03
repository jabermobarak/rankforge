"""Chronological training and full-catalog, user-macro evaluation."""

from __future__ import annotations

import copy
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import Dataset, validate
from .engine import Engine, artifact_version
from .model import TwoTower, pairwise_loss


def group(events):
    result = defaultdict(list)
    for e in events:
        result[e["user_id"]].append(e["item_id"])
    return dict(result)


def metrics(ranked: list[str], relevant: set[str], k=10):
    hits = np.array([float(i in relevant) for i in ranked[:k]])
    dcg = float((hits / np.log2(np.arange(len(hits)) + 2)).sum())
    ideal = float((1 / np.log2(np.arange(min(k, len(relevant))) + 2)).sum())
    return {
        "recall": float(hits.sum() / max(len(relevant), 1)),
        "ndcg": dcg / ideal if ideal else 0.0,
    }


def evaluate(engine: Engine, events, cutoff: int, strategy="neural", extra_seen=None, alpha=None):
    truth = group(events)
    rows, coverage = [], set()
    cold = 0
    for user, relevant in truth.items():
        cold += int(user not in engine.user_map)
        recs, _ = engine.recommend(
            user,
            k=10,
            strategy=strategy,
            now=cutoff,
            extra_seen=(extra_seen or {}).get(user, []),
            alpha=alpha,
        )
        ranked = [r["item_id"] for r in recs]
        coverage.update(ranked)
        rows.append(metrics(ranked, set(relevant)))
    return {
        "recall_at_10": float(np.mean([r["recall"] for r in rows])),
        "ndcg_at_10": float(np.mean([r["ndcg"] for r in rows])),
        "catalog_coverage_at_10": len(coverage) / len(engine.items),
        "users": len(rows),
        "cold_users": cold,
    }


def train(data: Dataset, output: Path, seed=42, epochs=30, dim=24):
    validate(data)
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_events, val_events, test_events = data.split()
    users = sorted({e["user_id"] for e in train_events})
    user_map = {u: n + 1 for n, u in enumerate(users)}
    items = data.items
    item_map = {i["item_id"]: n for n, i in enumerate(items)}
    categories = sorted({i["category"] for i in items})
    cat_ids = np.array([categories.index(i["category"]) for i in items])
    histories = group(train_events)
    ui = np.array([(user_map[e["user_id"]], item_map[e["item_id"]]) for e in train_events])
    profiles = np.zeros((len(users) + 1, len(categories)), dtype=np.float32)
    seen = {user_map[u]: {item_map[i] for i in rows} for u, rows in histories.items()}
    for u, indices in seen.items():
        counts = np.bincount(cat_ids[list(indices)], minlength=len(categories))
        profiles[u] = counts / counts.sum()
    eligible = {n for n, item in enumerate(items) if item["available_at"] <= data.train_end}
    negative_pool = {u: np.array(sorted(eligible - indices)) for u, indices in seen.items()}
    model = TwoTower(len(users), len(items), len(categories), dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.01)
    popularity = np.bincount(ui[:, 1], minlength=len(items)).tolist()
    base = {
        "schema_version": 1,
        "items": items,
        "users": users,
        "categories": categories,
        "histories": histories,
        "popularity": popularity,
        "dim": dim,
        "alpha": 0.0,
    }
    best, best_weights, log = -1.0, None, []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in np.array_split(rng.permutation(len(ui)), max(1, int(np.ceil(len(ui) / 512)))):
            u, pos = ui[batch].T
            neg = np.array([rng.choice(negative_pool[int(x)]) for x in u])
            user_tensor = torch.tensor(u)
            profile_tensor = torch.tensor(profiles[u])
            positive = model(
                user_tensor, profile_tensor, torch.tensor(pos), torch.tensor(cat_ids[pos])
            )
            negative = model(
                user_tensor, profile_tensor, torch.tensor(neg), torch.tensor(cat_ids[neg])
            )
            loss = pairwise_loss(positive, negative)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(batch)
        artifact = {
            **base,
            "version": "training",
            "weights": {k: v.detach().tolist() for k, v in model.state_dict().items()},
        }
        validation = evaluate(Engine(artifact), val_events, data.validation_end)
        log.append(dict(epoch=epoch + 1, loss=total / len(ui), **validation))
        if validation["ndcg_at_10"] > best:
            best, best_weights = validation["ndcg_at_10"], copy.deepcopy(artifact["weights"])
    artifact = {**base, "weights": best_weights, "version": "selection"}
    engine = Engine(artifact)
    # Checkpoint and blend selection use validation only. Test is read below, once selection ends.
    blend_trials = {
        str(a): evaluate(engine, val_events, data.validation_end, alpha=a)
        for a in (0.0, 0.15, 0.3, 0.5)
    }
    artifact["alpha"] = float(max(blend_trials, key=lambda a: blend_trials[a]["ndcg_at_10"]))
    artifact["version"] = artifact_version({k: v for k, v in artifact.items() if k != "version"})
    engine = Engine(artifact)
    test_cutoff = max(e["timestamp"] for e in test_events)
    test_results = {
        strategy: evaluate(engine, test_events, test_cutoff, strategy, extra_seen=group(val_events))
        for strategy in ("popular", "category", "neural")
    }
    candidate_recalls = []
    for user, relevant in group(test_events).items():
        scores, _ = engine.scores(user)
        excluded = set(engine.histories.get(user, [])) | set(group(val_events).get(user, []))
        eligible_indices = [
            n
            for n, i in enumerate(items)
            if i["item_id"] not in excluded and i["available_at"] <= test_cutoff
        ]
        candidates = sorted(eligible_indices, key=lambda n: -scores[n])[:80]
        candidate_recalls.append(
            len({items[n]["item_id"] for n in candidates} & set(relevant)) / len(relevant)
        )
    latencies = []
    for user in users[:100]:
        t = time.perf_counter()
        engine.recommend(user)
        latencies.append((time.perf_counter() - t) * 1000)
    report = {
        "model_version": artifact["version"],
        "data_source": data.source,
        "seed": seed,
        "dataset_sha256": data.fingerprint(),
        "train_end": data.train_end,
        "validation_end": data.validation_end,
        "test_end": test_cutoff,
        "sizes": {
            "users": len(users),
            "items": len(items),
            "train": len(train_events),
            "validation": len(val_events),
            "test": len(test_events),
        },
        "protocol": "Global chronological split; full catalog; no sampled evaluation negatives; user-macro metrics; validation-only selection",
        "selected_epoch": max(log, key=lambda r: r["ndcg_at_10"])["epoch"],
        "selected_alpha": artifact["alpha"],
        "validation_blend_trials": blend_trials,
        "test": test_results,
        "candidate_recall_at_80": float(np.mean(candidate_recalls)),
        "training_seconds": time.perf_counter() - started,
        "inference_ms": {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "n": len(latencies),
            "scope": "In-process CPU; excludes HTTP, database and network",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "machine": platform.machine(),
        },
        "epochs": log,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "model.json").write_text(
        json.dumps(artifact, separators=(",", ":")), encoding="utf-8"
    )
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
