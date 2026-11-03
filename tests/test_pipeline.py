import copy
import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from rankforge.api import create_app
from rankforge.data import Dataset, export_csv, from_csv, synthetic, validate
from rankforge.engine import Engine
from rankforge.model import mmr_order, pairwise_loss
from rankforge.store import Conflict, Store, assignment
from rankforge.train import metrics, train


@pytest.fixture(scope="session")
def fixture_data():
    return synthetic(users=16, items=80)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, fixture_data):
    directory = tmp_path_factory.mktemp("artifact")
    train(fixture_data, directory, epochs=3)
    return directory


@pytest.fixture
def client(trained, tmp_path):
    with TestClient(create_app(trained, tmp_path / "events.db", api_token="")) as c:
        yield c


def test_global_split_and_reproducibility(fixture_data):
    a, b, c = fixture_data.split()
    assert max(e["timestamp"] for e in a) < min(e["timestamp"] for e in b)
    assert max(e["timestamp"] for e in b) < min(e["timestamp"] for e in c)
    assert fixture_data.fingerprint() == synthetic(users=16, items=80).fingerprint()


def test_csv_roundtrip(fixture_data, tmp_path):
    export_csv(fixture_data, tmp_path)
    loaded = from_csv(tmp_path, fixture_data.train_end, fixture_data.validation_end)
    assert loaded.items == fixture_data.items
    assert loaded.events == fixture_data.events


@pytest.mark.parametrize("mutation", ["duplicate", "unknown", "early", "empty"])
def test_bad_data_rejected(fixture_data, mutation):
    values = copy.deepcopy(fixture_data.to_dict())
    if mutation == "duplicate":
        values["events"].append(values["events"][0])
    elif mutation == "unknown":
        values["events"][0]["item_id"] = "missing"
    elif mutation == "early":
        values["events"][0]["timestamp"] = 0
    else:
        values["events"] = []
    with pytest.raises(ValueError):
        validate(Dataset(**values))


def test_metrics_have_known_values():
    result = metrics(["x", "a", "b"], {"a", "b"}, 3)
    assert result["recall"] == 1
    assert result["ndcg"] == pytest.approx((1 / np.log2(3) + 1 / np.log2(4)) / (1 + 1 / np.log2(3)))


def test_loss_pushes_positive_up():
    positive = torch.tensor([0.0], requires_grad=True)
    negative = torch.tensor([0.0], requires_grad=True)
    pairwise_loss(positive, negative).backward()
    assert positive.grad.item() < 0 < negative.grad.item()


def test_mmr_diversifies_and_respects_k():
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert mmr_order(np.array([1.0, 0.99, 0.9]), vectors, 2, 0) == [0, 1]
    assert mmr_order(np.array([1.0, 0.99, 0.9]), vectors, 2, 0.5) == [0, 2]


def test_recommendations_exclude_seen_and_future(trained):
    engine = Engine.load(trained)
    user = engine.users[0]
    rows, _ = engine.recommend(user, k=30)
    assert len({r["item_id"] for r in rows}) == len(rows)
    assert not {r["item_id"] for r in rows} & set(engine.histories[user])
    assert engine.recommend(user, now=0)[0] == []
    extra = [rows[0]["item_id"]]
    assert extra[0] not in {r["item_id"] for r in engine.recommend(user, recent=extra)[0]}


def test_cold_start_fallback(trained):
    rows, method = Engine.load(trained).recommend("new-visitor")
    assert rows and method == "popularity_fallback"


def test_model_selection_does_not_read_test_labels(fixture_data, trained, tmp_path):
    values = copy.deepcopy(fixture_data.to_dict())
    # Delete most test labels. Training and validation remain byte-identical.
    first_test = next(e for e in values["events"] if e["timestamp"] > values["validation_end"])
    values["events"] = [
        e for e in values["events"] if e["timestamp"] <= values["validation_end"]
    ] + [first_test]
    train(Dataset(**values), tmp_path, epochs=3)
    assert json.loads((trained / "model.json").read_text()) == json.loads(
        (tmp_path / "model.json").read_text()
    )


def test_health_and_readiness(client):
    assert client.get("/health").json() == {"status": "alive"}
    assert client.get("/ready").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_missing_artifacts_fail_readiness(tmp_path):
    with TestClient(create_app(tmp_path, tmp_path / "e.db")) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        assert c.get("/api/recommendations?user_id=u").status_code == 503


def test_api_validation(client):
    for query in ("k=0", "k=31", "diversity=0.7", "diversity=-1", "user_id="):
        prefix = "" if query.startswith("user_id") else "user_id=user-0000&"
        assert client.get("/api/recommendations?" + prefix + query).status_code == 422


def test_feedback_attribution_retry_and_exclusion(client):
    rec = client.get("/api/recommendations?user_id=user-0000").json()
    body = {
        "event_id": "event-1",
        "request_id": rec["request_id"],
        "user_id": rec["user_id"],
        "item_id": rec["items"][0]["item_id"],
    }
    assert client.post("/api/feedback", json=body).json()["inserted"] is True
    assert client.post("/api/feedback", json=body).json()["inserted"] is False
    assert (
        client.post("/api/feedback", json={**body, "event_id": "event-2"}).json()["inserted"]
        is False
    )
    assert client.post("/api/feedback", json={**body, "user_id": "other"}).status_code == 409
    bad = {**body, "event_id": "event-3", "item_id": "not-exposed"}
    assert client.post("/api/feedback", json=bad).status_code == 422
    wrong_user = {**body, "event_id": "event-4", "user_id": "wrong-user"}
    assert client.post("/api/feedback", json=wrong_user).status_code == 422
    next_feed = client.get("/api/recommendations?user_id=user-0000").json()
    assert body["item_id"] not in [r["item_id"] for r in next_feed["items"]]
    ledger = client.get("/api/experiments").json()
    assert ledger["feedback_events"] == 1
    assert ledger["variants"][0]["request_ctr"] == 0.5
    assert "rankforge_recommendations_total 2" in client.get("/metrics").text


def test_token_protects_read_and_write(trained, tmp_path):
    with TestClient(create_app(trained, tmp_path / "e.db", api_token="secret")) as c:
        for path in ("/api/users", "/api/report", "/api/experiments", "/metrics"):
            assert c.get(path).status_code == 401
            assert c.get(path, headers={"Authorization": "Bearer secret"}).status_code == 200
        assert c.post("/api/feedback", json={}).status_code == 401


def test_store_survives_restart_and_is_parameterized(tmp_path):
    path = tmp_path / "e.db"
    user = "x'; DROP TABLE feedback;--"
    store = Store(path)
    store.exposure("r", user, "B", "v", ["i"])
    store.feedback("e", "r", user, "i")
    reopened = Store(path)
    assert reopened.recent(user) == ["i"]
    assert reopened.summary()["feedback_events"] == 1
    with pytest.raises(Conflict):
        reopened.feedback("e", "r2", user, "i")


def test_assignment_stable_and_both_arms():
    assignments = [assignment(f"user-{n}") for n in range(100)]
    assert set(assignments) == {"A", "B"}
    assert assignments == [assignment(f"user-{n}") for n in range(100)]


def test_report_provenance(trained):
    report = json.loads((trained / "report.json").read_text())
    assert report["data_source"] == "synthetic_demo"
    assert len(report["dataset_sha256"]) == 64
    assert 1 <= report["selected_epoch"] <= 3
    assert report["selected_alpha"] in (0, 0.15, 0.3, 0.5)
    assert 0 <= report["candidate_recall_at_80"] <= 1
