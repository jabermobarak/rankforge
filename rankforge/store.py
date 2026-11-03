"""SQLite exposure and feedback ledger. One connection per operation; transactional writes."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


class Conflict(ValueError):
    pass


def assignment(user_id: str) -> str:
    return (
        "B"
        if int(hashlib.sha256(("rankforge-v1:" + user_id).encode()).hexdigest()[:8], 16) % 2
        else "A"
    )


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS exposures (
                    request_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                    variant TEXT NOT NULL, model_version TEXT NOT NULL,
                    items TEXT NOT NULL, created_at INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS feedback (
                    event_id TEXT PRIMARY KEY, request_id TEXT NOT NULL,
                    user_id TEXT NOT NULL, item_id TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    UNIQUE(request_id, item_id),
                    FOREIGN KEY(request_id) REFERENCES exposures(request_id));
                CREATE INDEX IF NOT EXISTS feedback_user ON feedback(user_id);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def exposure(self, request_id, user_id, variant, version, items):
        with self.connection() as db:
            db.execute(
                "INSERT INTO exposures VALUES (?,?,?,?,?,?)",
                (request_id, user_id, variant, version, json.dumps(items), int(time.time())),
            )

    def recent(self, user_id):
        with self.connection() as db:
            return [
                r[0]
                for r in db.execute(
                    "SELECT item_id FROM feedback WHERE user_id=? ORDER BY created_at DESC LIMIT 100",
                    (user_id,),
                )
            ]

    def feedback(self, event_id, request_id, user_id, item_id):
        with self.connection() as db:
            # Serialize duplicate checks across workers before inspecting the ledger.
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT * FROM feedback WHERE event_id=?", (event_id,)).fetchone()
            if existing:
                if (existing["request_id"], existing["user_id"], existing["item_id"]) != (
                    request_id,
                    user_id,
                    item_id,
                ):
                    raise Conflict("event_id already belongs to a different payload")
                return False
            exposure = db.execute(
                "SELECT * FROM exposures WHERE request_id=?", (request_id,)
            ).fetchone()
            if not exposure or exposure["user_id"] != user_id:
                raise ValueError("Unknown exposure for this user")
            if item_id not in json.loads(exposure["items"]):
                raise ValueError("Feedback item was not in this exposure")
            previous = db.execute(
                "SELECT 1 FROM feedback WHERE request_id=? AND item_id=?", (request_id, item_id)
            ).fetchone()
            if previous:
                return False
            db.execute(
                "INSERT INTO feedback VALUES (?,?,?,?,?)",
                (event_id, request_id, user_id, item_id, int(time.time())),
            )
            return True

    def summary(self):
        with self.connection() as db:
            rows = db.execute("""
                SELECT e.variant, COUNT(*) AS requests,
                    SUM(CASE WHEN EXISTS (SELECT 1 FROM feedback f
                        WHERE f.request_id=e.request_id) THEN 1 ELSE 0 END) AS clicked_requests
                FROM exposures e GROUP BY e.variant ORDER BY e.variant
            """).fetchall()
            events = db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        return {
            "variants": [
                {**dict(r), "request_ctr": r["clicked_requests"] / r["requests"]} for r in rows
            ],
            "feedback_events": events,
            "note": "Local observed demo traffic; no causal uplift or significance claim",
        }
