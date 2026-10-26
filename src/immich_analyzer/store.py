"""SQLite state: what has been analysed, why, and what has been synced.

Persisting results makes every run idempotent and resumable (a crashed
backfill skips work already done), keeps embeddings around for duplicate
checks against future uploads, and means an asset a human removed from an
album is never re-added (``synced`` stays 1).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    id          TEXT PRIMARY KEY,
    owner_id    TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    taken_ts    REAL NOT NULL,
    is_favorite INTEGER NOT NULL,
    pixels      INTEGER NOT NULL,
    sharpness   REAL NOT NULL,
    embedding   BLOB NOT NULL,
    label       TEXT NOT NULL,
    reason      TEXT NOT NULL,
    scores      TEXT NOT NULL,
    album       TEXT,
    synced      INTEGER NOT NULL DEFAULT 0,
    analyzed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS assets_owner_taken ON assets (owner_id, taken_ts);
CREATE INDEX IF NOT EXISTS assets_unsynced ON assets (owner_id, synced) WHERE album IS NOT NULL;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@dataclass
class Row:
    id: str
    taken_ts: float
    is_favorite: bool
    pixels: int
    sharpness: float
    embedding: np.ndarray
    label: str


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    # -- meta -----------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.db.commit()

    # -- assets ---------------------------------------------------------------
    def analyzed_ids(self, owner_id: str) -> set[str]:
        cur = self.db.execute("SELECT id FROM assets WHERE owner_id=?", (owner_id,))
        return {r[0] for r in cur}

    def insert(
        self,
        *,
        asset_id: str,
        owner_id: str,
        created_at: str,
        taken_ts: float,
        is_favorite: bool,
        pixels: int,
        sharpness: float,
        embedding: np.ndarray,
        label: str,
        reason: str,
        scores: dict,
        album: str | None,
    ) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO assets(id, owner_id, created_at, taken_ts, is_favorite, "
            "pixels, sharpness, embedding, label, reason, scores, album) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                asset_id,
                owner_id,
                created_at,
                taken_ts,
                int(is_favorite),
                pixels,
                sharpness,
                embedding.astype(np.float32).tobytes(),
                label,
                reason,
                json.dumps(scores, sort_keys=True),
                album,
            ),
        )

    def commit(self) -> None:
        self.db.commit()

    def dedupe_candidates(self, owner_id: str, ts_min: float, ts_max: float) -> list[Row]:
        """Camera photos in a time range — receipts/screenshots never dedupe."""
        cur = self.db.execute(
            "SELECT id, taken_ts, is_favorite, pixels, sharpness, embedding, label FROM assets "
            "WHERE owner_id=? AND taken_ts BETWEEN ? AND ? "
            "AND label NOT IN ('receipt', 'screenshot') ORDER BY taken_ts",
            (owner_id, ts_min, ts_max),
        )
        return [
            Row(r[0], r[1], bool(r[2]), r[3], r[4], np.frombuffer(r[5], dtype=np.float32), r[6])
            for r in cur
        ]

    def mark_duplicate(self, asset_id: str, keeper_id: str, cosine: float, album: str) -> None:
        self.db.execute(
            "UPDATE assets SET label='duplicate', reason=?, album=? WHERE id=? AND label='keep'",
            (f"near-duplicate of {keeper_id} (cos={cosine:.3f})", album, asset_id),
        )

    def unsynced(self, owner_id: str) -> dict[str, list[str]]:
        by_album: dict[str, list[str]] = {}
        cur = self.db.execute(
            "SELECT album, id FROM assets WHERE owner_id=? AND album IS NOT NULL AND synced=0",
            (owner_id,),
        )
        for album, asset_id in cur:
            by_album.setdefault(album, []).append(asset_id)
        return by_album

    def mark_synced(self, ids: Iterable[str]) -> None:
        self.db.executemany("UPDATE assets SET synced=1 WHERE id=?", [(i,) for i in ids])
        self.db.commit()

    def label_counts(self) -> dict[str, int]:
        cur = self.db.execute("SELECT label, COUNT(*) FROM assets GROUP BY label")
        return dict(cur.fetchall())

    def samples(self, label: str, limit: int = 10) -> list[tuple[str, str]]:
        cur = self.db.execute(
            "SELECT id, reason FROM assets WHERE label=? ORDER BY analyzed_at DESC LIMIT ?",
            (label, limit),
        )
        return cur.fetchall()
