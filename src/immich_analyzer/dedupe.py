"""Near-duplicate grouping over CLIP embeddings.

Comparing every pair in a 100k-photo library is ~5e9 dot products and finds
"two beach photos from different years". Duplicates that matter — burst
shots, re-saves, WhatsApp copies — are taken seconds apart, so we only compare
inside a sliding time window. Pairs above the cosine threshold are merged with
union-find; each group keeps its best photo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .store import Row


@dataclass(frozen=True)
class Group:
    keeper: Row
    extras: list[tuple[Row, float]]  # (row, cosine to keeper)


def _quality(r: Row) -> tuple:
    # Favorites first, then sharpest, then largest; id breaks ties deterministically.
    return (r.is_favorite, r.sharpness, r.pixels, r.id)


def find_groups(rows: list[Row], cosine: float, window_s: float) -> list[Group]:
    """``rows`` must be sorted by ``taken_ts``."""
    n = len(rows)
    if n < 2:
        return []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ts = np.array([r.taken_ts for r in rows])
    emb = np.stack([r.embedding for r in rows])
    for i in range(n):
        j_end = int(np.searchsorted(ts, ts[i] + window_s, side="right"))
        if j_end <= i + 1:
            continue
        sims = emb[i + 1 : j_end] @ emb[i]
        for off in np.nonzero(sims >= cosine)[0]:
            a, b = find(i), find(i + 1 + int(off))
            if a != b:
                parent[b] = a

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    groups = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        keeper_idx = max(members, key=lambda k: _quality(rows[k]))
        keeper = rows[keeper_idx]
        extras = [
            (rows[k], float(rows[k].embedding @ keeper.embedding))
            for k in members
            if k != keeper_idx
        ]
        groups.append(Group(keeper, extras))
    return groups
