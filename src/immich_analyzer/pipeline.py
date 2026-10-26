"""One analysis run for one Immich user.

list assets ─► fetch previews ─► sharpness + CLIP embed (GPU, batched)
    ─► zero-shot classify ─► persist ─► dedupe in time windows
    ─► sync albums (unless dry-run) ─► advance watermark
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum

import httpx
import numpy as np

from .classify import BLUR_PROMPTS, PROMPTS, Label, PromptBank, classify
from .config import Config
from .dedupe import find_groups
from .images import (
    ApiPreviewSource,
    NfsPreviewSource,
    PreviewSource,
    downscale_for_clip,
    sharpness,
)
from .immich import Asset, ImmichClient, parse_ts
from .inference import EmbeddingCountMismatch, InferenceClient
from .store import Store

log = logging.getLogger(__name__)

ALBUM_DESCRIPTIONS = {
    "receipts": "Receipts found by roboflow-immich-analyzer (CLIP zero-shot).",
    "screenshots": "Screenshots found by roboflow-immich-analyzer.",
    "safe_to_delete": (
        "Blurry photos and near-duplicate extras found by roboflow-immich-analyzer. "
        "Nothing is deleted automatically: review, then delete what you agree with."
    ),
}


class Mode(StrEnum):
    AUTO = "auto"
    BACKFILL = "backfill"
    INCREMENTAL = "incremental"


@dataclass
class Summary:
    user: str
    mode: str
    source: str
    listed: int = 0
    analyzed: int = 0
    missing_preview: int = 0
    errors: int = 0
    labels: Counter = field(default_factory=Counter)
    synced: int = 0
    seconds: float = 0.0


@dataclass
class _Result:
    asset: Asset
    jpeg_ok: bool
    sharpness: float = 0.0
    embedding: np.ndarray | None = None
    error: str | None = None  # permanent per-asset failure; recorded, not retried


class Analyzer:
    def __init__(self, cfg: Config, store: Store, inference: InferenceClient):
        self.cfg = cfg
        self.store = store
        self.inference = inference
        self._content: PromptBank | None = None
        self._blur: PromptBank | None = None

    def _banks(self) -> tuple[PromptBank, PromptBank]:
        if self._content is None or self._blur is None:
            self._content = PromptBank.build(PROMPTS, self.inference.embed_texts)
            self._blur = PromptBank.build(BLUR_PROMPTS, self.inference.embed_texts)
        return self._content, self._blur

    def _album_for(self, label: Label) -> str | None:
        a = self.cfg.albums
        return {
            Label.RECEIPT: a.receipts,
            Label.SCREENSHOT: a.screenshots,
            Label.BLURRY: a.safe_to_delete,
            Label.DUPLICATE: a.safe_to_delete,
        }.get(label)

    # ------------------------------------------------------------------------
    def run(self, immich: ImmichClient, mode: Mode = Mode.AUTO) -> Summary:
        started = time.monotonic()
        me = immich.whoami()
        owner = me["id"]
        wm_key = f"watermark:{owner}"
        watermark = self.store.get_meta(wm_key)

        if mode is Mode.AUTO:
            mode = Mode.BACKFILL if watermark is None else Mode.INCREMENTAL
        api_source = ApiPreviewSource(immich)
        source: PreviewSource = api_source
        if mode is Mode.BACKFILL and self.cfg.library_root:
            source = NfsPreviewSource(self.cfg.library_root, fallback=api_source)

        summary = Summary(user=me.get("email", owner), mode=mode.value, source=source.name)
        log.info(
            "run start",
            extra={
                "user": summary.user,
                "mode": mode.value,
                "source": source.name,
                "watermark": watermark,
            },
        )

        done = self.store.analyzed_ids(owner)
        since = None if mode is Mode.BACKFILL else watermark
        todo, max_created = [], watermark
        for a in immich.iter_images(created_after=since):
            summary.listed += 1
            if max_created is None or parse_ts(a.created_at) > parse_ts(max_created):
                max_created = a.created_at
            if a.id not in done:
                todo.append(a)
        log.info("listed", extra={"listed": summary.listed, "todo": len(todo)})

        missing: list[Asset] = []
        if todo:
            missing = self._analyze(todo, source, owner, summary)
            ts = [a.taken_ts for a in todo]
            self._dedupe(owner, min(ts), max(ts), summary)

        if not self.cfg.dry_run:
            summary.synced = self._sync_albums(immich, owner)

        # Never move the watermark past an asset we couldn't see yet (Immich
        # generates previews asynchronously after upload); it'll be retried.
        if missing:
            oldest = min(parse_ts(a.created_at) for a in missing)
            new_wm = (datetime.fromtimestamp(oldest, UTC) - timedelta(milliseconds=1)).isoformat()
        else:
            new_wm = max_created
        if new_wm and not self.cfg.dry_run:
            self.store.set_meta(wm_key, new_wm)

        summary.seconds = round(time.monotonic() - started, 1)
        log.info("run done", extra={**summary.__dict__, "labels": dict(summary.labels)})
        return summary

    # ------------------------------------------------------------------------
    def _work(self, batch: list[Asset], source: PreviewSource) -> list[_Result]:
        results, jpegs, idx = [], [], []
        for a in batch:
            jpeg = source.get(a)
            if jpeg is None:  # preview not generated yet: retry next run
                results.append(_Result(a, jpeg_ok=False))
                continue
            try:
                s = sharpness(jpeg)
                small = downscale_for_clip(jpeg)
            except Exception as e:  # corrupt preview: record, don't kill the run
                log.warning("unreadable preview", extra={"asset": a.id, "error": str(e)})
                results.append(_Result(a, jpeg_ok=True, error=f"unreadable preview: {e}"))
                continue
            results.append(_Result(a, jpeg_ok=True, sharpness=s))
            jpegs.append(small)
            idx.append(len(results) - 1)
        if jpegs:
            self._embed_into(results, jpegs, idx)
        return results

    def _embed_into(self, results: list[_Result], jpegs: list[bytes], idx: list[int]) -> None:
        """Embed a batch; if the server rejects it, isolate the bad image(s).

        Transport errors (server down) propagate and fail the run, so an
        outage never turns into thousands of permanent per-asset errors.
        """
        try:
            emb = self.inference.embed_images(jpegs)
            for k, i in enumerate(idx):
                results[i].embedding = emb[k]
            return
        except (httpx.HTTPStatusError, EmbeddingCountMismatch) as e:
            if len(jpegs) == 1:
                a = results[idx[0]].asset
                log.warning("embedding failed", extra={"asset": a.id, "error": str(e)[:300]})
                results[idx[0]].error = f"embedding failed: {str(e)[:300]}"
                return
        for j, i in zip(jpegs, idx, strict=True):
            self._embed_into(results, [j], [i])

    def _analyze(
        self, todo: list[Asset], source: PreviewSource, owner: str, summary: Summary
    ) -> list[Asset]:
        content, blur = self._banks()
        t = self.cfg.thresholds
        bs, conc = self.cfg.batch_size, self.cfg.concurrency
        batches = [todo[i : i + bs] for i in range(0, len(todo), bs)]
        missing: list[Asset] = []
        t0, last_log = time.monotonic(), time.monotonic()

        with ThreadPoolExecutor(max_workers=conc) as pool:
            inflight: list[Future] = []
            it = iter(batches)
            # Bounded window: at most 2x concurrency batches in memory at once.
            for b in it:
                inflight.append(pool.submit(self._work, b, source))
                if len(inflight) < conc * 2:
                    continue
                self._persist(inflight.pop(0).result(), owner, content, blur, t, summary, missing)
                if time.monotonic() - last_log > 30:
                    last_log = time.monotonic()
                    rate = summary.analyzed / (last_log - t0)
                    log.info(
                        "progress",
                        extra={
                            "analyzed": summary.analyzed,
                            "total": len(todo),
                            "per_second": round(rate, 1),
                            "eta_min": round(
                                (len(todo) - summary.analyzed) / max(rate, 1e-6) / 60, 1
                            ),
                        },
                    )
            for f in inflight:
                self._persist(f.result(), owner, content, blur, t, summary, missing)
        self.store.commit()
        return missing

    def _persist(self, results, owner, content, blur, t, summary, missing) -> None:
        for r in results:
            if r.error:
                self.store.record_error(r.asset.id, owner, r.error)
                summary.errors += 1
                continue
            if not r.jpeg_ok or r.embedding is None:
                summary.missing_preview += 1
                missing.append(r.asset)
                continue
            a = r.asset
            v = classify(a, r.embedding, r.sharpness, content, blur, t)
            self.store.insert(
                asset_id=a.id,
                owner_id=owner,
                created_at=a.created_at,
                taken_ts=a.taken_ts,
                is_favorite=a.is_favorite,
                pixels=(a.width or 0) * (a.height or 0),
                sharpness=r.sharpness,
                embedding=r.embedding,
                label=v.label.value,
                reason=v.reason,
                scores=v.scores,
                album=self._album_for(v.label),
            )
            summary.analyzed += 1
            summary.labels[v.label.value] += 1
        self.store.commit()

    def _dedupe(self, owner: str, ts_min: float, ts_max: float, summary: Summary) -> None:
        t = self.cfg.thresholds
        w = t.duplicate_window_s
        rows = self.store.dedupe_candidates(owner, ts_min - w, ts_max + w)
        groups = find_groups(rows, t.duplicate_cosine, w)
        for g in groups:
            for row, cos in g.extras:
                if row.label == Label.KEEP and not row.is_favorite:
                    self.store.mark_duplicate(
                        row.id, g.keeper.id, cos, self.cfg.albums.safe_to_delete
                    )
                    summary.labels[Label.DUPLICATE.value] += 1
                    summary.labels[Label.KEEP.value] -= 1
        self.store.commit()
        log.info("dedupe", extra={"candidates": len(rows), "groups": len(groups)})

    def _sync_albums(self, immich: ImmichClient, owner: str) -> int:
        names = {
            self.cfg.albums.receipts: ALBUM_DESCRIPTIONS["receipts"],
            self.cfg.albums.screenshots: ALBUM_DESCRIPTIONS["screenshots"],
            self.cfg.albums.safe_to_delete: ALBUM_DESCRIPTIONS["safe_to_delete"],
        }
        total = 0
        for album, ids in self.store.unsynced(owner).items():
            album_id = immich.ensure_album(album, names.get(album, ""))
            ok = immich.add_to_album(album_id, ids)
            self.store.mark_synced(ok)
            total += len(ok)
            log.info("album sync", extra={"album": album, "added": len(ok), "wanted": len(ids)})
        return total
