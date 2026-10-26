"""End-to-end: real clients, fake Immich + fake Inference server over httpx MockTransport."""

from __future__ import annotations

import base64
import json
import uuid

import httpx
import numpy as np
import pytest

from immich_analyzer.config import Config
from immich_analyzer.images import downscale_for_clip
from immich_analyzer.immich import ImmichClient
from immich_analyzer.inference import InferenceClient
from immich_analyzer.pipeline import Analyzer, Mode
from immich_analyzer.store import Store

from .conftest import make_jpeg

E = np.eye(5, dtype=np.float32)  # receipt, screenshot, photo, document, blur


def text_vec(t: str) -> list[float]:
    t = t.lower()
    if "receipt" in t or "bill" in t:
        return E[0].tolist()
    if "screenshot" in t:
        return E[1].tolist()
    if "blurry" in t or "blurred" in t:
        return E[4].tolist()
    if any(w in t for w in ("document", "page", "whiteboard")):
        return E[3].tolist()
    return E[2].tolist()  # people, pets, "sharp in focus", ...


class FakeImmich:
    def __init__(self):
        self.assets: list[dict] = []
        self.previews: dict[str, bytes] = {}
        self.albums: dict[str, dict] = {}
        self.search_bodies: list[dict] = []

    def add(self, created: str, taken: str, jpeg: bytes | None, **kw) -> str:
        aid = str(uuid.uuid4())
        self.assets.append(
            {
                "id": aid,
                "ownerId": "owner-1",
                "createdAt": created,
                "localDateTime": taken,
                "originalFileName": kw.get("name", "IMG_0001.JPG"),
                "originalMimeType": kw.get("mime", "image/jpeg"),
                "checksum": aid,
                "isFavorite": kw.get("fav", False),
                "width": 4000,
                "height": 3000,
                "exifInfo": {"make": "Apple", "model": "iPhone"},
            }
        )
        if jpeg is not None:
            self.previews[aid] = jpeg
        return aid

    def handler(self, req: httpx.Request) -> httpx.Response:
        assert req.headers["x-api-key"] == "k1"
        p = req.url.path
        if p == "/api/users/me":
            return httpx.Response(200, json={"id": "owner-1", "email": "me@example.com"})
        if p == "/api/search/metadata":
            body = json.loads(req.content)
            self.search_bodies.append(body)
            gt = body["filter"].get("createdAt", {}).get("gt")
            items = [a for a in self.assets if not gt or a["createdAt"] > gt]
            return httpx.Response(200, json={"assets": {"items": items, "nextCursor": None}})
        if p.endswith("/thumbnail"):
            aid = p.split("/")[3]
            if aid not in self.previews:
                return httpx.Response(404)
            return httpx.Response(200, content=self.previews[aid])
        if p == "/api/albums" and req.method == "GET":
            return httpx.Response(
                200, json=[{"id": a["id"], "albumName": n} for n, a in self.albums.items()]
            )
        if p == "/api/albums" and req.method == "POST":
            name = json.loads(req.content)["albumName"]
            self.albums[name] = {"id": f"album-{len(self.albums)}", "assets": set()}
            return httpx.Response(201, json={"id": self.albums[name]["id"], "albumName": name})
        if p.startswith("/api/albums/") and req.method == "PUT":
            album = next(a for a in self.albums.values() if a["id"] == p.split("/")[3])
            out = []
            for i in json.loads(req.content)["ids"]:
                dup = i in album["assets"]
                album["assets"].add(i)
                out.append({"id": i, "success": not dup, **({"error": "duplicate"} if dup else {})})
            return httpx.Response(200, json=out)
        return httpx.Response(404, json={"path": p})

    def album(self, name: str) -> set[str]:
        return self.albums.get(name, {}).get("assets", set())


class FakeInference:
    """Fake Roboflow server. ``mode`` reproduces inference 1.8.x list handling:
    "correct" | "collapse" (same-size list -> 1 embedding) | "upstream" (mixed
    sizes 500, same sizes collapse)."""

    def __init__(self, mode: str = "correct"):
        self.by_jpeg: dict[bytes, list[float]] = {}
        self.poison: set[bytes] = set()
        self.image_calls = 0
        self.mode = mode

    def handler(self, req: httpx.Request) -> httpx.Response:
        if req.url.path == "/readiness":
            return httpx.Response(200)
        body = json.loads(req.content)
        if req.url.path == "/clip/embed_text":
            return httpx.Response(200, json={"embeddings": [text_vec(t) for t in body["text"]]})
        if req.url.path == "/clip/embed_image":
            self.image_calls += 1
            imgs = body["image"] if isinstance(body["image"], list) else [body["image"]]
            raw = [base64.b64decode(i["value"]) for i in imgs]
            if any(r in self.poison for r in raw):
                return httpx.Response(500, json={"message": "boom"})
            if isinstance(body["image"], list) and len(raw) > 1 and self.mode != "correct":
                if self.mode == "upstream":
                    return httpx.Response(500, json={"message": "ValueError: dimensions"})
                return httpx.Response(200, json={"embeddings": [self.by_jpeg[raw[0]]]})
            return httpx.Response(200, json={"embeddings": [self.by_jpeg[r] for r in raw]})
        return httpx.Response(404)


@pytest.fixture
def env(tmp_path):
    immich, inf = FakeImmich(), FakeInference()
    cfg = Config(
        immich_url="http://immich",
        immich_api_keys=("k1",),
        inference_url="http://inference",
        state_path=tmp_path / "s.sqlite3",
        batch_size=2,
        concurrency=2,
    )
    store = Store(cfg.state_path)
    ic = InferenceClient(
        cfg.inference_url,
        client=httpx.Client(transport=httpx.MockTransport(inf.handler)),
        retries=0,
    )
    analyzer = Analyzer(cfg, store, ic)

    def client():
        return ImmichClient(
            cfg.immich_url, "k1", client=httpx.Client(transport=httpx.MockTransport(immich.handler))
        )

    yield immich, inf, analyzer, client, store
    store.close()


def photo(inf: FakeInference, emb, blur=0.0, seed=0) -> bytes:
    j = make_jpeg(blur=blur, seed=seed)
    inf.by_jpeg[downscale_for_clip(j)] = np.asarray(emb, dtype=float).tolist()
    return j


def test_backfill_then_incremental(env):
    immich, inf, analyzer, client, store = env
    t0 = "2026-01-01T00:00:0{}.000Z"
    receipt = immich.add("2026-02-01T00:00:00.000Z", t0.format(0), photo(inf, E[0], seed=1))
    shot = immich.add(
        "2026-02-01T00:00:01.000Z",
        t0.format(0),
        photo(inf, E[1], seed=2),
        name="Screenshot_2026.png",
        mime="image/png",
    )
    best = immich.add("2026-02-01T00:00:02.000Z", t0.format(1), photo(inf, E[2], seed=3))
    dup = immich.add("2026-02-01T00:00:03.000Z", t0.format(3), photo(inf, E[2], blur=1.2, seed=3))
    blurry = immich.add(
        "2026-02-01T00:00:04.000Z", "2026-03-01T00:00:00.000Z", photo(inf, E[2], blur=8, seed=4)
    )
    fav_blurry = immich.add(
        "2026-02-01T00:00:05.000Z",
        "2026-04-01T00:00:00.000Z",
        photo(inf, E[2], blur=8, seed=5),
        fav=True,
    )
    late = immich.add("2026-02-01T00:00:06.000Z", "2026-05-01T00:00:00.000Z", None)

    s = analyzer.run(client(), Mode.AUTO)
    assert s.mode == "backfill" and s.analyzed == 6 and s.missing_preview == 1
    assert immich.album("Receipts") == {receipt}
    assert immich.album("Screenshots") == {shot}
    assert immich.album("Safe to Delete") == {dup, blurry}
    assert best not in immich.album("Safe to Delete")
    assert fav_blurry not in immich.album("Safe to Delete")
    # Watermark held just before the asset without a preview.
    assert store.get_meta("watermark:owner-1") < "2026-02-01T00:00:06"

    # Preview appears; a new upload arrives. Second run is incremental.
    immich.previews[late] = photo(inf, E[0], seed=6)
    new = immich.add(
        "2026-02-02T00:00:00.000Z",
        "2026-06-01T00:00:00.000Z",
        photo(inf, E[1], seed=7),
        name="Screen Shot 2026.png",
    )
    s2 = analyzer.run(client(), Mode.AUTO)
    assert s2.mode == "incremental" and s2.analyzed == 2
    assert "createdAt" in immich.search_bodies[-1]["filter"]
    assert immich.album("Receipts") == {receipt, late}
    assert new in immich.album("Screenshots")

    # Nothing new: no GPU calls, nothing re-added.
    calls = inf.image_calls
    s3 = analyzer.run(client(), Mode.AUTO)
    assert s3.analyzed == 0 and s3.synced == 0 and inf.image_calls == calls


def test_user_removal_from_album_is_respected(env):
    immich, inf, analyzer, client, _ = env
    r = immich.add("2026-02-01T00:00:00.000Z", "2026-01-01T00:00:00.000Z", photo(inf, E[0], seed=1))
    analyzer.run(client(), Mode.AUTO)
    immich.albums["Receipts"]["assets"].discard(r)  # human says "not a receipt"
    analyzer.run(client(), Mode.BACKFILL)
    assert r not in immich.album("Receipts")


def test_dry_run_touches_nothing(env, tmp_path):
    immich, inf, analyzer, client, store = env
    analyzer.cfg = Config(**{**analyzer.cfg.__dict__, "dry_run": True})
    immich.add("2026-02-01T00:00:00.000Z", "2026-01-01T00:00:00.000Z", photo(inf, E[0], seed=1))
    s = analyzer.run(client(), Mode.AUTO)
    assert s.analyzed == 1 and immich.albums == {}
    assert store.get_meta("watermark:owner-1") is None


@pytest.mark.parametrize("mode", ["collapse", "upstream"])
def test_survives_upstream_batching_bug(env, mode):
    """Batches the server mishandles fall back to one image per request."""
    immich, inf, analyzer, client, _ = env
    inf.mode = mode
    r = immich.add("2026-02-01T00:00:00.000Z", "2026-01-01T00:00:00.000Z", photo(inf, E[0], seed=1))
    s = immich.add(
        "2026-02-01T00:00:01.000Z",
        "2026-02-01T00:00:00.000Z",
        photo(inf, E[1], seed=2),
        name="Screenshot.png",
        mime="image/png",
    )
    summary = analyzer.run(client(), Mode.AUTO)  # batch_size=2 in this fixture
    assert summary.analyzed == 2 and summary.errors == 0
    assert immich.album("Receipts") == {r} and immich.album("Screenshots") == {s}


def test_one_bad_image_is_recorded_not_fatal(env):
    immich, inf, analyzer, client, store = env
    good = immich.add(
        "2026-02-01T00:00:00.000Z", "2026-01-01T00:00:00.000Z", photo(inf, E[0], seed=1)
    )
    bad_jpeg = photo(inf, E[0], seed=2)
    inf.poison.add(downscale_for_clip(bad_jpeg))
    bad = immich.add("2026-02-01T00:00:01.000Z", "2026-01-02T00:00:00.000Z", bad_jpeg)

    s1 = analyzer.run(client(), Mode.AUTO)
    assert s1.analyzed == 1 and s1.errors == 1
    assert immich.album("Receipts") == {good}
    assert store.error_count() == 1
    # Permanent errors don't hold the watermark or get retried every run.
    assert store.get_meta("watermark:owner-1") >= "2026-02-01T00:00:01"
    calls = inf.image_calls
    s2 = analyzer.run(client(), Mode.AUTO)
    assert s2.analyzed == 0 and s2.errors == 0 and inf.image_calls == calls
    assert bad not in immich.album("Receipts")


def test_inference_outage_fails_the_run_instead_of_recording_errors(env):
    immich, inf, analyzer, client, store = env
    immich.add("2026-02-01T00:00:00.000Z", "2026-01-01T00:00:00.000Z", photo(inf, E[0], seed=1))
    analyzer._banks()  # text prompts embedded while the server was still up

    def down(req):
        raise httpx.ConnectError("connection refused", request=req)

    analyzer.inference._http = httpx.Client(transport=httpx.MockTransport(down))
    analyzer.inference._retries = 0
    with pytest.raises(httpx.ConnectError):
        analyzer.run(client(), Mode.AUTO)
    assert store.error_count() == 0
