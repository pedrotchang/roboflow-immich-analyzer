"""Runtime configuration, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


@dataclass(frozen=True)
class Thresholds:
    """Decision thresholds. Conservative by default: a false positive costs a
    human a few seconds in an album; a false negative costs nothing."""

    receipt_prob: float = 0.60
    screenshot_prob: float = 0.60
    # Variance of the Laplacian on a 512px grayscale preview.
    blur_hard: float = 15.0  # below this: blurry regardless of CLIP
    blur_soft: float = 60.0  # below this: blurry if CLIP agrees
    blur_clip_prob: float = 0.50
    # Cosine similarity between CLIP embeddings for a near-duplicate.
    duplicate_cosine: float = 0.95
    # Only compare photos taken within this many seconds of each other.
    duplicate_window_s: int = 120


@dataclass(frozen=True)
class Albums:
    receipts: str = "Receipts"
    screenshots: str = "Screenshots"
    safe_to_delete: str = "Safe to Delete"


@dataclass(frozen=True)
class Config:
    immich_url: str
    immich_api_keys: tuple[str, ...]
    inference_url: str
    clip_version: str = "ViT-B-16"
    state_path: Path = Path("/state/analyzer.sqlite3")
    library_root: Path | None = None  # NFS mount of Immich's upload dir
    # 1 until roboflow/inference batches CLIP images correctly (see
    # InferenceClient.embed_images); throughput comes from concurrency instead.
    batch_size: int = 1
    concurrency: int = 8
    dry_run: bool = False
    thresholds: Thresholds = field(default_factory=Thresholds)
    albums: Albums = field(default_factory=Albums)

    @classmethod
    def from_env(cls) -> Config:
        keys = tuple(
            k.strip()
            for k in os.environ.get("IMMICH_API_KEYS", "").replace("\n", ",").split(",")
            if k.strip()
        )
        if not keys:
            raise SystemExit("IMMICH_API_KEYS is empty: set at least one Immich API key")
        lib = os.environ.get("LIBRARY_ROOT")
        return cls(
            immich_url=os.environ.get("IMMICH_URL", "http://immich-server.immich:2283").rstrip("/"),
            immich_api_keys=keys,
            inference_url=os.environ.get("INFERENCE_URL", "http://roboflow-inference:9001").rstrip(
                "/"
            ),
            clip_version=os.environ.get("CLIP_VERSION_ID", "ViT-B-16"),
            state_path=Path(os.environ.get("STATE_PATH", "/state/analyzer.sqlite3")),
            library_root=Path(lib) if lib else None,
            batch_size=_int("BATCH_SIZE", 1),
            concurrency=_int("CONCURRENCY", 8),
            dry_run=_bool("DRY_RUN", False),
            thresholds=Thresholds(
                receipt_prob=_float("RECEIPT_PROB", Thresholds.receipt_prob),
                screenshot_prob=_float("SCREENSHOT_PROB", Thresholds.screenshot_prob),
                blur_hard=_float("BLUR_HARD", Thresholds.blur_hard),
                blur_soft=_float("BLUR_SOFT", Thresholds.blur_soft),
                blur_clip_prob=_float("BLUR_CLIP_PROB", Thresholds.blur_clip_prob),
                duplicate_cosine=_float("DUPLICATE_COSINE", Thresholds.duplicate_cosine),
                duplicate_window_s=_int("DUPLICATE_WINDOW_S", Thresholds.duplicate_window_s),
            ),
            albums=Albums(
                receipts=os.environ.get("ALBUM_RECEIPTS", Albums.receipts),
                screenshots=os.environ.get("ALBUM_SCREENSHOTS", Albums.screenshots),
                safe_to_delete=os.environ.get("ALBUM_SAFE_TO_DELETE", Albums.safe_to_delete),
            ),
        )
