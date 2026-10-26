from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from immich_analyzer.immich import Asset


def make_jpeg(blur: float = 0.0, seed: int = 0) -> bytes:
    """A textured test image; ``blur`` applies a Gaussian blur radius."""
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (800, 600), "white")
    d = ImageDraw.Draw(im)
    for _ in range(200):
        x, y = rng.integers(0, 800), rng.integers(0, 600)
        d.rectangle([x, y, x + 20, y + 20], fill=tuple(int(c) for c in rng.integers(0, 255, 3)))
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def make_asset(i: int = 0, **kw) -> Asset:
    base = dict(
        id=f"{i:08x}-0000-4000-8000-000000000000",
        owner_id="owner-1",
        created_at="2026-01-01T00:00:00.000Z",
        taken_at="2026-01-01T00:00:00.000Z",
        original_file_name=f"IMG_{i:04d}.JPG",
        mime_type="image/jpeg",
        checksum=f"sum{i}",
        is_favorite=False,
        width=4000,
        height=3000,
        camera_make="Apple",
        camera_model="iPhone 15",
    )
    base.update(kw)
    return Asset(**base)


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def sharp_jpeg() -> bytes:
    return make_jpeg()


@pytest.fixture
def blurry_jpeg() -> bytes:
    return make_jpeg(blur=8)
