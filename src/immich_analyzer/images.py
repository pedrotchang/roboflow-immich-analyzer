"""Where preview pixels come from, and the one pixel-level metric we compute.

Two sources, same output (Immich's ~1440px preview JPEG):
- ``NfsPreviewSource`` reads the file straight off a read-only mount of
  Immich's upload directory. Used for backfills: no HTTP, no load on the
  Immich server, bounded only by NFS throughput.
- ``ApiPreviewSource`` fetches it over the Immich API. Used for incremental
  runs where only a handful of new assets need pixels.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .immich import Asset, ImmichClient

SHARPNESS_SIDE = 512


class PreviewSource(Protocol):
    name: str

    def get(self, asset: Asset) -> bytes | None: ...


class NfsPreviewSource:
    """Immich layout: thumbs/<ownerId>/<id[0:2]>/<id[2:4]>/<id>-preview.jpeg"""

    name = "nfs"

    def __init__(self, root: Path, fallback: PreviewSource | None = None):
        self._root = root
        self._fallback = fallback

    def path_for(self, asset: Asset) -> Path:
        a = asset.id
        return self._root / "thumbs" / asset.owner_id / a[0:2] / a[2:4] / f"{a}-preview.jpeg"

    def get(self, asset: Asset) -> bytes | None:
        try:
            return self.path_for(asset).read_bytes()
        except FileNotFoundError:
            return self._fallback.get(asset) if self._fallback else None


class ApiPreviewSource:
    name = "api"

    def __init__(self, immich: ImmichClient):
        self._immich = immich

    def get(self, asset: Asset) -> bytes | None:
        return self._immich.preview(asset.id)


def sharpness(jpeg: bytes) -> float:
    """Variance of the Laplacian on a fixed-size grayscale image.

    Classic focus measure (Pech-Pacheco et al., 2000): blurry images have few
    strong second-derivative responses. Resizing to a fixed side first makes
    the score comparable across resolutions.
    """
    with Image.open(io.BytesIO(jpeg)) as im:
        im = im.convert("L")
        im.thumbnail((SHARPNESS_SIDE, SHARPNESS_SIDE))
        g = np.asarray(im, dtype=np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(lap.var())
