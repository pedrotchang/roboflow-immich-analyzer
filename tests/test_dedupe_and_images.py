import numpy as np
import pytest

from immich_analyzer.dedupe import find_groups
from immich_analyzer.images import NfsPreviewSource, sharpness
from immich_analyzer.store import Row

from .conftest import make_asset, unit


def row(i, ts, emb, sharp=100.0, fav=False, pixels=12_000_000):
    return Row(f"a{i}", ts, fav, pixels, sharp, unit(emb), "keep")


def test_burst_shots_group_and_keep_the_sharpest():
    rows = [
        row(0, 0, [1, 0, 0], sharp=50),
        row(1, 2, [1, 0.01, 0], sharp=300),
        row(2, 4, [1, 0, 0.01], sharp=80),
    ]
    (g,) = find_groups(rows, cosine=0.95, window_s=120)
    assert g.keeper.id == "a1"
    assert {r.id for r, _ in g.extras} == {"a0", "a2"}


def test_favorite_beats_sharpness():
    rows = [row(0, 0, [1, 0], sharp=10, fav=True), row(1, 1, [1, 0], sharp=999)]
    (g,) = find_groups(rows, 0.95, 120)
    assert g.keeper.id == "a0"


def test_similar_but_far_apart_in_time_are_not_duplicates():
    rows = [row(0, 0, [1, 0]), row(1, 10_000, [1, 0])]
    assert find_groups(rows, 0.95, 120) == []


def test_different_content_close_in_time_is_not_duplicate():
    rows = [row(0, 0, [1, 0]), row(1, 1, [0, 1])]
    assert find_groups(rows, 0.95, 120) == []


def test_transitive_chain_forms_one_group():
    rows = [row(0, 0, [1, 0]), row(1, 100, [1, 0.2]), row(2, 200, [1, 0.4])]
    groups = find_groups(rows, 0.97, 120)
    assert len(groups) == 1 and len(groups[0].extras) == 2


def test_sharpness_separates_sharp_from_blurred(sharp_jpeg, blurry_jpeg):
    assert sharpness(sharp_jpeg) > 10 * sharpness(blurry_jpeg)


def test_nfs_source_uses_immich_thumbnail_layout(tmp_path, sharp_jpeg):
    a = make_asset(1)
    src = NfsPreviewSource(tmp_path)
    p = src.path_for(a)
    assert p == tmp_path / "thumbs" / "owner-1" / a.id[:2] / a.id[2:4] / f"{a.id}-preview.jpeg"
    assert src.get(a) is None
    p.parent.mkdir(parents=True)
    p.write_bytes(sharp_jpeg)
    assert src.get(a) == sharp_jpeg


def test_nfs_source_falls_back(tmp_path):
    class Fallback:
        name = "api"

        def get(self, asset):
            return b"from-api"

    assert NfsPreviewSource(tmp_path, Fallback()).get(make_asset()) == b"from-api"


def test_embeddings_roundtrip_float32():
    v = unit(np.arange(1, 513))
    assert np.frombuffer(v.astype(np.float32).tobytes(), dtype=np.float32).shape == (512,)


def test_downscale_for_clip_keeps_aspect_and_never_upscales(sharp_jpeg):
    import io

    from PIL import Image

    from immich_analyzer.images import downscale_for_clip

    small = Image.open(io.BytesIO(downscale_for_clip(sharp_jpeg)))
    assert min(small.size) == 336 and small.size[0] / small.size[1] == pytest.approx(
        800 / 600, 0.01
    )
    tiny = downscale_for_clip(downscale_for_clip(sharp_jpeg, 100))
    assert min(Image.open(io.BytesIO(tiny)).size) == 100
