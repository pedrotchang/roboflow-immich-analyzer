import numpy as np

from immich_analyzer.classify import Label, PromptBank, classify, looks_like_screenshot
from immich_analyzer.config import Thresholds

from .conftest import make_asset, unit

# 4-d toy embedding space: axis per content class.
CONTENT = PromptBank(
    ["receipt", "screenshot", "photo", "document"],
    np.eye(4, dtype=np.float32),
)
BLUR = PromptBank(["blurry", "sharp"], np.array([[0, 0, 1, 1], [0, 0, 1, -1]], np.float32) / 2**0.5)
T = Thresholds()


def test_receipt_wins_when_clip_is_confident():
    v = classify(make_asset(), unit([1, 0, 0.1, 0]), 200.0, CONTENT, BLUR, T)
    assert v.label is Label.RECEIPT
    assert v.scores["receipt"] > 0.6


def test_screenshot_from_clip_alone():
    v = classify(make_asset(), unit([0, 1, 0.1, 0]), 200.0, CONTENT, BLUR, T)
    assert v.label is Label.SCREENSHOT


def test_screenshot_metadata_prior_lowers_the_bar():
    # CLIP is torn between screenshot and photo; filename tips it.
    emb = unit([0, 1, 1, 0])
    plain = classify(make_asset(), emb, 200.0, CONTENT, BLUR, T)
    named = classify(
        make_asset(original_file_name="Screenshot 2026-01-01 at 10.00.00.png"),
        emb,
        200.0,
        CONTENT,
        BLUR,
        T,
    )
    assert plain.label is Label.KEEP
    assert named.label is Label.SCREENSHOT


def test_very_low_sharpness_is_blurry_regardless_of_clip():
    v = classify(make_asset(), unit([0, 0, 1, 0]), 5.0, CONTENT, BLUR, T)
    assert v.label is Label.BLURRY


def test_soft_blur_needs_clip_agreement():
    sharp_looking = unit([0, 0, 1, -0.3])
    blurry_looking = unit([0, 0, 1, 0.3])
    assert classify(make_asset(), sharp_looking, 40.0, CONTENT, BLUR, T).label is Label.KEEP
    assert classify(make_asset(), blurry_looking, 40.0, CONTENT, BLUR, T).label is Label.BLURRY


def test_favorites_are_never_blurry():
    v = classify(make_asset(is_favorite=True), unit([0, 0, 1, 0]), 1.0, CONTENT, BLUR, T)
    assert v.label is Label.KEEP


def test_documents_are_not_judged_for_blur():
    v = classify(make_asset(), unit([0, 0, 0.2, 1]), 5.0, CONTENT, BLUR, T)
    assert v.label is Label.KEEP


def test_looks_like_screenshot_heuristics():
    assert looks_like_screenshot(make_asset(original_file_name="Screen Shot 2020.png"))
    assert looks_like_screenshot(make_asset(original_file_name="SCR_001.jpg"))
    assert looks_like_screenshot(
        make_asset(mime_type="image/png", camera_make=None, camera_model=None)
    )
    assert not looks_like_screenshot(make_asset(mime_type="image/png"))  # has camera EXIF
    assert not looks_like_screenshot(make_asset())


def test_prompt_bank_build_averages_and_normalises():
    def fake_embed(texts):
        return np.array([unit([1, 0]), unit([1, 1]), unit([0, 1])])

    bank = PromptBank.build({"a": ["x", "y"], "b": ["z"]}, fake_embed)
    assert bank.classes == ["a", "b"]
    np.testing.assert_allclose(np.linalg.norm(bank.embeddings, axis=1), 1.0, rtol=1e-6)
