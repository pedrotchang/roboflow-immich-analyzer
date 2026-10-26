"""Zero-shot classification on top of CLIP embeddings.

Text prompts are embedded once per run; each image costs one GPU embedding
call that is reused for classification *and* duplicate detection. Scores are
a softmax over per-class mean cosine similarity (CLIP's logit scale is 100).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .config import Thresholds
from .immich import Asset

CLIP_LOGIT_SCALE = 100.0

# Prompt ensembles. "other" classes absorb probability mass so a normal photo
# never has to pick between receipt and screenshot.
PROMPTS: dict[str, list[str]] = {
    "receipt": [
        "a photo of a paper receipt",
        "a photo of a printed shopping receipt with prices",
        "a close-up photo of a store receipt",
        "a photo of a restaurant bill",
    ],
    "screenshot": [
        "a screenshot of a phone screen",
        "a screenshot of a mobile app",
        "a screenshot of a computer screen",
        "a screenshot of a text message conversation",
    ],
    "photo": [
        "a photo of people",
        "a photo of a family",
        "a photo of a landscape",
        "a photo of a pet",
        "a photo of food",
        "a photo of a building",
        "a photo of an object",
    ],
    "document": [
        "a photo of a document",
        "a photo of a page of a book",
        "a photo of a whiteboard",
    ],
}
BLUR_PROMPTS: dict[str, list[str]] = {
    "blurry": [
        "a blurry out of focus photo",
        "a motion blurred photo",
        "a shaky blurry picture",
    ],
    "sharp": [
        "a sharp in focus photo",
        "a clear high quality photo",
    ],
}

SCREENSHOT_NAME = re.compile(r"screen[\s_-]?shot|screen[\s_-]?recording|^scr_|simulator", re.I)


class Label(StrEnum):
    RECEIPT = "receipt"
    SCREENSHOT = "screenshot"
    BLURRY = "blurry"
    DUPLICATE = "duplicate"
    KEEP = "keep"


@dataclass(frozen=True)
class Verdict:
    label: Label
    reason: str
    scores: dict[str, float]


class PromptBank:
    """Class-mean text embeddings for one prompt set."""

    def __init__(self, classes: list[str], embeddings: np.ndarray):
        self.classes = classes
        self.embeddings = embeddings  # (k, d), each row L2-normalised

    @classmethod
    def build(cls, prompts: dict[str, list[str]], embed) -> PromptBank:
        names = list(prompts)
        flat = [p for n in names for p in prompts[n]]
        vecs = embed(flat)
        means, i = [], 0
        for n in names:
            m = vecs[i : i + len(prompts[n])].mean(axis=0)
            means.append(m / np.linalg.norm(m))
            i += len(prompts[n])
        return cls(names, np.stack(means))

    def probs(self, image_emb: np.ndarray) -> dict[str, float]:
        logits = CLIP_LOGIT_SCALE * (self.embeddings @ image_emb)
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        return {c: round(float(v), 4) for c, v in zip(self.classes, p, strict=True)}


def looks_like_screenshot(asset: Asset) -> bool:
    """Metadata prior: screenshot filenames, or a PNG with no camera EXIF."""
    if SCREENSHOT_NAME.search(asset.original_file_name):
        return True
    return asset.mime_type == "image/png" and not asset.camera_make and not asset.camera_model


def classify(
    asset: Asset,
    image_emb: np.ndarray,
    sharp: float,
    content: PromptBank,
    blur: PromptBank,
    t: Thresholds,
) -> Verdict:
    """Single-asset verdict. Duplicates are decided later, across assets."""
    p = content.probs(image_emb)
    b = blur.probs(image_emb)
    scores = {**p, "blurry": b["blurry"], "sharpness": round(sharp, 2)}

    if p["receipt"] >= t.receipt_prob:
        return Verdict(Label.RECEIPT, f"clip receipt={p['receipt']:.2f}", scores)

    prior = looks_like_screenshot(asset)
    if p["screenshot"] >= t.screenshot_prob or (prior and p["screenshot"] >= 0.25):
        why = "filename/exif + clip" if prior else "clip"
        return Verdict(Label.SCREENSHOT, f"{why} screenshot={p['screenshot']:.2f}", scores)

    # Documents and screenshots are naturally low-texture; only judge blur on
    # camera photos. Favorites are never proposed for deletion.
    if not asset.is_favorite and p["document"] < 0.5:
        if sharp < t.blur_hard:
            return Verdict(Label.BLURRY, f"sharpness={sharp:.1f}<{t.blur_hard}", scores)
        if sharp < t.blur_soft and b["blurry"] >= t.blur_clip_prob:
            return Verdict(
                Label.BLURRY, f"sharpness={sharp:.1f} + clip blurry={b['blurry']:.2f}", scores
            )

    return Verdict(Label.KEEP, "", scores)
