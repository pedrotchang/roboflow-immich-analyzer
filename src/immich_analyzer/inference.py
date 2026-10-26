"""Client for a self-hosted Roboflow Inference server's CLIP routes.

The server holds the Roboflow API key (env ROBOFLOW_API_KEY) and falls back to
it when a request carries none, so this client never sees the key.
"""

from __future__ import annotations

import base64
import time

import httpx
import numpy as np


class EmbeddingCountMismatch(RuntimeError):
    """Server returned a different number of embeddings than images sent."""


class InferenceClient:
    def __init__(
        self,
        base_url: str,
        clip_version: str = "ViT-B-16",
        *,
        client: httpx.Client | None = None,
        retries: int = 3,
    ):
        self._http = client or httpx.Client(timeout=httpx.Timeout(120.0, connect=5.0))
        self._base = base_url
        self._version = clip_version
        self._retries = retries

    def close(self) -> None:
        self._http.close()

    def _post(self, path: str, body: dict) -> dict:
        for attempt in range(self._retries + 1):
            try:
                resp = self._http.post(f"{self._base}{path}", json=body)
                if resp.status_code < 500:
                    resp.raise_for_status()
                    return resp.json()
                resp.raise_for_status()
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                is_client_error = (
                    isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500
                )
                if is_client_error or attempt == self._retries:
                    raise
                time.sleep(2**attempt)  # model cold-load / GPU contention
        raise AssertionError("unreachable")

    def ready(self) -> bool:
        try:
            return self._http.get(f"{self._base}/readiness", timeout=5.0).status_code == 200
        except httpx.TransportError:
            return False

    def embed_images(self, jpegs: list[bytes]) -> np.ndarray:
        """L2-normalised CLIP embeddings, shape (n, d).

        A single image is sent as an object, not a one-element list: in
        inference 1.8.x the ``inference-models`` CLIP backend concatenates
        list inputs along the height axis before preprocessing, so mixed sizes
        500 and same-width images silently collapse into one embedding.
        The count check below fails closed if that ever happens.
        """
        images = [{"type": "base64", "value": base64.b64encode(j).decode("ascii")} for j in jpegs]
        body = {
            "clip_version_id": self._version,
            "image": images[0] if len(images) == 1 else images,
        }
        emb = np.asarray(self._post("/clip/embed_image", body)["embeddings"])
        if emb.ndim != 2 or emb.shape[0] != len(jpegs):
            raise EmbeddingCountMismatch(
                f"sent {len(jpegs)} images, got {emb.shape[0] if emb.ndim == 2 else emb.shape}"
            )
        return _normalise(emb)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        body = {"clip_version_id": self._version, "text": texts}
        return _normalise(np.asarray(self._post("/clip/embed_text", body)["embeddings"]))


def _normalise(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
