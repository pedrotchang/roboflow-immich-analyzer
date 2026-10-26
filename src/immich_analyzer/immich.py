"""Minimal Immich API client (targets Immich v3.2+).

Only the endpoints the analyzer needs. Every call authenticates with an API
key; an Immich API key is scoped to one user, so one client == one library.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime

import httpx

log = logging.getLogger(__name__)

# Immich caps PUT /albums/{id}/assets payloads well above this; 500 keeps
# request bodies small and failures cheap to retry.
ALBUM_BATCH = 500


@dataclass(frozen=True)
class Asset:
    id: str
    owner_id: str
    created_at: str  # upload time (ISO-8601) — drives the incremental watermark
    taken_at: str  # localDateTime — drives duplicate time windows
    original_file_name: str
    mime_type: str
    checksum: str
    is_favorite: bool
    width: int | None
    height: int | None
    camera_make: str | None
    camera_model: str | None

    @property
    def taken_ts(self) -> float:
        return parse_ts(self.taken_at)

    @classmethod
    def from_api(cls, d: dict) -> Asset:
        exif = d.get("exifInfo") or {}
        return cls(
            id=d["id"],
            owner_id=d["ownerId"],
            created_at=d["createdAt"],
            taken_at=d.get("localDateTime") or d.get("fileCreatedAt") or d["createdAt"],
            original_file_name=d.get("originalFileName", ""),
            mime_type=d.get("originalMimeType") or "",
            checksum=d.get("checksum", ""),
            is_favorite=bool(d.get("isFavorite")),
            width=d.get("width") or exif.get("exifImageWidth"),
            height=d.get("height") or exif.get("exifImageHeight"),
            camera_make=exif.get("make"),
            camera_model=exif.get("model"),
        )


def parse_ts(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


class ImmichClient:
    def __init__(self, base_url: str, api_key: str, *, client: httpx.Client | None = None):
        self._http = client or httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0))
        self._base = f"{base_url}/api"
        self._headers = {"x-api-key": api_key, "accept": "application/json"}
        self._album_ids: dict[str, str] = {}

    def close(self) -> None:
        self._http.close()

    def _req(self, method: str, path: str, **kw) -> httpx.Response:
        resp = self._http.request(method, f"{self._base}{path}", headers=self._headers, **kw)
        resp.raise_for_status()
        return resp

    def whoami(self) -> dict:
        return self._req("GET", "/users/me").json()

    def iter_images(
        self, created_after: str | None = None, page_size: int = 1000
    ) -> Iterator[Asset]:
        """Yield every IMAGE asset, optionally only those uploaded after a timestamp.

        Uses the v3.2 structured filter + cursor pagination.
        """
        filt: dict = {"type": {"eq": "IMAGE"}}
        if created_after:
            filt["createdAt"] = {"gt": created_after}
        body: dict = {"filter": filt, "size": page_size, "withExif": True}
        while True:
            data = self._req("POST", "/search/metadata", json=body).json()["assets"]
            for item in data["items"]:
                yield Asset.from_api(item)
            cursor = data.get("nextCursor")
            if not cursor:
                return
            body["cursor"] = cursor

    def preview(self, asset_id: str) -> bytes | None:
        """Preview JPEG (~1440px). None when Immich hasn't generated it yet."""
        try:
            return self._req(
                "GET", f"/assets/{asset_id}/thumbnail", params={"size": "preview"}
            ).content
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (404, 400):
                return None
            raise

    def ensure_album(self, name: str, description: str) -> str:
        if name in self._album_ids:
            return self._album_ids[name]
        for album in self._req("GET", "/albums", params={"isOwned": "true"}).json():
            if album["albumName"] == name:
                self._album_ids[name] = album["id"]
                return album["id"]
        album = self._req(
            "POST", "/albums", json={"albumName": name, "description": description}
        ).json()
        log.info("created album", extra={"album": name, "album_id": album["id"]})
        self._album_ids[name] = album["id"]
        return album["id"]

    def add_to_album(self, album_id: str, asset_ids: list[str]) -> set[str]:
        """Add assets; returns ids now in the album (added or already present)."""
        ok: set[str] = set()
        for i in range(0, len(asset_ids), ALBUM_BATCH):
            chunk = asset_ids[i : i + ALBUM_BATCH]
            results = self._req("PUT", f"/albums/{album_id}/assets", json={"ids": chunk}).json()
            for r in results:
                if r.get("success") or r.get("error") == "duplicate":
                    ok.add(r["id"])
                else:
                    log.warning(
                        "album add failed", extra={"asset": r["id"], "error": r.get("error")}
                    )
        return ok
