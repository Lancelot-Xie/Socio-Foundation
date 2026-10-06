"""Content-addressed, atomic verification-result cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..artifacts import atomic_write_json, load_json
from ..errors import ArtifactError
from ..json_utils import canonical_json, sha256_digest


class VerificationCache:
    def __init__(self, root: str | Path, *, store_completion: bool) -> None:
        self.root = Path(root).resolve()
        self.store_completion = store_completion

    @staticmethod
    def key_for(identity: Mapping[str, Any]) -> str:
        return sha256_digest(dict(identity))

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def load(self, identity: Mapping[str, Any]) -> Mapping[str, Any] | None:
        key = self.key_for(identity)
        path = self._path(key)
        if not path.is_file():
            return None
        value = load_json(path)
        if not isinstance(value, Mapping):
            raise ArtifactError(f"verification cache entry is not an object: {path}")
        if value.get("cache_key") != key or canonical_json(value.get("identity")) != canonical_json(identity):
            raise ArtifactError(f"verification cache identity mismatch: {path}")
        verification = value.get("verification")
        if not isinstance(verification, Mapping):
            raise ArtifactError(f"verification cache entry lacks verification result: {path}")
        return verification

    def store(
        self,
        identity: Mapping[str, Any],
        verification: Mapping[str, Any],
        *,
        completion: str,
    ) -> Path:
        key = self.key_for(identity)
        path = self._path(key)
        payload = {
            "schema_version": "1.0",
            "cache_key": key,
            "identity": dict(identity),
            "verification": dict(verification),
            **({"completion": completion} if self.store_completion else {}),
        }
        atomic_write_json(path, payload)
        return path


__all__ = ["VerificationCache"]

