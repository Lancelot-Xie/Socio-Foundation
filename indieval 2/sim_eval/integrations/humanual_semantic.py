"""Project-pinned embedding cosine scorer for HUMANUAL responses."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..errors import ConfigurationError, OptionalDependencyError, ValidationError


HUMANUAL_EMBEDDING_MODEL = "sentence-transformers/all-roberta-large-v1"
HUMANUAL_EMBEDDING_CONTRACT = "humanual-response-reference-cosine-project-pinned-v1"
HUMANUAL_EMBEDDING_DEFAULT_MODEL_PATH = (
    "models/all-roberta-large-v1"
)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValidationError(
            "HUMANUAL semantic encoder returned embeddings with different dimensions"
        )
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


class HumanualSentenceTransformerScorer:
    """Load one pinned encoder per run and batch response/reference pairs."""

    def __init__(
        self,
        *,
        model_revision: str,
        model_path: str = HUMANUAL_EMBEDDING_DEFAULT_MODEL_PATH,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if not model_revision.strip():
            raise ConfigurationError("HUMANUAL embedding model_revision must be pinned")
        if not model_path.strip():
            raise ConfigurationError("HUMANUAL embedding model_path must be non-empty")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ConfigurationError("HUMANUAL embedding batch_size must be a positive integer")
        self.model_path = Path(model_path.strip())
        if not self.model_path.is_dir():
            raise ConfigurationError(
                f"HUMANUAL embedding model directory does not exist: {self.model_path}"
            )
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise OptionalDependencyError(
                "HUMANUAL embedding cosine requires sentence-transformers"
            ) from exc
        kwargs: dict[str, Any] = {}
        if device:
            kwargs["device"] = device
        self.model = SentenceTransformer(str(self.model_path), **kwargs)
        self.model_revision = model_revision
        self.batch_size = batch_size

    def protocol_identity(self) -> Mapping[str, Any]:
        return {
            "model_id": HUMANUAL_EMBEDDING_MODEL,
            "model_revision": self.model_revision,
            "contract_revision": HUMANUAL_EMBEDDING_CONTRACT,
            "metric": "cosine_similarity",
            "encoder_selection": "project_pinned_user_requested",
            "case_batch_size": self.batch_size,
        }

    def score_batch(
        self,
        items: Sequence[tuple[str, str]],
    ) -> tuple[Mapping[str, Any], ...]:
        if not items:
            raise ValidationError("HUMANUAL embedding batch cannot be empty")
        if len(items) > self.batch_size:
            raise ValidationError(
                f"HUMANUAL embedding batch has {len(items)} cases, exceeding configured "
                f"batch_size={self.batch_size}"
            )
        flattened: list[str] = []
        for index, item in enumerate(items):
            if len(item) != 2:
                raise ValidationError(
                    f"HUMANUAL embedding batch item #{index} requires prediction/reference text"
                )
            prediction, reference = (str(value) for value in item)
            if not reference.strip():
                raise ValidationError(
                    f"HUMANUAL embedding batch item #{index} requires a nonempty reference"
                )
            flattened.extend((prediction, reference))
        embeddings = self.model.encode(flattened, batch_size=self.batch_size)
        expected = len(items) * 2
        if len(embeddings) != expected:
            raise ValidationError(
                f"HUMANUAL semantic encoder returned {len(embeddings)} embeddings; expected {expected}"
            )
        observed = len(items)
        evidence = []
        for position in range(observed):
            prediction_embedding, reference_embedding = embeddings[
                position * 2 : position * 2 + 2
            ]
            evidence.append(
                {
                    **self.protocol_identity(),
                    "similarity": _cosine(prediction_embedding, reference_embedding),
                    "batch": {
                        "configured_case_batch_size": self.batch_size,
                        "observed_case_batch_size": observed,
                        "encoder_text_batch_size": self.batch_size,
                        "position": position,
                    },
                }
            )
        return tuple(evidence)


__all__ = [
    "HUMANUAL_EMBEDDING_CONTRACT",
    "HUMANUAL_EMBEDDING_DEFAULT_MODEL_PATH",
    "HUMANUAL_EMBEDDING_MODEL",
    "HumanualSentenceTransformerScorer",
]
