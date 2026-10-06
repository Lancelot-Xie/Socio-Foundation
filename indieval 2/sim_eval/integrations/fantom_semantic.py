"""Official FANToM free-text belief semantic scorer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..benchmarks.fantom import FANTOM_SEMANTIC_CONTRACT, FANTOM_SEMANTIC_MODEL
from ..errors import ConfigurationError, OptionalDependencyError, ValidationError


FANTOM_SEMANTIC_DEFAULT_MODEL_PATH = (
    "models/all-roberta-large-v1"
)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


class FantomSentenceTransformerScorer:
    """Load the exact upstream model and return versioned cosine evidence."""

    def __init__(
        self,
        *,
        model_revision: str,
        model_path: str = FANTOM_SEMANTIC_DEFAULT_MODEL_PATH,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if not model_revision.strip():
            raise ConfigurationError("FANToM semantic model_revision must be pinned")
        if not model_path.strip():
            raise ConfigurationError("FANToM semantic model_path must be non-empty")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ConfigurationError("FANToM semantic batch_size must be a positive integer")
        self.model_path = Path(model_path.strip())
        if not self.model_path.is_dir():
            raise ConfigurationError(
                f"FANToM semantic model directory does not exist: {self.model_path}"
            )
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise OptionalDependencyError(
                "FANToM free-text belief scoring requires sentence-transformers"
            ) from exc
        kwargs: dict[str, Any] = {}
        if device:
            kwargs["device"] = device
        self.model = SentenceTransformer(str(self.model_path), **kwargs)
        self.model_revision = model_revision
        self.batch_size = batch_size

    def __call__(self, prediction: str, correct: str, wrong: str) -> Mapping[str, Any]:
        return self.score_batch(((prediction, correct, wrong),))[0]

    def protocol_identity(self) -> Mapping[str, Any]:
        return {
            "model_id": FANTOM_SEMANTIC_MODEL,
            "model_revision": self.model_revision,
            "contract_revision": FANTOM_SEMANTIC_CONTRACT,
            "metric": "cosine_similarity",
            "case_batch_size": self.batch_size,
        }

    def score_batch(
        self,
        items: Sequence[tuple[str, str, str]],
    ) -> tuple[Mapping[str, Any], ...]:
        """Score one non-empty case batch and preserve input order."""

        if not items:
            raise ValidationError("FANToM semantic batch cannot be empty")
        if len(items) > self.batch_size:
            raise ValidationError(
                f"FANToM semantic batch has {len(items)} cases, exceeding configured "
                f"batch_size={self.batch_size}"
            )
        flattened: list[str] = []
        for index, item in enumerate(items):
            if len(item) != 3 or any(not str(value).strip() for value in item):
                raise ValidationError(
                    f"FANToM semantic batch item #{index} requires prediction/correct/wrong text"
                )
            flattened.extend(str(value) for value in item)
        embeddings = self.model.encode(flattened, batch_size=self.batch_size)
        expected = len(items) * 3
        if len(embeddings) != expected:
            raise ValidationError(
                f"FANToM semantic encoder returned {len(embeddings)} embeddings; expected {expected}"
            )
        observed = len(items)
        evidence = []
        for position in range(observed):
            prediction_embedding, correct_embedding, wrong_embedding = embeddings[
                position * 3 : position * 3 + 3
            ]
            evidence.append(
                {
                    "contract_revision": FANTOM_SEMANTIC_CONTRACT,
                    "model_id": FANTOM_SEMANTIC_MODEL,
                    "model_revision": self.model_revision,
                    "metric": "cosine_similarity",
                    "correct_similarity": _cosine(
                        prediction_embedding, correct_embedding
                    ),
                    "wrong_similarity": _cosine(prediction_embedding, wrong_embedding),
                    "batch": {
                        "configured_case_batch_size": self.batch_size,
                        "observed_case_batch_size": observed,
                        "encoder_text_batch_size": self.batch_size,
                        "position": position,
                    },
                }
            )
        return tuple(evidence)


__all__ = ["FANTOM_SEMANTIC_DEFAULT_MODEL_PATH", "FantomSentenceTransformerScorer"]
