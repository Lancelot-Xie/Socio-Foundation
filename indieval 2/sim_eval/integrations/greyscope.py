"""Local Greyscope AI-text detector integration for UserLM.

The configured checkpoint is a merged Hugging Face sequence-classification
model, not a generative/vLLM model.  Its bundled ``calibration.json`` is part
of the scoring contract: it defines the prompt, score orientation/scaling,
and label thresholds.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..errors import ConfigurationError, OptionalDependencyError, ValidationError


GREYSCOPE_MODEL_ID = "yaoandy107/greyscope-qwen3.5-4b"
GREYSCOPE_DEFAULT_MODEL_PATH = (
    "models/greyscope-qwen3.5-4b"
)
GREYSCOPE_DEFAULT_MODEL_REVISION = "bb25d4158a6795c6fce225156c0e330084ffb665"
GREYSCOPE_CONTRACT_REVISION = "greyscope-v1-calibrated-seqcls-v1"
GREYSCOPE_PREPROCESS_REVISION = "greyscope-v1-editlens-clean-v1"


@dataclass(frozen=True)
class GreyscopeDetectorConfig:
    enabled: bool = False
    model_path: str = GREYSCOPE_DEFAULT_MODEL_PATH
    model_id: str = GREYSCOPE_MODEL_ID
    model_revision: str = GREYSCOPE_DEFAULT_MODEL_REVISION
    device: str = "auto"
    batch_size: int = 8

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "GreyscopeDetectorConfig":
        values = dict(raw or {})
        backend = str(values.get("backend") or "local_greyscope_seqcls").strip()
        if backend != "local_greyscope_seqcls":
            raise ConfigurationError(
                "resources.ai_text_detector.backend must be local_greyscope_seqcls"
            )
        enabled = values.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigurationError("resources.ai_text_detector.enabled must be boolean")
        batch_size = values.get("batch_size", 8)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ConfigurationError(
                "resources.ai_text_detector.batch_size must be a positive integer"
            )
        config = cls(
            enabled=enabled,
            model_path=str(values.get("model_path") or GREYSCOPE_DEFAULT_MODEL_PATH).strip(),
            model_id=str(values.get("model_id") or GREYSCOPE_MODEL_ID).strip(),
            model_revision=str(
                values.get("model_revision") or GREYSCOPE_DEFAULT_MODEL_REVISION
            ).strip(),
            device=str(values.get("device") or "auto").strip().casefold(),
            batch_size=batch_size,
        )
        if config.device not in {"auto", "cpu", "cuda", "mps"} and not re.fullmatch(
            r"cuda:\d+", config.device
        ):
            raise ConfigurationError(
                "resources.ai_text_detector.device must be auto, cpu, mps, cuda, or cuda:N"
            )
        if config.enabled and not all(
            (config.model_path, config.model_id, config.model_revision)
        ):
            raise ConfigurationError(
                "enabled Greyscope detector requires model_path, model_id, and model_revision"
            )
        if config.model_id != GREYSCOPE_MODEL_ID:
            raise ConfigurationError(
                f"resources.ai_text_detector.model_id is locked to {GREYSCOPE_MODEL_ID}"
            )
        if config.model_revision != GREYSCOPE_DEFAULT_MODEL_REVISION:
            raise ConfigurationError(
                "resources.ai_text_detector.model_revision is locked to "
                f"{GREYSCOPE_DEFAULT_MODEL_REVISION}"
            )
        return config

    def identity(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backend": "local_greyscope_seqcls",
            "model_path": self.model_path,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "device": self.device,
            "batch_size": self.batch_size,
            "contract_revision": GREYSCOPE_CONTRACT_REVISION,
            "preprocess_revision": GREYSCOPE_PREPROCESS_REVISION,
            "network_required_at_runtime": False,
        }

    def preflight(self) -> dict[str, Any]:
        root = Path(self.model_path)
        required_files = {
            name: (root / name).is_file()
            for name in (
                "calibration.json",
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
            )
        }
        dependencies = {
            name: importlib.util.find_spec(name) is not None
            for name in ("torch", "transformers", "emoji")
        }
        ready = (
            (not self.enabled)
            or (root.is_dir() and all(required_files.values()) and all(dependencies.values()))
        )
        blockers = []
        if self.enabled and not root.is_dir():
            blockers.append("model_path_missing")
        elif self.enabled and not all(required_files.values()):
            blockers.append("checkpoint_files_incomplete")
        blockers.extend(
            f"dependency_missing:{name}"
            for name, available in dependencies.items()
            if self.enabled and not available
        )
        return {
            **self.identity(),
            "model_path_exists": root.is_dir(),
            "required_files": required_files,
            "dependencies": dependencies,
            "ready_for_local_load": ready,
            "blocking_reasons": blockers,
        }

    def protocol_identity(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.identity().items()
            if key not in {"model_path", "device"}
        }


def _validated_calibration(raw: Mapping[str, Any]) -> dict[str, Any]:
    calibration = dict(raw)
    required = {
        "task",
        "n_buckets",
        "label_names",
        "bucket_descriptions",
        "flip",
        "score_min",
        "score_max",
        "h_thresh",
        "ai_thresh",
        "prompt_template",
        "lowercase",
        "max_length",
    }
    missing = sorted(required - set(calibration))
    if missing:
        raise ConfigurationError(
            f"Greyscope calibration.json is missing fields: {missing}"
        )
    if calibration["task"] != "editlens-ternary":
        raise ConfigurationError("Greyscope calibration task must be editlens-ternary")
    if calibration.get("head_type", "seqcls") != "seqcls":
        raise ConfigurationError("Greyscope calibration must use the pinned seqcls head")
    n_buckets = calibration["n_buckets"]
    if isinstance(n_buckets, bool) or n_buckets != 4:
        raise ConfigurationError("Greyscope calibration must contain four buckets")
    descriptions = calibration["bucket_descriptions"]
    if (
        not isinstance(descriptions, Sequence)
        or isinstance(descriptions, (str, bytes))
        or list(descriptions) != ["none", "light", "moderate", "heavy"]
    ):
        raise ConfigurationError("Greyscope bucket descriptions do not match the pinned model")
    labels = calibration["label_names"]
    if (
        not isinstance(labels, Sequence)
        or isinstance(labels, (str, bytes))
        or list(labels) != ["human", "AI-generated", "AI-edited"]
    ):
        raise ConfigurationError("Greyscope labels do not match the pinned model")
    prompt = calibration["prompt_template"]
    if not isinstance(prompt, str) or "{text}" not in prompt:
        raise ConfigurationError("Greyscope prompt_template must contain {text}")
    lo, hi = float(calibration["score_min"]), float(calibration["score_max"])
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        raise ConfigurationError("Greyscope score_min/score_max are invalid")
    h_threshold = float(calibration["h_thresh"])
    ai_threshold = float(calibration["ai_thresh"])
    if not 0 <= h_threshold < ai_threshold <= 1:
        raise ConfigurationError("Greyscope ternary thresholds are invalid")
    max_length = calibration["max_length"]
    if isinstance(max_length, bool) or max_length != 2048:
        raise ConfigurationError("Greyscope calibration max_length must be 2048")
    if calibration["lowercase"] is not True:
        raise ConfigurationError("Greyscope calibration must enable training-time cleaning")
    return calibration


def _validated_checkpoint_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = dict(raw)
    if checkpoint.get("model_type") != "qwen3_5_text":
        raise ConfigurationError("Greyscope config.json has an unexpected model_type")
    architectures = checkpoint.get("architectures")
    if architectures != ["Qwen3_5ForSequenceClassification"]:
        raise ConfigurationError("Greyscope config.json has an unexpected architecture")
    id_to_label = checkpoint.get("id2label")
    if not isinstance(id_to_label, Mapping) or set(id_to_label) != {"0", "1", "2", "3"}:
        raise ConfigurationError("Greyscope config.json must expose four classifier logits")
    if checkpoint.get("dtype") != "bfloat16":
        raise ConfigurationError("Greyscope config.json must identify the bf16 checkpoint")
    return checkpoint


def decode_greyscope_logits(
    logits: Sequence[float], calibration: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the checkpoint's calibrated four-bucket decode without NumPy."""

    config = _validated_calibration(calibration)
    n_buckets = int(config["n_buckets"])
    if config.get("head_type", "seqcls") != "seqcls":
        raise ConfigurationError(
            "this integration supports the downloaded Greyscope seqcls checkpoint only"
        )
    if len(logits) != n_buckets:
        raise ValidationError(
            f"Greyscope returned {len(logits)} logits; calibration requires {n_buckets}"
        )
    values = [float(value) for value in logits]
    if not all(math.isfinite(value) for value in values):
        raise ValidationError("Greyscope returned non-finite logits")
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    probabilities = [value / denominator for value in exponentials]
    scalar = sum(
        probability * index for index, probability in enumerate(probabilities)
    ) / (n_buckets - 1)
    oriented = -scalar if bool(config["flip"]) else scalar
    low, high = float(config["score_min"]), float(config["score_max"])
    ai_involvement = min(max((oriented - low) / (high - low), 0.0), 1.0)
    if ai_involvement < float(config["h_thresh"]):
        label_index = 0
    elif ai_involvement > float(config["ai_thresh"]):
        label_index = 1
    else:
        label_index = 2
    return {
        "label": str(config["label_names"][label_index]),
        "ai_involvement": ai_involvement,
        "human_likelihood": 1.0 - ai_involvement,
        "bucket_probs": {
            str(name): probability
            for name, probability in zip(config["bucket_descriptions"], probabilities)
        },
    }


_BOILERPLATE_STARTS = (
    "Sure", "Here", "Abstract", "Title", "I'm happy to help", "Certainly"
)


def clean_greyscope_text(text: str) -> str:
    """Reproduce the pinned Greyscope v1/EditLens preprocessing exactly."""

    try:
        import emoji  # type: ignore
    except ImportError as exc:
        raise OptionalDependencyError(
            "Greyscope scoring requires the optional `emoji` package; install .[greyscope]"
        ) from exc
    value = emoji.demojize(str(text))
    if "</think>" in value:
        value = value.split("</think>")[1].strip()
    paragraphs = [paragraph for paragraph in value.split("\n") if paragraph.strip()]
    if paragraphs:
        first = re.sub(r"^[^a-zA-Z0-9]*", "", paragraphs[0])
        first = emoji.replace_emoji(first, "")
        if any(first.startswith(prefix) for prefix in _BOILERPLATE_STARTS) and len(paragraphs) > 1:
            value = "\n".join(paragraphs[1:])
    return re.sub(r"\s+", " ", value.lower()).strip()


class GreyscopeLocalScorer:
    """Load one local Greyscope model and return auditable human-likeness evidence."""

    def __init__(self, config: GreyscopeDetectorConfig) -> None:
        if not config.enabled:
            raise ConfigurationError("cannot build a disabled Greyscope detector")
        self.config = config
        self.model_path = Path(config.model_path)
        if not self.model_path.is_dir():
            raise ConfigurationError(
                f"Greyscope model directory does not exist: {self.model_path}"
            )
        model_config_path = self.model_path / "config.json"
        calibration_path = self.model_path / "calibration.json"
        if not model_config_path.is_file():
            raise ConfigurationError(
                f"Greyscope model is missing config.json: {model_config_path}"
            )
        if not calibration_path.is_file():
            raise ConfigurationError(
                f"Greyscope model is missing calibration.json: {calibration_path}"
            )
        try:
            raw_model_config = json.loads(model_config_path.read_bytes())
            calibration_bytes = calibration_path.read_bytes()
            raw_calibration = json.loads(calibration_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(
                f"cannot read Greyscope checkpoint metadata from {self.model_path}"
            ) from exc
        if not isinstance(raw_model_config, Mapping):
            raise ConfigurationError("Greyscope config.json must contain an object")
        if not isinstance(raw_calibration, Mapping):
            raise ConfigurationError("Greyscope calibration.json must contain an object")
        self.checkpoint_config = _validated_checkpoint_config(raw_model_config)
        self.calibration = _validated_calibration(raw_calibration)
        self.calibration_sha256 = hashlib.sha256(calibration_bytes).hexdigest()
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise OptionalDependencyError(
                "Greyscope scoring requires torch and transformers>=5.5.0; install .[greyscope]"
            ) from exc
        self._torch = torch
        self.device = self._resolve_device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path),
                dtype=torch.bfloat16,
                local_files_only=True,
            ).eval()
        except (TypeError, ValueError, OSError) as exc:
            raise ConfigurationError(
                "failed to load Greyscope; the checkpoint requires transformers>=5.5.0"
            ) from exc
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = self.model.to(self.device)
        self._lock = threading.Lock()

    def _resolve_device(self, requested: str) -> str:
        if requested == "cuda" or requested.startswith("cuda:"):
            if not self._torch.cuda.is_available():
                raise ConfigurationError(
                    f"Greyscope device={requested} but CUDA is unavailable"
                )
            if requested.startswith("cuda:"):
                index = int(requested.split(":", 1)[1])
                if index >= self._torch.cuda.device_count():
                    raise ConfigurationError(
                        f"Greyscope requested {requested}, but only "
                        f"{self._torch.cuda.device_count()} CUDA devices are visible"
                    )
            return requested
        if requested == "mps":
            if not self._torch.backends.mps.is_available():
                raise ConfigurationError("Greyscope device=mps but MPS is unavailable")
            return "mps"
        if requested == "cpu":
            return "cpu"
        if self._torch.cuda.is_available():
            return "cuda"
        if self._torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def __call__(self, text: str) -> Mapping[str, Any]:
        return self.score_batch((text,))[0]

    def score_batch(self, texts: Sequence[str]) -> tuple[Mapping[str, Any], ...]:
        """Score one non-empty text batch and preserve input order."""

        if not texts:
            raise ValidationError("Greyscope batch cannot be empty")
        if len(texts) > self.config.batch_size:
            raise ValidationError(
                f"Greyscope batch has {len(texts)} texts, exceeding configured "
                f"batch_size={self.config.batch_size}"
            )
        source_texts = tuple(str(text).strip() for text in texts)
        if any(not text for text in source_texts):
            raise ValidationError("Greyscope cannot score empty text")
        bodies = tuple(
            clean_greyscope_text(text) if self.calibration["lowercase"] else text
            for text in source_texts
        )
        prompts = tuple(
            str(self.calibration["prompt_template"]).format(text=body)
            for body in bodies
        )
        untruncated = self.tokenizer(list(prompts), add_special_tokens=False)
        token_counts = tuple(len(values) for values in untruncated["input_ids"])
        maximum = int(self.calibration["max_length"])
        encoded = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=maximum,
            add_special_tokens=False,
        ).to(self.model.device)
        with self._lock, self._torch.inference_mode():
            logits_batch = self.model(**encoded).logits.float().cpu().tolist()
        if len(logits_batch) != len(source_texts):
            raise ValidationError(
                f"Greyscope returned {len(logits_batch)} logit rows; "
                f"expected {len(source_texts)}"
            )
        observed = len(source_texts)
        evidence = []
        for position, (source_text, token_count, logits) in enumerate(
            zip(source_texts, token_counts, logits_batch)
        ):
            decoded = decode_greyscope_logits(logits, self.calibration)
            evidence.append(
                {
                    **decoded,
                    "model_id": self.config.model_id,
                    "model_revision": self.config.model_revision,
                    "model_path": str(self.model_path),
                    "device": self.device,
                    "contract_revision": GREYSCOPE_CONTRACT_REVISION,
                    "preprocess_revision": GREYSCOPE_PREPROCESS_REVISION,
                    "calibration_sha256": self.calibration_sha256,
                    "max_length": maximum,
                    "input_token_count": token_count,
                    "truncated": token_count > maximum,
                    "word_count": len(re.findall(r"\b\w+\b", source_text)),
                    "text_sha256": hashlib.sha256(
                        source_text.encode("utf-8")
                    ).hexdigest(),
                    "batch": {
                        "configured_case_batch_size": self.config.batch_size,
                        "observed_case_batch_size": observed,
                        "position": position,
                    },
                }
            )
        return tuple(evidence)


__all__ = [
    "GREYSCOPE_CONTRACT_REVISION",
    "GREYSCOPE_DEFAULT_MODEL_PATH",
    "GREYSCOPE_DEFAULT_MODEL_REVISION",
    "GREYSCOPE_MODEL_ID",
    "GreyscopeDetectorConfig",
    "GreyscopeLocalScorer",
    "clean_greyscope_text",
    "decode_greyscope_logits",
]
