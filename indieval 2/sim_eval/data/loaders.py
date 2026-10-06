"""Local-only import manifests, format readers, normalization, and schema probing."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..contracts import BenchmarkCase, SourceManifest
from ..errors import ConfigurationError, OptionalDependencyError, ValidationError
from ..json_utils import canonical_json
from .ids import derive_case_id, derive_group_id
from .schemas import SCHEMAS, probe_case


SOCIAL_R1_HUMAN_SIM_FORMAT = "social_r1_human_sim_jsonl"
SOCIAL_R1_COMPAT_VARIANT = "supplemental_compat_v1"
HUMANUAL_OFFICIAL_COLLECTION_FORMAT = "humanual_official_collection_v1"
HUMANUAL_DOMAINS = ("news", "book", "opinion", "politics", "chat", "email")
_SOCIAL_R1_COMPAT_POLICIES = {
    "none",
    "report_exact_message_sequence",
    "exclude_exact_message_sequence",
}


@dataclass(frozen=True)
class ImportSpec:
    benchmark_id: str
    path: Path
    format: str
    source_kind: str
    source_revision: str
    split: str
    field_map: Mapping[str, str] = field(default_factory=dict)
    checksum_sha256: str | None = None
    canonical_population: int | str | None = None
    license_acknowledged: bool = False
    urls: Sequence[str] = field(default_factory=tuple)
    license: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    reference_train_path: Path | None = None
    reference_train_checksum_sha256: str | None = None
    contamination_policy: str = "none"
    expected_reference_overlap_count: int | None = None
    humanual_domains: Sequence[str] = field(default_factory=tuple)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _humanual_collection_sources(collection_path: Path) -> tuple[Mapping[str, Any], tuple[tuple[str, Path, int, str], ...]]:
    """Resolve and verify the six immutable raw-official-schema subset files."""

    try:
        collection = json.loads(collection_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"invalid HUMANUAL collection manifest {collection_path}: {exc}") from exc
    if not isinstance(collection, Mapping) or collection.get("benchmark") != "HUMANUAL":
        raise ValidationError("HUMANUAL collection manifest must declare benchmark=HUMANUAL")
    domains = collection.get("domains")
    if not isinstance(domains, Mapping) or set(domains) != set(HUMANUAL_DOMAINS):
        raise ValidationError(
            f"HUMANUAL collection must contain exactly the six domains {list(HUMANUAL_DOMAINS)}"
        )
    resolved: list[tuple[str, Path, int, str]] = []
    for domain in HUMANUAL_DOMAINS:
        entry = domains[domain]
        if not isinstance(entry, Mapping):
            raise ValidationError(f"HUMANUAL collection domain {domain!r} must be an object")
        declared_path = Path(str(entry.get("path") or ""))
        candidates = (
            collection_path.parent / domain / "records.jsonl",
            collection_path.parent / declared_path,
            collection_path.parent.parent / declared_path,
        )
        source_path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        if source_path is None:
            raise ConfigurationError(
                f"HUMANUAL collection cannot resolve records for domain {domain!r}: {declared_path}"
            )
        rows = entry.get("records")
        expected_sha = str(entry.get("sha256") or "").lower()
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ValidationError(f"HUMANUAL collection domain {domain!r} requires a positive record count")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise ValidationError(f"HUMANUAL collection domain {domain!r} requires a SHA-256 digest")
        actual_sha = file_sha256(source_path)
        if actual_sha.lower() != expected_sha:
            raise ValidationError(
                f"checksum mismatch for {source_path}: expected {expected_sha}, got {actual_sha}"
            )
        resolved.append((domain, source_path, rows, actual_sha))
    declared_total = collection.get("total_records")
    resolved_total = sum(item[2] for item in resolved)
    if declared_total != resolved_total:
        raise ValidationError(
            f"HUMANUAL collection total_records={declared_total!r}, expected {resolved_total}"
        )
    return collection, tuple(resolved)


def load_import_spec(path: str | Path) -> ImportSpec:
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"import manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid import manifest JSON: {exc}") from exc
    required = ("benchmark_id", "path", "source_kind", "source_revision", "split")
    missing = [key for key in required if not raw.get(key)]
    if missing:
        raise ConfigurationError(f"import manifest missing fields: {missing}")
    source_path = Path(raw["path"])
    if not source_path.is_absolute():
        source_path = (manifest_path.parent / source_path).resolve()
    reference_train_path = raw.get("reference_train_path")
    if reference_train_path is not None:
        reference_train_path = Path(reference_train_path)
        if not reference_train_path.is_absolute():
            reference_train_path = (manifest_path.parent / reference_train_path).resolve()
    format_name = str(raw.get("format") or source_path.suffix.lstrip(".")).lower()
    raw_humanual_domains = raw.get("humanual_domains")
    humanual_domains: tuple[str, ...] = ()
    if raw_humanual_domains is not None:
        if format_name != HUMANUAL_OFFICIAL_COLLECTION_FORMAT:
            raise ConfigurationError(
                "humanual_domains is only valid for format="
                f"{HUMANUAL_OFFICIAL_COLLECTION_FORMAT!r}"
            )
        if (
            isinstance(raw_humanual_domains, (str, bytes))
            or not isinstance(raw_humanual_domains, Sequence)
            or not raw_humanual_domains
        ):
            raise ConfigurationError("humanual_domains must be a nonempty array")
        normalized_domains = tuple(
            str(value).strip().casefold() for value in raw_humanual_domains
        )
        if any(not value for value in normalized_domains):
            raise ConfigurationError("humanual_domains may not contain empty values")
        if len(normalized_domains) != len(set(normalized_domains)):
            raise ConfigurationError("humanual_domains may not contain duplicates")
        unsupported_domains = sorted(set(normalized_domains) - set(HUMANUAL_DOMAINS))
        if unsupported_domains:
            raise ConfigurationError(
                f"unsupported HUMANUAL domains: {unsupported_domains}; "
                f"expected a subset of {list(HUMANUAL_DOMAINS)}"
            )
        # Store the selection in the canonical project order so equivalent
        # manifests produce identical case ordering and source identities.
        requested = set(normalized_domains)
        humanual_domains = tuple(domain for domain in HUMANUAL_DOMAINS if domain in requested)
    contamination_policy = str(raw.get("contamination_policy") or "none")
    if contamination_policy not in _SOCIAL_R1_COMPAT_POLICIES:
        raise ConfigurationError(
            "contamination_policy must be one of "
            f"{sorted(_SOCIAL_R1_COMPAT_POLICIES)}"
        )
    if contamination_policy != "none" and reference_train_path is None:
        raise ConfigurationError(
            f"contamination_policy={contamination_policy!r} requires reference_train_path"
        )
    if raw.get("reference_train_checksum_sha256") and reference_train_path is None:
        raise ConfigurationError("reference_train_checksum_sha256 requires reference_train_path")
    expected_overlap = raw.get("expected_reference_overlap_count")
    if expected_overlap is not None and (
        isinstance(expected_overlap, bool) or not isinstance(expected_overlap, int) or expected_overlap < 0
    ):
        raise ConfigurationError("expected_reference_overlap_count must be a non-negative integer")
    if expected_overlap is not None and reference_train_path is None:
        raise ConfigurationError("expected_reference_overlap_count requires reference_train_path")
    return ImportSpec(
        benchmark_id=str(raw["benchmark_id"]),
        path=source_path,
        format=format_name,
        source_kind=str(raw["source_kind"]),
        source_revision=str(raw["source_revision"]),
        split=str(raw["split"]),
        field_map=dict(raw.get("field_map") or {}),
        checksum_sha256=raw.get("checksum_sha256"),
        canonical_population=raw.get("canonical_population"),
        license_acknowledged=bool(raw.get("license_acknowledged", False)),
        urls=tuple(raw.get("urls") or ()),
        license=raw.get("license"),
        metadata=dict(raw.get("metadata") or {}),
        reference_train_path=reference_train_path,
        reference_train_checksum_sha256=raw.get("reference_train_checksum_sha256"),
        contamination_policy=contamination_policy,
        expected_reference_overlap_count=expected_overlap,
        humanual_domains=humanual_domains,
    )


def _read_records(spec: ImportSpec) -> Iterable[Mapping[str, Any]]:
    if not spec.path.exists():
        raise ConfigurationError(f"local source file not found: {spec.path}")
    if spec.checksum_sha256:
        actual = file_sha256(spec.path)
        if actual.lower() != spec.checksum_sha256.lower():
            raise ValidationError(f"checksum mismatch for {spec.path}: expected {spec.checksum_sha256}, got {actual}")
    if spec.format == HUMANUAL_OFFICIAL_COLLECTION_FORMAT:
        if spec.benchmark_id != "humanual":
            raise ValidationError(
                f"format={HUMANUAL_OFFICIAL_COLLECTION_FORMAT!r} is only valid for benchmark_id='humanual'"
            )
        _, all_sources = _humanual_collection_sources(spec.path)
        selected_domains = set(spec.humanual_domains or HUMANUAL_DOMAINS)
        sources = tuple(
            source for source in all_sources if source[0] in selected_domains
        )
        for domain, source_path, expected_rows, _ in sources:
            observed_rows = 0
            with source_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValidationError(
                            f"invalid HUMANUAL JSONL at {source_path}:{line_number}: {exc}"
                        ) from exc
                    if not isinstance(value, dict):
                        raise ValidationError(
                            f"HUMANUAL record at {source_path}:{line_number} must be an object"
                        )
                    if "__humanual_domain" in value:
                        raise ValidationError("HUMANUAL raw rows may not use reserved __humanual_domain")
                    observed_rows += 1
                    yield {**value, "__humanual_domain": domain}
            if observed_rows != expected_rows:
                raise ValidationError(
                    f"HUMANUAL domain {domain!r} contains {observed_rows} rows, expected {expected_rows}"
                )
        return
    if spec.format in {"jsonl", SOCIAL_R1_HUMAN_SIM_FORMAT}:
        with spec.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValidationError(f"invalid JSONL at {spec.path}:{line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValidationError(f"JSONL record at {spec.path}:{line_number} must be an object")
                yield value
        return
    if spec.format == "json":
        value = json.loads(spec.path.read_text(encoding="utf-8"))
        records = value.get("records") if isinstance(value, dict) else value
        if not isinstance(records, list):
            raise ValidationError("JSON source must be an array or an object with a records array")
        for index, item in enumerate(records):
            if not isinstance(item, dict):
                raise ValidationError(f"JSON record #{index} must be an object")
            yield item
        return
    if spec.format == "csv":
        with spec.path.open(encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    if spec.format in {"parquet", "pq"}:
        try:
            import pyarrow.parquet as parquet  # type: ignore
        except ImportError as exc:
            raise OptionalDependencyError(
                "Parquet import requires pyarrow. Install it in your evaluation environment or convert the file to JSONL; no data was read."
            ) from exc
        for batch in parquet.ParquetFile(spec.path).iter_batches():
            yield from batch.to_pylist()
        return
    raise ConfigurationError(
        f"unsupported local format {spec.format!r}; use jsonl, json, csv, parquet, "
        f"or {SOCIAL_R1_HUMAN_SIM_FORMAT}"
    )


def _normalized_overlap_text(value: Any) -> str:
    """Normalize only representation noise; do not do fuzzy or semantic matching."""

    text = unicodedata.normalize("NFC", str(value)).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def _social_r1_human_sim_payload(
    record: Mapping[str, Any],
    index: int,
    *,
    expected_split: str,
) -> dict[str, Any]:
    """Validate and unpack the author-project human-sim compatibility wrapper."""

    user_id = record.get("user_id")
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValidationError(f"record #{index} requires a nonempty user_id")
    user_meta = record.get("user_meta")
    if not isinstance(user_meta, Mapping):
        raise ValidationError(f"record #{index} requires object-valued user_meta")
    if user_meta.get("split") != expected_split:
        raise ValidationError(
            f"record #{index} user_meta.split={user_meta.get('split')!r}, expected {expected_split!r}"
        )
    if user_meta.get("dataset") != "social-r1-data":
        raise ValidationError(
            f"record #{index} user_meta.dataset must be exactly 'social-r1-data'"
        )

    conversations = record.get("conversations")
    if (
        isinstance(conversations, (str, bytes))
        or not isinstance(conversations, Sequence)
        or len(conversations) != 1
        or not isinstance(conversations[0], Mapping)
    ):
        raise ValidationError(f"record #{index} requires exactly one conversation object")
    conversation = conversations[0]
    conversation_id = conversation.get("id")
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ValidationError(f"record #{index} conversation requires a nonempty id")
    if conversation.get("source") != "social-r1-data":
        raise ValidationError(
            f"record #{index} conversation.source must be exactly 'social-r1-data'"
        )

    messages = conversation.get("messages")
    if (
        isinstance(messages, (str, bytes))
        or not isinstance(messages, Sequence)
        or len(messages) != 2
        or not all(isinstance(message, Mapping) for message in messages)
    ):
        raise ValidationError(f"record #{index} requires exactly two message objects")
    roles = [message.get("role") for message in messages]
    if roles != ["user", "assistant"]:
        raise ValidationError(
            f"record #{index} message roles must be exactly ['user', 'assistant']; got {roles!r}"
        )
    prompt_text = messages[0].get("content")
    assistant_text = messages[1].get("content")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValidationError(f"record #{index} user message content must be nonempty text")
    if not isinstance(assistant_text, str) or not assistant_text.strip():
        raise ValidationError(f"record #{index} assistant message content must be nonempty text")

    marker_matches = list(re.finditer(r"(?m)^Options:\s*$", prompt_text))
    if len(marker_matches) != 1:
        raise ValidationError(
            f"record #{index} user prompt must contain exactly one standalone Options: marker"
        )
    marker = marker_matches[0]
    question = prompt_text[: marker.start()].strip()
    option_block = prompt_text[marker.end() :]
    if not question:
        raise ValidationError(f"record #{index} story/question text before Options: is empty")
    parsed_options: list[tuple[str, str]] = []
    for line_number, line in enumerate(option_block.splitlines(), start=1):
        if not line.strip():
            continue
        match = re.fullmatch(r"\s*([A-Z])[.)]\s+(.+?)\s*", line)
        if not match:
            raise ValidationError(
                f"record #{index} option line #{line_number} must be 'A. nonempty text'"
            )
        parsed_options.append((match.group(1), match.group(2)))

    source_metadata = conversation.get("metadata")
    if not isinstance(source_metadata, Mapping):
        raise ValidationError(f"record #{index} conversation.metadata must be an object")
    if source_metadata.get("task") != "social_reasoning_mcq":
        raise ValidationError(
            f"record #{index} metadata.task must be exactly 'social_reasoning_mcq'"
        )
    num_options = source_metadata.get("num_options")
    if (
        isinstance(num_options, bool)
        or not isinstance(num_options, int)
        or not 2 <= num_options <= 6
    ):
        raise ValidationError(f"record #{index} metadata.num_options must be an integer from 2 through 6")
    expected_labels = [chr(ord("A") + offset) for offset in range(num_options)]
    labels = [label for label, _ in parsed_options]
    if labels != expected_labels:
        raise ValidationError(
            f"record #{index} option labels must be contiguous {expected_labels!r}; got {labels!r}"
        )
    if len(parsed_options) != num_options:
        raise ValidationError(
            f"record #{index} parsed {len(parsed_options)} options but metadata.num_options={num_options}"
        )

    answer_letter = source_metadata.get("answer_letter")
    answer_text = source_metadata.get("answer_text")
    if not isinstance(answer_letter, str) or answer_letter not in expected_labels:
        raise ValidationError(
            f"record #{index} metadata.answer_letter must be one of {expected_labels!r}"
        )
    if not isinstance(answer_text, str) or not answer_text.strip():
        raise ValidationError(f"record #{index} metadata.answer_text must be nonempty text")
    option_by_label = dict(parsed_options)
    if _normalized_overlap_text(option_by_label[answer_letter]) != _normalized_overlap_text(answer_text):
        raise ValidationError(
            f"record #{index} metadata.answer_text does not exactly match the selected option"
        )
    assistant_match = re.fullmatch(r"\s*([A-Z])[.)]\s+(.+?)\s*", assistant_text, flags=re.DOTALL)
    if not assistant_match:
        raise ValidationError(
            f"record #{index} assistant gold must be formatted as 'A. answer text'"
        )
    if assistant_match.group(1) != answer_letter:
        raise ValidationError(
            f"record #{index} assistant answer letter disagrees with metadata.answer_letter"
        )
    if _normalized_overlap_text(assistant_match.group(2)) != _normalized_overlap_text(answer_text):
        raise ValidationError(
            f"record #{index} assistant answer text disagrees with metadata.answer_text"
        )

    message_sequence = [
        {
            "role": str(message["role"]),
            "content": _normalized_overlap_text(message["content"]),
        }
        for message in messages
    ]
    overlap_digest = hashlib.sha256(
        json.dumps(message_sequence, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "user_id": user_id.strip(),
        "conversation_id": conversation_id.strip(),
        "prompt_text": prompt_text.strip(),
        "question": question,
        "options": [text for _, text in parsed_options],
        "answer_letter": answer_letter,
        "answer_text": answer_text.strip(),
        "num_options": num_options,
        "source_task": str(source_metadata["task"]),
        "source_language": str(source_metadata.get("language") or ""),
        "overlap_digest": overlap_digest,
    }


def normalize_social_r1_human_sim_record(
    spec: ImportSpec,
    record: Mapping[str, Any],
    index: int,
) -> BenchmarkCase:
    """Normalize locally authorized compatibility Social-R1 data without claiming official status."""

    if spec.benchmark_id != "social_r1":
        raise ValidationError(
            f"format={SOCIAL_R1_HUMAN_SIM_FORMAT!r} is only valid for benchmark_id='social_r1'"
        )
    if spec.source_kind != "local_compatibility":
        raise ValidationError(
            f"format={SOCIAL_R1_HUMAN_SIM_FORMAT!r} requires source_kind='local_compatibility'; "
            "the author-project snapshot is not the official ToMBench-Hard release"
        )
    if spec.canonical_population is not None:
        raise ValidationError(
            "Supplemental-compatible Social-R1 must leave canonical_population unset; "
            "its 198-row pool is noncanonical"
        )
    if spec.split != "test":
        raise ValidationError(
            f"format={SOCIAL_R1_HUMAN_SIM_FORMAT!r} only imports the project-side test pool"
        )
    declared_variant = spec.metadata.get("protocol_variant")
    if declared_variant != SOCIAL_R1_COMPAT_VARIANT:
        raise ValidationError(
            f"format={SOCIAL_R1_HUMAN_SIM_FORMAT!r} requires "
            f"metadata.protocol_variant={SOCIAL_R1_COMPAT_VARIANT!r}"
        )
    payload = _social_r1_human_sim_payload(record, index, expected_split=spec.split)
    case = BenchmarkCase(
        benchmark_id=spec.benchmark_id,
        case_id=derive_case_id(spec.benchmark_id, spec.source_revision, payload["user_id"]),
        group_id=derive_group_id(spec.benchmark_id, spec.source_revision, payload["conversation_id"]),
        split=spec.split,
        source_revision=spec.source_revision,
        input_data={
            "question": payload["question"],
            "prompt_text": payload["prompt_text"],
            "options": payload["options"],
            "num_options": payload["num_options"],
        },
        gold=payload["answer_letter"],
        metadata={
            "source_id": payload["user_id"],
            "source_group_id": payload["conversation_id"],
            "source_kind": spec.source_kind,
            "protocol_variant": SOCIAL_R1_COMPAT_VARIANT,
            "contamination_policy": spec.contamination_policy,
            "reference_train_configured": spec.reference_train_path is not None,
            "source_answer_text": payload["answer_text"],
            "strata": {
                "protocol_variant": SOCIAL_R1_COMPAT_VARIANT,
                "num_options": payload["num_options"],
            },
            "social_r1_source": {
                "wrapper": "human-sim",
                "dataset": "social-r1-data",
                "task": payload["source_task"],
                "language": payload["source_language"],
                "num_options": payload["num_options"],
                "canonical_tombench_hard": False,
            },
        },
    )
    probe_case(case)
    return case


def _get(record: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _mapped(record: Mapping[str, Any], field_map: Mapping[str, str], key: str, default: Any = None) -> Any:
    return _get(record, field_map.get(key, key), default)


def _decode_json_cell(value: Any, field_name: str, index: int) -> Any:
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"record #{index} has invalid JSON in {field_name}: {exc}") from exc
    return value


def normalize_record(spec: ImportSpec, record: Mapping[str, Any], index: int) -> BenchmarkCase:
    if spec.format == SOCIAL_R1_HUMAN_SIM_FORMAT:
        return normalize_social_r1_human_sim_record(spec, record, index)
    if spec.format == HUMANUAL_OFFICIAL_COLLECTION_FORMAT:
        return normalize_humanual_official_record(spec, record, index)
    from ..extensions.supplemental import BENCHMARK_IDS as SUPPLEMENTAL_IDS
    if spec.benchmark_id not in SCHEMAS and spec.benchmark_id not in SUPPLEMENTAL_IDS:
        raise ValidationError(f"no schema registered for {spec.benchmark_id!r}")
    source_id = _mapped(record, spec.field_map, "source_id")
    if source_id is None:
        raise ValidationError(f"record #{index} requires source_id or a field_map.source_id mapping")
    source_group_id = _mapped(record, spec.field_map, "group_source_id", source_id)
    input_data = _decode_json_cell(_mapped(record, spec.field_map, "input"), "input", index)
    if not isinstance(input_data, Mapping):
        raise ValidationError(f"record #{index} requires an object-valued input field")
    strata = _decode_json_cell(_mapped(record, spec.field_map, "strata"), "strata", index)
    if not isinstance(strata, Mapping):
        raise ValidationError(f"record #{index} requires an object-valued strata field")
    record_split = str(_mapped(record, spec.field_map, "split", spec.split))
    if record_split != spec.split:
        raise ValidationError(f"record #{index} declares split {record_split!r}, expected {spec.split!r}")
    metadata = _decode_json_cell(_mapped(record, spec.field_map, "metadata", {}), "metadata", index)
    if not isinstance(metadata, Mapping):
        raise ValidationError(f"record #{index} metadata must be an object")
    case = BenchmarkCase(
        benchmark_id=spec.benchmark_id,
        case_id=derive_case_id(spec.benchmark_id, spec.source_revision, str(source_id)),
        group_id=derive_group_id(spec.benchmark_id, spec.source_revision, str(source_group_id)),
        split=spec.split,
        source_revision=spec.source_revision,
        input_data=dict(input_data),
        gold=_decode_json_cell(_mapped(record, spec.field_map, "gold"), "gold", index),
        metadata={
            **dict(metadata),
            "source_id": str(source_id),
            "source_group_id": str(source_group_id),
            "source_kind": spec.source_kind,
            "strata": dict(strata),
        },
    )
    probe_case(case)
    return case


def normalize_humanual_official_record(
    spec: ImportSpec,
    record: Mapping[str, Any],
    index: int,
) -> BenchmarkCase:
    """Normalize one released HUMANUAL row without rewriting its prompt or gold response."""

    if spec.benchmark_id != "humanual" or spec.split != "test":
        raise ValidationError("HUMANUAL collection imports require benchmark_id='humanual' and split='test'")
    domain = str(record.get("__humanual_domain") or "").strip().casefold()
    if domain not in HUMANUAL_DOMAINS:
        raise ValidationError(f"HUMANUAL record #{index} has an unsupported domain {domain!r}")
    prompt = _decode_json_cell(record.get("prompt"), "prompt", index)
    if isinstance(prompt, (str, bytes)) or not isinstance(prompt, Sequence) or not prompt:
        raise ValidationError(f"HUMANUAL record #{index} requires a nonempty prompt array")
    normalized_prompt: list[dict[str, str]] = []
    for message_index, message in enumerate(prompt):
        if not isinstance(message, Mapping):
            raise ValidationError(
                f"HUMANUAL record #{index} prompt message #{message_index} must be an object"
            )
        content = message.get("content")
        source_role = message.get("role")
        if not isinstance(content, str):
            raise ValidationError(
                f"HUMANUAL record #{index} prompt message #{message_index} requires string content"
            )
        if not isinstance(source_role, str) or not source_role.strip():
            raise ValidationError(
                f"HUMANUAL record #{index} prompt message #{message_index} requires a source role"
            )
        normalized_prompt.append(
            _normalize_humanual_prompt_message(
                domain=domain,
                message=message,
                record_index=index,
                message_index=message_index,
            )
        )
    persona = record.get("persona")
    completion = record.get("completion")
    user_id = record.get("user_id")
    post_id = record.get("post_id")
    turn_id = record.get("turn_id")
    if not isinstance(persona, str):
        raise ValidationError(f"HUMANUAL record #{index} requires string persona")
    if not isinstance(completion, str) or not completion.strip():
        raise ValidationError(f"HUMANUAL record #{index} requires nonempty completion")
    if user_id is None or not str(user_id).strip():
        raise ValidationError(f"HUMANUAL record #{index} requires user_id")
    if post_id is None or not str(post_id).strip():
        raise ValidationError(f"HUMANUAL record #{index} requires post_id")
    if isinstance(turn_id, bool) or not isinstance(turn_id, int) or turn_id <= 0:
        raise ValidationError(f"HUMANUAL record #{index} requires a positive integer turn_id")
    source_id = canonical_json(
        {
            "domain": domain,
            "user_id": str(user_id),
            "post_id": str(post_id),
            "turn_id": turn_id,
            "timestamp": record.get("timestamp"),
        }
    )
    case = BenchmarkCase(
        benchmark_id="humanual",
        case_id=derive_case_id("humanual", spec.source_revision, source_id),
        group_id=derive_group_id("humanual", spec.source_revision, source_id),
        split="test",
        source_revision=spec.source_revision,
        input_data={
            "domain": domain,
            "persona": persona,
            "prompt": normalized_prompt,
            "target_user_id": str(user_id),
            "post_id": str(post_id),
            "turn_id": turn_id,
        },
        gold=completion,
        metadata={
            "source_id": source_id,
            "source_group_id": source_id,
            "source_kind": spec.source_kind,
            "strata": {"domain": domain, "user": str(user_id)},
            "source_raw_files_unmodified": True,
            "official_preprocessing_revision": (
                "zou-group/humanlm@6faaf072b14b3efb4d434237d6d2af13f7a91d00:"
                "humanlm/process_dataset.py"
            ),
        },
    )
    probe_case(case)
    return case


def _humanual_message_metadata(
    message: Mapping[str, Any],
    *,
    record_index: int,
    message_index: int,
) -> dict[str, Any]:
    raw = message.get("metadata")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"HUMANUAL record #{record_index} prompt message #{message_index} has invalid metadata JSON"
            ) from exc
    if not isinstance(raw, Mapping):
        raise ValidationError(
            f"HUMANUAL record #{record_index} prompt message #{message_index} requires object metadata"
        )
    return dict(raw)


def _humanual_text(value: Any, label: str, record_index: int, message_index: int) -> str:
    if not isinstance(value, str):
        raise ValidationError(
            f"HUMANUAL record #{record_index} prompt message #{message_index} requires string {label}"
        )
    return value


def _normalize_humanual_prompt_message(
    *,
    domain: str,
    message: Mapping[str, Any],
    record_index: int,
    message_index: int,
) -> dict[str, str]:
    """Apply the six released DatasetMapper transformations without mutating source rows."""

    raw_role = _humanual_text(message.get("role"), "role", record_index, message_index)
    raw_content = _humanual_text(message.get("content"), "content", record_index, message_index)
    if domain in {"chat", "email"}:
        return {"role": raw_role, "content": raw_content}

    metadata = _humanual_message_metadata(
        message,
        record_index=record_index,
        message_index=message_index,
    )
    if domain == "opinion":
        role = _humanual_text(metadata.get("author"), "metadata.author", record_index, message_index)
        return {"role": role, "content": raw_content}
    if domain == "politics":
        content = raw_content[len("POLITICS\n\n") :] if raw_content.startswith("POLITICS\n\n") else raw_content
        return {"role": raw_role, "content": content}
    if domain == "news":
        kind = _humanual_text(metadata.get("kind"), "metadata.kind", record_index, message_index)
        snippet = metadata.get("snippet")
        if not isinstance(snippet, Mapping):
            raise ValidationError(
                f"HUMANUAL record #{record_index} prompt message #{message_index} requires metadata.snippet"
            )
        if "#video" in kind:
            role = _humanual_text(
                snippet.get("channelTitle"), "metadata.snippet.channelTitle", record_index, message_index
            )
            title = _humanual_text(metadata.get("title"), "metadata.title", record_index, message_index)
            description = _humanual_text(
                metadata.get("description"), "metadata.description", record_index, message_index
            )
            transcript = _humanual_text(
                metadata.get("transcript"), "metadata.transcript", record_index, message_index
            )
            return {
                "role": role,
                "content": (
                    f"- Title: {title}\n\n"
                    f"- Description: {description}\n\n"
                    f"- Transcript: {transcript}"
                ),
            }
        if "#comment" in kind:
            role = _humanual_text(
                snippet.get("authorDisplayName"),
                "metadata.snippet.authorDisplayName",
                record_index,
                message_index,
            )
            return {"role": role, "content": raw_content}
        raise ValidationError(
            f"HUMANUAL record #{record_index} prompt message #{message_index} has unknown news kind {kind!r}"
        )
    if domain == "book":
        for key in ("details", "author"):
            if metadata.get(key) is not None and isinstance(metadata[key], str):
                try:
                    metadata[key] = ast.literal_eval(metadata[key])
                except (SyntaxError, ValueError) as exc:
                    raise ValidationError(
                        f"HUMANUAL record #{record_index} prompt message #{message_index} "
                        f"has invalid metadata.{key} literal"
                    ) from exc
        categories = metadata.get("categories")
        description = metadata.get("description")
        features = metadata.get("features")
        details = metadata.get("details")
        if not isinstance(categories, list) or not all(isinstance(item, str) for item in categories):
            raise ValidationError("HUMANUAL book metadata.categories must be a string array")
        if not isinstance(description, list) or not all(isinstance(item, str) for item in description):
            raise ValidationError("HUMANUAL book metadata.description must be a string array")
        if not isinstance(features, list) or not all(isinstance(item, str) for item in features):
            raise ValidationError("HUMANUAL book metadata.features must be a string array")
        if not isinstance(details, Mapping):
            raise ValidationError("HUMANUAL book metadata.details must be an object")
        # The released Amazon mapper interpolates these scalar fields directly;
        # nullable subtitle/price values therefore intentionally become "None".
        role = f"Amazon store: {metadata.get('store')}"
        content = (
            f"- Category: {'->'.join(categories)}\n"
            f"- Title: {metadata.get('title')}\n"
            f"- Subtitle: {metadata.get('subtitle')}\n"
            f"- Price: {metadata.get('price')}\n"
        )
        author = metadata.get("author")
        if isinstance(author, Mapping):
            content += f"- Author: {author.get('name')}\n"
            if "about" in author:
                about = author["about"]
                if not isinstance(about, list) or not all(isinstance(item, str) for item in about):
                    raise ValidationError("HUMANUAL book metadata.author.about must be a string array")
                content += "  " + " ".join(about) + "\n"
        content += (
            "- Description: " + " ".join(description) + "\n"
            "- Features:\n" + " ".join(features) + "\n"
            "- Details:\n" + "\n".join(f"  {key}: {value}" for key, value in details.items())
        )
        return {"role": role, "content": content}
    raise AssertionError(domain)


def load_local_cases(spec: ImportSpec) -> tuple[list[BenchmarkCase], SourceManifest]:
    if spec.source_kind in {"official", "authorized_local"} and not spec.license_acknowledged:
        raise ConfigurationError(
            f"{spec.source_kind} import for {spec.benchmark_id} requires license_acknowledged=true in its manifest"
        )
    indexed_records = list(enumerate(_read_records(spec), start=1))
    cases = [normalize_record(spec, record, index) for index, record in indexed_records]
    if not cases:
        raise ValidationError(f"source contains no records: {spec.path}")
    raw_population = len(cases)
    contamination_audit: dict[str, Any] = {
        "policy": spec.contamination_policy,
        "comparison_revision": "exact-role-content-sequence-nfc-line-endings-v1",
        "reference_checked": False,
        "overlap_count": None,
        "excluded_count": 0,
        "status": "not_checked",
    }
    transformations: list[Mapping[str, Any]] = []
    reference_file_hashes: dict[str, str] = {}
    if spec.format == SOCIAL_R1_HUMAN_SIM_FORMAT and spec.reference_train_path is not None:
        reference_path = spec.reference_train_path
        if not reference_path.exists():
            raise ConfigurationError(f"reference train file not found: {reference_path}")
        reference_checksum = file_sha256(reference_path)
        if spec.reference_train_checksum_sha256:
            expected_checksum = spec.reference_train_checksum_sha256.lower()
            if reference_checksum.lower() != expected_checksum:
                raise ValidationError(
                    f"checksum mismatch for {reference_path}: expected "
                    f"{spec.reference_train_checksum_sha256}, got {reference_checksum}"
                )
        reference_spec = ImportSpec(
            benchmark_id="social_r1",
            path=reference_path,
            format=SOCIAL_R1_HUMAN_SIM_FORMAT,
            source_kind="local_compatibility",
            source_revision=spec.source_revision,
            split="train",
            metadata={"protocol_variant": SOCIAL_R1_COMPAT_VARIANT},
        )
        reference_keys: set[str] = set()
        for index, record in enumerate(_read_records(reference_spec), start=1):
            payload = _social_r1_human_sim_payload(record, index, expected_split="train")
            reference_keys.add(str(payload["overlap_digest"]))
        overlaps: list[tuple[BenchmarkCase, str]] = []
        for (index, record), case in zip(indexed_records, cases):
            payload = _social_r1_human_sim_payload(record, index, expected_split=spec.split)
            digest = str(payload["overlap_digest"])
            if digest in reference_keys:
                overlaps.append((case, digest))
        overlap_count = len(overlaps)
        if (
            spec.expected_reference_overlap_count is not None
            and overlap_count != spec.expected_reference_overlap_count
        ):
            raise ValidationError(
                "reference-train overlap count mismatch: expected "
                f"{spec.expected_reference_overlap_count}, got {overlap_count}"
            )
        excluded_count = 0
        if spec.contamination_policy == "exclude_exact_message_sequence":
            excluded_ids = {case.case_id for case, _ in overlaps}
            cases = [case for case in cases if case.case_id not in excluded_ids]
            excluded_count = overlap_count
            transformations.append(
                {
                    "type": "exclude_reference_train_overlap",
                    "comparison_revision": "exact-role-content-sequence-nfc-line-endings-v1",
                    "reference_filename": reference_path.name,
                    "reference_sha256": reference_checksum,
                    "excluded_count": excluded_count,
                    "excluded_source_ids": [
                        str(case.metadata["source_id"])
                        for case, _ in overlaps
                    ],
                    "excluded_message_sha256": sorted(digest for _, digest in overlaps),
                }
            )
        contamination_audit = {
            "policy": spec.contamination_policy,
            "comparison_revision": "exact-role-content-sequence-nfc-line-endings-v1",
            "reference_checked": True,
            "reference_filename": reference_path.name,
            "reference_sha256": reference_checksum,
            "reference_distinct_sequence_count": len(reference_keys),
            "overlap_count": overlap_count,
            "excluded_count": excluded_count,
            "status": (
                "known_overlap_excluded"
                if excluded_count
                else "known_overlap_retained"
                if overlap_count
                else "no_exact_overlap_found"
            ),
        }
        reference_file_hashes[reference_path.name] = reference_checksum
    elif spec.format == SOCIAL_R1_HUMAN_SIM_FORMAT and spec.contamination_policy != "none":
        raise ConfigurationError(
            f"contamination_policy={spec.contamination_policy!r} requires reference_train_path"
        )
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValidationError(f"duplicate stable case IDs in {spec.path}")
    source_file_hashes = {spec.path.name: file_sha256(spec.path)}
    humanual_metadata: dict[str, Any] = {}
    if spec.format == HUMANUAL_OFFICIAL_COLLECTION_FORMAT:
        collection, all_sources = _humanual_collection_sources(spec.path)
        selected_domains = tuple(spec.humanual_domains or HUMANUAL_DOMAINS)
        selected_domain_set = set(selected_domains)
        sources = tuple(
            source for source in all_sources if source[0] in selected_domain_set
        )
        source_file_hashes.update(
            {f"{domain}/records.jsonl": digest for domain, _, _, digest in sources}
        )
        humanual_metadata = {
            "official_score_eligible": False,
            "source_status": (
                "deterministic_100_per_domain_subset_of_pinned_official_test_splits"
                if selected_domain_set == set(HUMANUAL_DOMAINS)
                else "deterministic_selected_domain_subset_of_pinned_official_test_splits"
            ),
            "selected_domains": list(selected_domains),
            "domain_counts": {domain: rows for domain, _, rows, _ in sources},
            "selection_seed": collection.get("seed"),
        }
    source_manifest = SourceManifest(
        benchmark_id=spec.benchmark_id,
        source_kind=spec.source_kind,
        source_revision=spec.source_revision,
        split=spec.split,
        resolved_population=len(cases),
        urls=tuple(spec.urls),
        file_hashes=source_file_hashes,
        license=spec.license,
        transformations=tuple(transformations),
        metadata={
            **dict(spec.metadata),
            "canonical_population": spec.canonical_population,
            "raw_population": raw_population,
            "source_filename": spec.path.name,
            "format": spec.format,
            **humanual_metadata,
            **(
                {
                    "official_score_eligible": False,
                    "source_status": "author_project_local_compatibility_not_official_tombench_hard",
                    "contamination_audit": contamination_audit,
                    "reference_file_hashes": reference_file_hashes,
                }
                if spec.format == SOCIAL_R1_HUMAN_SIM_FORMAT
                else {}
            ),
        },
    )
    return cases, source_manifest


def load_fixture_suite(directory: str | Path) -> Mapping[str, tuple[list[BenchmarkCase], SourceManifest]]:
    root = Path(directory)
    index_path = root / "index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"invalid fixture index {index_path}: {exc}") from exc
    if index.get("source_kind") != "synthetic_fixture":
        raise ValidationError("fixture index must declare source_kind=synthetic_fixture")
    revision = index.get("source_revision")
    if not isinstance(revision, str) or not revision:
        raise ValidationError("fixture index requires source_revision")
    entries = index.get("benchmarks")
    if not isinstance(entries, Mapping):
        raise ValidationError("fixture index requires a benchmarks object")
    result: dict[str, tuple[list[BenchmarkCase], SourceManifest]] = {}
    for benchmark_id, filename in entries.items():
        spec = ImportSpec(
            benchmark_id=str(benchmark_id),
            path=root / str(filename),
            format="jsonl",
            source_kind="synthetic_fixture",
            source_revision=revision,
            split="fixture",
            license_acknowledged=True,
            license="CC0-1.0 repository-owned synthetic text",
            metadata={"fixture_label": "synthetic_offline_smoke_not_a_benchmark_score"},
        )
        result[str(benchmark_id)] = load_local_cases(spec)
    if set(result) != set(SCHEMAS):
        raise ValidationError(f"fixture coverage mismatch; missing={sorted(set(SCHEMAS)-set(result))}, extra={sorted(set(result)-set(SCHEMAS))}")
    return result
