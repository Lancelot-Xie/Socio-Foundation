# Data and runtime setup

## Data manifests

`examples/lifechoices/import_manifest.json` points only to synthetic fixtures.
For real data, provide normalized JSONL or a supported source format and a manifest:

```json
{
  "benchmark_id": "lifechoices",
  "path": "records.jsonl",
  "format": "jsonl",
  "source_kind": "authorized_local",
  "source_revision": "immutable-dataset-revision",
  "split": "test",
  "checksum_sha256": "<sha256-of-records-file>",
  "license_acknowledged": true
}
```

Use the real revision and checksum, not the illustrative placeholders. Record
applicable licensing/selection metadata. Paths are relative to the manifest.
Schema requirements are in `sim_eval/data/schemas.py`; synthetic fixtures show
minimal shapes, not the distribution or scale of an actual benchmark.

```bash
python -m sim_eval data probe --manifest local_data/lifechoices/import_manifest.json
```

Data loaders verify checksums and role-visible fields and do not download gated
data. Keep selection groups intact for chains, sets and multi-profile templates.

## Model setup

The example candidate is Qwen3-8B with thinking disabled. Use a matching served
model name and local tokenizer, and replace both revision placeholders. Tokenizer
paths are resolved by the local loader; absolute paths are safest in generated
private configurations. Hosted endpoints and keys are supplied by the operator.
Default public profiles use localhost, public standard API locations or reserved
`.example` hostnames; `.example` services cannot be used as real endpoints.

`tools/create_runtime.py` exports a model-independent protocol template. Configure
all required candidate, partner, assistant, environment and judge roles. The
template preserves generation/turn/scoring settings; changing those creates a
new experimental condition. Generic global support-model presets intentionally
contain placeholders. Pin them explicitly before using a global override.

The `--validate-only` and `--plan-only` paths make zero model calls. Inspect
`live_ready` and its blockers, not only `status: valid`. Actual service reachability,
authorization and immutable model contents still require operational verification.

## Additional resources

- FANToM and HUMANUAL: the declared sentence-transformer and pinned revision.
- UserLM: the declared local naturalness scorer; code-task scoring also requires
  an isolated supported executor or a trusted revision-matched verifier cache.
- tau-USI: authorized annotations, their checksum, a difficulty map/checksum,
  and a pinned upstream tau-bench tool/runtime checkout. Configure the scoring paths and export `SIM_EVAL_TAU_BENCH_ROOT` as the
  absolute path to your runtime checkout. Source checkouts also default to
  `third_party/tau-bench`. The expected
  runtime digest is retained in the protocol definition. Human data are absent.
- Supplemental interactive tasks: `pip install -e '.[supplemental]'` supplies
  Pydantic 2; their data remain user supplied.

The core synthetic workflow runs on CPU. Real neural scorers and model serving
may need additional memory or accelerators. They are not started by installation.

## Campaigns and reports

Ready-to-edit Qwen3-8B no-thinking full-suite examples and commands are documented
in [full evaluation setup](full_evaluation.md).

A `suite` configuration explicitly supplies `suite_mode: formal`, a suite ID,
seed, candidate model, execution settings, catalog and entries with data manifests
and runtime configurations. Paths in a suite resolve relative to its file.
Consult `_prepare_smoke` in `sim_eval/smoke.py` for the validated campaign schema.
Use a new output directory after changing data, candidate or protocol.

```bash
python -m sim_eval suite --config local_data/suite.yaml --plan-only
python -m sim_eval suite --config local_data/suite.yaml --output artifacts/experiment
python scripts/summarize_eval_metrics.py artifacts/experiment
```

`smoke --config ...` is an explicit live diagnostic campaign; it is distinct from
the credential-free `run --backend replay` workflow.
