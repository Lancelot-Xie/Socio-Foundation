# Full Qwen3-8B no-thinking evaluation

Use the repository root as your working directory. There are two suites because
core and supplemental benchmarks have separate catalog contracts:

- `configs/examples/qwen3_8b_nothinking_full.yaml`: 14 entries, 13 core benchmarks.
- `configs/examples/qwen3_8b_nothinking_supplemental_full.yaml`: seven supplemental entries.

Run both to evaluate all 20 supported benchmarks. Every supplied record and its
complete dependency group is selected. "Full" means the full imported population,
not a guarantee that a manifest contains an entire upstream dataset. No real data
or human annotations are included. UserLM Section 3 and LiC must use their respective
protocol-compatible datasets, not two copies of one manifest. LiC uses 10 repetitions.

## Configure candidate and supporting models

In BOTH suite files, pin `model.model_revision` and
`model.token_accounting.model_revision`; supply the matching Qwen3-8B tokenizer at
`models/Qwen3-8B` or edit the path. Set `VLLM_BASE_URL` to your candidate's compatible
endpoint and, if needed, `VLLM_API_KEY`. The served model name defaults to
`Qwen/Qwen3-8B`; change `model.model` if your deployment uses another name. Thinking
is explicitly disabled for every candidate entry. Benchmark generation temperatures,
per-request token caps, episode budgets and turn limits come from the protocol
resources rather than a shared generation override.

The suites select `global_eval_model: Deepseek` as an EXAMPLE support route.
Edit `presets.Deepseek.model` and `presets.Deepseek.model_revision` in
`sim_eval/resources/global_eval_models.yaml` to the actual held-fixed support model.
Set `DEEPSEEK_BASE_URL` and `DEEPSEEK_API_KEY` in your shell; never put real keys in
tracked files. Check the `deepseek` dialect in `sim_eval/resources/api_profiles.yaml`
against your service. Its `.example` default endpoint is intentionally unusable.
No particular helper checkpoint is claimed as the paper's reproduction setting.

For a local support GPU endpoint, select `global_eval_model: Qwen` in BOTH files,
and configure `presets.Qwen` with the actual model/revision, its `base_url`,
`base_url_env: SUPPORT_BASE_URL`, and `api_key_env: SUPPORT_API_KEY`. Set the latter
environment variables for the support service so it is independent of the candidate.
The preset backend/profile must match the chosen service. You can also use the
other declared global presets with their corresponding provider configuration.

Keep `global_eval_model` set: omitting it makes the existing suite runner assign
the candidate to every network role. The selected preset applies to all support
roles while preserving each role's protocol settings, including judge temperatures.
Changing helper/judge models changes the experimental condition and may change
scores. Match published support-model revisions for reproduction.

## Supply data and local resources

Each suite entry lists its expected manifest at
`local_data/<entry-id>/import_manifest.json`. For example:

```text
local_data/
  lifechoices/import_manifest.json
  fantom/import_manifest.json
  userlm_section3/import_manifest.json
  userlm_lic/import_manifest.json
  tau_usi/import_manifest.json
  tau_usi/annotations.json
  tau_usi/difficulty.json
  supplemental_hitom/import_manifest.json
  ... one manifest for every entry in the two suites ...
models/
  Qwen3-8B/
```

Use the exact entry IDs in the YAML files; both UserLM manifests use
`benchmark_id: userlm`. Supplemental IDs and embedded protocol labels must use the
anonymous `supplemental_*` identifiers. See [data and runtime setup](data_and_runtime.md)
for manifest schema, checksums, licenses, neural scorers and the Linux code executor.

Install the required optional dependencies and scorer weights. Runtime JSON files
referenced by `runtime_config` contain editable scorer paths/revisions. The suites
set neural scorer devices to `cuda:0`; change `execution.local_device` in both files
if necessary. Model servers are independent of this scorer-device setting.
Concurrency defaults are deliberately modest and can be adjusted to your capacity.

For tau-USI, edit annotation/difficulty checksums in
`configs/examples/tau_usi_full_runtime.yaml`. Its relative paths resolve from that
file and point to `local_data/tau_usi/`. Supply all 165 required tasks and the pinned
tau-bench runtime, and set `SIM_EVAL_TAU_BENCH_ROOT` to the runtime checkout. Retain
the task runtime digest, 60 user rounds, 64 assistant steps per round, required
survey and candidate/assistant token caps of 256/512.

## Validate and run

After supplying data and resources, run these zero-model-call preflights:

```bash
python -m sim_eval suite --config configs/examples/qwen3_8b_nothinking_full.yaml --plan-only
python -m sim_eval suite --config configs/examples/qwen3_8b_nothinking_supplemental_full.yaml --plan-only
```

Resolve all `live_readiness_blockers` and inspect `live_ready`, not only `status`.
Missing manifests must be supplied before planning; this is not an offline demo.
Inspect `source_population`, `selected_case_count` and `selected_group_count` to
confirm every intended record/group is present. Preflight does not test service
reachability or certify actual checkpoint contents.

```bash
python -m sim_eval suite --config configs/examples/qwen3_8b_nothinking_full.yaml --output artifacts/qwen3_8b_core
python -m sim_eval suite --config configs/examples/qwen3_8b_nothinking_supplemental_full.yaml --output artifacts/qwen3_8b_supplemental
python scripts/summarize_eval_metrics.py artifacts/qwen3_8b_core
python scripts/summarize_eval_metrics.py artifacts/qwen3_8b_supplemental
```

Repeat the same run command/output directory to resume an unchanged experiment.
Use fresh output directories after changing data, models or protocol settings.
Keep per-benchmark metrics separate; do not average unrelated raw scores. The
examples have been checked using synthetic inputs; no real full-data model run
is asserted. Keep generated artifacts and private deployment settings out of
an anonymous code release.
