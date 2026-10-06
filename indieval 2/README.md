# Indieval

Anonymous research code for protocol-aware human-simulation evaluation.
Python 3.10 or newer is required. The Python API remains `sim_eval`; the installed
command is `indieval`.

The framework supports 13 core benchmarks and 7 optional supplemental benchmarks.
Each adapter owns its prompts, visible information, parsers, scoring and aggregation.
Shared infrastructure provides model routing, episode budgets, concurrency,
checkpointing, scoring-only continuation and reports.

## Install and verify offline

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[supplemental]'
python -m sim_eval catalog validate
python -m sim_eval doctor
python -m unittest discover -s tests
python tools/check_release.py
```

The following workflow needs no model, credentials or network access:

```bash
python -m sim_eval run --profile offline_smoke --backend replay --output artifacts/demo
python -m sim_eval resume --output artifacts/demo
python -m sim_eval report --run artifacts/demo --output reports/demo.md
```

These results are labeled `synthetic_offline_smoke_not_a_benchmark_score`.
Fixtures are software tests, not real evaluation data or evidence of model quality.

## Qwen3-8B without thinking: LifeChoices example

The single-benchmark experiment example is
[`configs/examples/qwen3_8b_nothinking_lifechoices.yaml`](configs/examples/qwen3_8b_nothinking_lifechoices.yaml).
It uses an OpenAI-compatible local server, `enable_thinking: false`, temperature
0, a 1,024-token request cap and the preserved LifeChoices answer-tag protocol.
No helper model or LLM judge is needed for this benchmark.

1. Serve an authorized Qwen3-8B checkpoint at `http://127.0.0.1:8000/v1`, with
   the served model name `Qwen/Qwen3-8B` (or edit the example).
2. Install the optional tokenizer dependencies with `pip install -e '.[huggingface]'`.
3. Put the matching tokenizer files in `models/Qwen3-8B`, or set the example's
   `token_accounting.model_path` to their location. Pin the actual model and
   tokenizer revisions in place of the `replace-with-*` values.
4. Validate the wiring before making model requests:

```bash
python -m sim_eval evaluate   --benchmark lifechoices   --manifest examples/lifechoices/import_manifest.json   --runtime-config configs/examples/qwen3_8b_nothinking_lifechoices.yaml   --output artifacts/qwen3_lifechoices_demo   --validate-only --require-live-ready
```

Remove `--validate-only --require-live-ready` to run the synthetic example against
your model server. Repeat the same command and output directory to resume.
For real evaluation, supply your own authorized dataset manifest instead of the
synthetic example. A `live_ready: true` result does not verify the model
server, credentials or immutable checkpoint contents; check those separately.

## Qwen3-8B without thinking: full evaluation examples

[`configs/examples/qwen3_8b_nothinking_full.yaml`](configs/examples/qwen3_8b_nothinking_full.yaml)
covers all 13 core benchmarks in 14 entries (UserLM Section 3 and LiC separately).
[`configs/examples/qwen3_8b_nothinking_supplemental_full.yaml`](configs/examples/qwen3_8b_nothinking_supplemental_full.yaml)
covers all seven supplemental benchmarks. Run both for all supported tasks.
Both use `suite_mode: formal`, which selects every case and complete dependency
group in each supplied manifest. They do not supply or certify the data population.

Follow [the full evaluation setup](docs/full_evaluation.md) to configure data,
model revisions, support models, local scorers and the tau runtime before running.
These are configuration examples, not previously measured model results.

## Benchmarks and configuration

Core: FANToM, Social-R1, LifeChoices, BehaviorChain, AlignX, HumanLLM, HUMANUAL,
UserLM, tau-USI, MirrorBench, CoSER, SOTOPIA and AgentSense. UserLM Section 3 and
LiC are distinct evaluation protocols.

Supplemental: HiToM, ParaToMi, Mistakes, TwinVoice, SocSci210, Sim-Doc and Sim-Math,
under neutral IDs such as `supplemental_hitom`.

`sim_eval/resources/` contains the benchmark catalog, metric mappings, backend
dialects and model-independent protocol defaults needed by the library. These
are not experiment campaigns. To obtain an editable configuration for another
benchmark:

```bash
python tools/create_runtime.py --benchmark coser --output local_data/coser_runtime.yaml
```

Fill its role models, revisions, service locations and local resources explicitly.
Multi-benchmark `suite` and coverage `smoke` commands require an explicit
`--config`; there is no hidden default experiment or external service.

See [architecture](docs/architecture.md), [protocols](docs/protocols.md),
[data and runtime setup](docs/data_and_runtime.md), and
[anonymity and validation](docs/anonymity_and_validation.md).

## Release scope

This archive contains code, neutral protocol resources and synthetic fixtures.
It excludes research datasets, human annotations, model checkpoints, credentials,
private experiment campaigns, historical outputs and paper assets. Install optional
local scorers and provide licensed inputs when a benchmark needs them.
Upstream benchmark citations remain in the catalog. The code is Apache-2.0;
synthetic fixtures are CC0-1.0. See `LICENSE` and `NOTICE`.

## Anonymous source archive

`python tools/package_release.py` writes `dist/indieval-anonymous.zip` with fixed
ZIP metadata. See [validation results](docs/validation.json) for the tested scope.
