# Architecture

`sim_eval.cli` dispatches single imported benchmarks through `evaluate`, explicit
campaigns through `suite`, coverage diagnostics through `smoke`, and synthetic
fixture replay through `run`. `suite` shares the scheduler in `smoke.py`.

Data manifests -> validated cases -> complete dependency groups -> adapter ->
role-routed model backend -> optional environment -> native metrics -> checkpoint
and report.

- `data/`: source/schema validation, hashes, stable IDs and grouped sampling.
- `benchmarks/`: model-visible prompts, parsers, scoring and aggregation.
- `environments/`: turn order, role visibility, tools, state and termination.
- `backends/`: HTTP/replay/local inference, role routing and shared endpoint limits.
- `generic_runner.py`: 11 core adapters plus the supplemental suite.
- `external_runner.py`: dedicated UserLM and tau-USI runners.
- `artifacts.py`, `judge_resume.py`: append-only attempts and validated continuation.
- `resources/`: packaged catalogs, protocol defaults and report mappings.

Cases separate `input_data` from evaluator-only `gold`. Backends receive requests,
not case objects or gold. Candidate and support roles retain separate logical
identities, even when they use one physical model. Candidate output budgets do
not charge fixed assistants or judges. Episode state transitions remain serial;
independent cases can run concurrently under shared endpoint limits.

A run identity pins data/sample, candidate, generation and protocol semantics.
Each `(case_id, repetition)` has an append-only attempt history; aggregation uses
the latest attempt. Failed attempts can retry. Valid zero scores remain valid.
Completed rollouts with missing required judgments can append scoring-only
continuations without changing existing output or valid scores. Protocol or
candidate changes require a new output directory. Support-model recovery retains
its existing provenance checks and reports mixed support-model conditions.

Installation paths and caller-selected output paths are independent. Reports do
not define new metrics. F/S/U/T/N group native metrics without a cross-task mean.
