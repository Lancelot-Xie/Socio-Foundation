# Anonymous release and validation

The release includes no author list, affiliations, contact addresses, repository
remotes, credentials, private provider locations, training checkpoints, historical
results, paper assets or real evaluation datasets. Public upstream benchmark
citations and the Apache license are retained. Neutral supplemental identifiers
replace project-specific names; no reverse mapping is shipped.

Protocol resources are required library definitions, not copied experimental
campaigns. Public examples cover Qwen3-8B without thinking on LifeChoices and full core/supplemental suites.
Runtime paths and report labels are portable. Non-identifying deterministic hash
constants remain unchanged because they influence sampling and option ordering.

Validation covers:

- Standalone offline unit/integration tests using synthetic fixtures.
- All 13 core adapters plus all 7 supplemental adapters.
- A private comparison against the pre-release implementation across 55 synthetic
  cases and 178 captured requests: prompts, request parameters, predictions,
  per-case metrics and available core aggregates agree after neutral label mapping.
- Local mock HTTP evaluation of the example, followed by checkpoint resume.
- Build/install/resource checks and anonymous archive scanning.

Tests tied solely to private datasets, internal provider diagnostics or historical
training campaigns are not redistributed. Optional upstream-runtime and
platform-specific tests may skip when their prerequisites are absent. The checked
in validation report records the actual executed checks and limitations.

These checks do not establish real Qwen model performance, hosted-service
availability, canonical leaderboard equivalence or every optional neural scorer's
live operation. They verify packaging and preserved implemented behavior on the
stated synthetic coverage. Real evaluation requires operator-supplied resources.

Before publishing new outputs, rerun `python tools/check_release.py`. Generated
manifests intentionally record execution provenance and may include your local
paths or service locations; they are ignored by default and should not be added
to the anonymous code archive without a separate review.
