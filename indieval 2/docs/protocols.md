# Preserved protocols

This release preserves the existing implementation's task semantics. It does not
claim that every compatibility variant reproduces an upstream leaderboard.
Read the catalog's source and access notes before comparing results.

| Benchmark | Preserved evaluation contract |
| --- | --- |
| FANToM | Question-family parsers, semantic belief scoring, complete-set All/All* aggregation and accessibility/context separation. |
| Social-R1 | Canonical and explicitly labeled local compatibility inputs remain distinct; no inferred ATOMS labels. |
| LifeChoices | Four-option decisions, natural-language persona/context, answer-tag parser and accuracy. |
| BehaviorChain | Chain-stable pseudonyms, declared history condition, node scores and chain-level AvgScore/CumScore. |
| AlignX | Separate conditioning variants, frozen preference materials and deterministic option handling; direct choice is not reference-normalized likelihood. |
| HumanLLM | 20 candidates, five ranked selections, separate Top-1/Hit@5/reciprocal-rank metrics. |
| HUMANUAL | Domain-specific preprocessing, response/state alignment and pinned local semantic scorer. |
| UserLM | Section 3 and LiC remain separate; LiC isolates task/test gold from the fixed assistant and preserves task/domain/repetition aggregation. |
| tau-USI | Text tools, serial tool execution, 60 main rounds, assistant-step handoff, original-history questionnaire and fixed-seed missing-survey aggregation. |
| MirrorBench | Reference visibility restrictions, fixed-assistant dialogue, judging and human-anchored lexical comparisons. |
| CoSER | Character-private history, environment/next-speaker roles, 20-round limit, four critic dimensions and original scoring equations. |
| SOTOPIA | Evaluated-role scores and configuration-level aggregation, with role-pair coverage reported. |
| AgentSense | Private goals/information, speaker identity, three logical judges and the enforced judge temperature of 0.8. |

Supplemental adapters preserve their task-specific answer extraction, exact
matching, score clipping, missing-judge contribution and dialogue loops. Mistakes
predicts `TargetOption`. Sim-Doc/Sim-Math retain their eight-round limit and
respective five-/three-component rewards. Compatibility labels stay explicit.

Transport failures, target-format failures, budget outcomes, missing judgments
and unavailable metrics follow each adapter's existing rules. They are not
collapsed into one universal denominator. Aggregate completeness and coverage
must be inspected alongside scores.

The generic historical ID/sampling namespace constants are retained verbatim:
changing these non-identifying constants can change IDs, option order and sampled
groups. Descriptive protocol labels and supplemental IDs use neutral names.
Do not resume an earlier installation's checkpoints under renamed protocol IDs.

Input data were not reselected or transformed for this release. No real data are
shipped. Use the exact intended sample manifest and protocol resources when
reproducing an experiment; the generic sampling defaults do not reconstruct an
unpublished research subset.
