import random
import unittest
from dataclasses import replace
from pathlib import Path

from sim_eval.catalog import load_sampling_profiles
from sim_eval.contracts import BenchmarkCase, SourceManifest
from sim_eval.data.loaders import load_fixture_suite
from sim_eval.data.sampling import (
    DeterministicStratifiedSampler,
    SamplingPlan,
    resolve_sampling_plan,
    validate_no_group_leakage,
)
from sim_eval.errors import ValidationError


ROOT = Path(__file__).resolve().parents[1]


def social_case(index, dimension, group_id=None, split="test"):
    return BenchmarkCase(
        benchmark_id="social_r1",
        case_id=f"case-{index}",
        group_id=group_id or f"group-{index}",
        split=split,
        source_revision="rev",
        input_data={"question": "q", "options": ["A", "B"], "atoms_dimension": dimension},
        gold="A",
        metadata={"strata": {"atoms_dimension": dimension}},
    )


def social_source(count):
    return SourceManifest(
        benchmark_id="social_r1",
        source_kind="official",
        source_revision="rev",
        split="test",
        resolved_population=count,
        metadata={"canonical_population": 100},
    )


class SamplingTests(unittest.TestCase):
    def test_input_order_does_not_change_ordered_manifest(self) -> None:
        cases = [social_case(i, "belief" if i < 4 else "emotion") for i in range(8)]
        plan = SamplingPlan(
            benchmark_id="social_r1",
            profile="research",
            result_label="test",
            seed=91,
            strategy="stratified_group_hash",
            target=4,
            unit="test_question",
            strata=("atoms_dimension",),
            repetitions=1,
            raw={},
        )
        first_cases, first = DeterministicStratifiedSampler(plan, social_source(8)).select(cases)
        shuffled = list(cases)
        random.Random(44).shuffle(shuffled)
        second_cases, second = DeterministicStratifiedSampler(plan, social_source(8)).select(shuffled)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual([case.case_id for case in first_cases], [case.case_id for case in second_cases])
        self.assertEqual(sorted(first.quotas.values()), [2, 2])

    def test_material_seed_change_changes_manifest_fingerprint(self) -> None:
        cases = [social_case(i, "belief" if i % 2 else "emotion") for i in range(10)]
        plan = SamplingPlan("social_r1", "research", "test", 1, "stratified_group_hash", 4, "test_question", ("atoms_dimension",), 1, {})
        first = DeterministicStratifiedSampler(plan, social_source(10)).select(cases)[1]
        second = DeterministicStratifiedSampler(replace(plan, seed=2), social_source(10)).select(cases)[1]
        self.assertNotEqual(first.digest, second.digest)

    def test_group_integrity_wins_over_row_target(self) -> None:
        cases = [
            social_case(1, "belief", group_id="shared"),
            social_case(2, "belief", group_id="shared"),
            social_case(3, "belief", group_id="other"),
        ]
        plan = SamplingPlan("social_r1", "research", "test", 3, "stratified_group_hash", 1, "group", ("atoms_dimension",), 1, {})
        selected, manifest = DeterministicStratifiedSampler(plan, social_source(3)).select(cases)
        selected_groups = {case.group_id for case in selected}
        self.assertEqual(len(selected_groups), 1)
        expected_count = sum(case.group_id in selected_groups for case in cases)
        self.assertEqual(len(selected), expected_count)
        self.assertEqual(manifest.metadata["selected_group_count"], 1)

    def test_group_crossing_boundaries_is_rejected(self) -> None:
        train = [social_case(1, "belief", group_id="same", split="train")]
        test = [social_case(2, "belief", group_id="same", split="test")]
        with self.assertRaisesRegex(ValidationError, "group leakage"):
            validate_no_group_leakage({"train": train, "test": test})

    def test_default_refuses_undeclared_undersized_population(self) -> None:
        profiles = load_sampling_profiles(ROOT / "sim_eval/resources" / "sampling_profiles.json")
        plan = resolve_sampling_plan(profiles, "default", "lifechoices")
        fixtures = load_fixture_suite(ROOT / "tests" / "fixtures")
        cases, fixture_source = fixtures["lifechoices"]
        official_source = replace(fixture_source, source_kind="official")
        with self.assertRaisesRegex(ValidationError, "undersized fallback"):
            DeterministicStratifiedSampler(plan, official_source).select(cases)

    def test_declared_population_fallback_is_visible(self) -> None:
        profiles = load_sampling_profiles(ROOT / "sim_eval/resources" / "sampling_profiles.json")
        plan = resolve_sampling_plan(profiles, "default", "behaviorchain")
        cases, fixture_source = load_fixture_suite(ROOT / "tests" / "fixtures")["behaviorchain"]
        official_source = replace(fixture_source, source_kind="official")
        _, manifest = DeterministicStratifiedSampler(plan, official_source).select(cases)
        self.assertTrue(manifest.metadata["population_exhausted"])
        self.assertEqual(manifest.exclusions[0]["kind"], "population_shortfall")
        self.assertTrue(manifest.exclusions[0]["reason"])

    def test_fixtures_cannot_impersonate_default_profile(self) -> None:
        profiles = load_sampling_profiles(ROOT / "sim_eval/resources" / "sampling_profiles.json")
        plan = resolve_sampling_plan(profiles, "default", "behaviorchain")
        cases, source = load_fixture_suite(ROOT / "tests" / "fixtures")["behaviorchain"]
        with self.assertRaisesRegex(ValidationError, "offline_smoke"):
            DeterministicStratifiedSampler(plan, source).select(cases)


if __name__ == "__main__":
    unittest.main()
