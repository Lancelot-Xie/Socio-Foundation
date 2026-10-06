import json
import tempfile
import unittest
from pathlib import Path

from sim_eval.backends.replay import ReplayBackend
from sim_eval.benchmarks.social_r1 import (
    SOCIAL_R1_COMPAT_VARIANT,
    SocialR1Adapter,
    parse_social_r1_compat_response,
    parse_social_r1_response,
)
from sim_eval.contracts import ResultStatus
from sim_eval.data.loaders import (
    SOCIAL_R1_HUMAN_SIM_FORMAT,
    ImportSpec,
    load_local_cases,
)
from sim_eval.errors import ParseError, ValidationError


def human_sim_record(
    sample_id: str,
    *,
    split: str = "test",
    options: tuple[str, ...] = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta"),
    answer_index: int = 5,
) -> dict:
    labels = tuple(chr(ord("A") + index) for index in range(len(options)))
    prompt = (
        f"Story: Synthetic social situation {sample_id}.\n"
        "Question: Which option follows?\n"
        "Options:\n"
        + "\n".join(f"{label}. {text}" for label, text in zip(labels, options))
    )
    answer_letter = labels[answer_index]
    answer_text = options[answer_index]
    return {
        "user_id": f"social-r1:{split}:{sample_id}",
        "user_meta": {"country": "", "dataset": "social-r1-data", "split": split},
        "conversations": [
            {
                "id": f"social-r1-{split}-{sample_id}",
                "source": "social-r1-data",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"{answer_letter}. {answer_text}"},
                ],
                "metadata": {
                    "model": "",
                    "language": "English",
                    "task": "social_reasoning_mcq",
                    "answer_letter": answer_letter,
                    "answer_text": answer_text,
                    "num_options": len(options),
                },
            }
        ],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def compatibility_spec(path: Path, **updates) -> ImportSpec:
    values = {
        "benchmark_id": "social_r1",
        "path": path,
        "format": SOCIAL_R1_HUMAN_SIM_FORMAT,
        "source_kind": "local_compatibility",
        "source_revision": "author-project-test-revision",
        "split": "test",
        "metadata": {"protocol_variant": SOCIAL_R1_COMPAT_VARIANT},
    }
    values.update(updates)
    return ImportSpec(**values)


class SocialR1CompatibilityLoaderTests(unittest.TestCase):
    def test_native_wrapper_loads_dynamic_options_and_preserves_exact_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            row = human_sim_record("six-way")
            write_jsonl(path, [row])
            cases, source = load_local_cases(compatibility_spec(path))

        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case.metadata["protocol_variant"], SOCIAL_R1_COMPAT_VARIANT)
        self.assertNotIn("atoms_dimension", case.input_data)
        self.assertNotIn("atoms_dimension", case.metadata["strata"])
        self.assertEqual(case.input_data["num_options"], 6)
        self.assertEqual(case.gold, "F")
        self.assertFalse(source.metadata["official_score_eligible"])
        self.assertIn("not_official_tombench_hard", source.metadata["source_status"])

        adapter = SocialR1Adapter()
        request = adapter.build_request(case, model="fixture-model", seed=7)
        self.assertEqual(request.messages[1].content, row["conversations"][0]["messages"][0]["content"])
        self.assertNotIn("zeta", request.messages[0].content)
        result = adapter.execute_case(
            case,
            backend=ReplayBackend(responses={f"{case.case_id}:choice": "reason\n<answer>F</answer>\nend"}),
            run_id="compat-test-run",
            seed=7,
            model="fixture-model",
        )
        self.assertEqual(result.status, ResultStatus.COMPLETED)
        self.assertEqual(result.metrics[0].value, 1)
        self.assertFalse(result.metadata["social_r1"]["canonical_score_eligible"])
        aggregate = adapter.aggregate([result])
        self.assertEqual(aggregate["social_r1.accuracy"].value, 1.0)
        self.assertFalse(aggregate["social_r1.accuracy"].metadata["official_primary"])
        self.assertEqual(aggregate["social_r1.compat.accuracy.num_options.6"].denominator, 1)

    def test_wrapper_validates_roles_split_num_options_and_gold_consistency(self) -> None:
        mutations = []
        wrong_roles = human_sim_record("roles")
        wrong_roles["conversations"][0]["messages"][0]["role"] = "assistant"
        mutations.append((wrong_roles, "message roles"))
        wrong_split = human_sim_record("split")
        wrong_split["user_meta"]["split"] = "train"
        mutations.append((wrong_split, "user_meta.split"))
        wrong_count = human_sim_record("count")
        wrong_count["conversations"][0]["metadata"]["num_options"] = 4
        mutations.append((wrong_count, "option labels"))
        wrong_text = human_sim_record("gold")
        wrong_text["conversations"][0]["metadata"]["answer_text"] = "not zeta"
        mutations.append((wrong_text, "answer_text"))

        for row, message in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.jsonl"
                write_jsonl(path, [row])
                with self.assertRaisesRegex(ValidationError, message):
                    load_local_cases(compatibility_spec(path))

    def test_compatibility_format_cannot_be_mislabeled_as_official(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            write_jsonl(path, [human_sim_record("official-label")])
            spec = compatibility_spec(path, source_kind="official", license_acknowledged=True)
            with self.assertRaisesRegex(ValidationError, "local_compatibility"):
                load_local_cases(spec)

    def test_reference_train_filter_is_in_memory_auditable_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            test_path = root / "test.jsonl"
            duplicate_train = human_sim_record("shared", split="train")
            duplicate_test = human_sim_record("shared", split="test")
            # IDs differ by split, while the compared user+assistant message sequence is identical.
            unique_one = human_sim_record("unique-one", options=("yes", "no"), answer_index=1)
            unique_two = human_sim_record("unique-two", options=("left", "middle", "right"), answer_index=0)
            write_jsonl(train_path, [duplicate_train, human_sim_record("train-only", split="train")])
            write_jsonl(test_path, [duplicate_test, unique_one, unique_two])

            spec = compatibility_spec(
                test_path,
                reference_train_path=train_path,
                contamination_policy="exclude_exact_message_sequence",
                expected_reference_overlap_count=1,
            )
            cases, source = load_local_cases(spec)

        self.assertEqual(len(cases), 2)
        self.assertEqual({case.metadata["source_id"] for case in cases}, {
            "social-r1:test:unique-one",
            "social-r1:test:unique-two",
        })
        audit = source.metadata["contamination_audit"]
        self.assertEqual(audit["overlap_count"], 1)
        self.assertEqual(audit["excluded_count"], 1)
        self.assertEqual(audit["status"], "known_overlap_excluded")
        self.assertEqual(source.metadata["raw_population"], 3)
        self.assertEqual(source.resolved_population, 2)
        self.assertEqual(source.transformations[0]["excluded_count"], 1)
        self.assertNotIn("Synthetic social situation shared", json.dumps(source.transformations[0]))

    def test_wrong_expected_overlap_count_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            test_path = root / "test.jsonl"
            write_jsonl(train_path, [human_sim_record("same", split="train")])
            write_jsonl(test_path, [human_sim_record("same", split="test")])
            spec = compatibility_spec(
                test_path,
                reference_train_path=train_path,
                contamination_policy="report_exact_message_sequence",
                expected_reference_overlap_count=0,
            )
            with self.assertRaisesRegex(ValidationError, "overlap count mismatch"):
                load_local_cases(spec)


class SocialR1CompatibilityParserTests(unittest.TestCase):
    def test_compat_parser_matches_project_first_answer_tag_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            write_jsonl(path, [human_sim_record("parser")])
            case = load_local_cases(compatibility_spec(path))[0][0]
        choices = SocialR1Adapter().choices_for_case(case, seed=0)
        prediction = parse_social_r1_compat_response(
            "<think>private</think>prefix <answer>F</answer> suffix <answer>A</answer>",
            choices,
        )
        self.assertEqual(prediction.source_id, "F")
        with self.assertRaises(ParseError):
            parse_social_r1_compat_response("F", choices)

    def test_canonical_parser_remains_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            write_jsonl(path, [human_sim_record("parser")])
            case = load_local_cases(compatibility_spec(path))[0][0]
        choices = SocialR1Adapter().choices_for_case(case, seed=0)
        with self.assertRaises(ParseError):
            parse_social_r1_response("prefix <answer>F</answer>", choices)


if __name__ == "__main__":
    unittest.main()
