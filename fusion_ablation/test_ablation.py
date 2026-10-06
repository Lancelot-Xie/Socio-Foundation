"""CPU-only tests with synthetic models; no downloads or judge requests."""

import json
import subprocess
import sys

import pytest
import torch
import yaml

from opd.config import load_config
from opd.data import load_data, write_rows
from opd.full_export import complete_export
from opd.smoke import make_fixture
from fusion_ablation.demos import build
from fusion_ablation.merge import merge, normalized_weights
from fusion_ablation.run import source_config, training_config


@pytest.fixture
def fixture(tmp_path):
    torch.set_num_threads(1)
    make_fixture(tmp_path / "fixture")
    cfg = load_config(tmp_path / "fixture/direct.yaml")
    rows = load_data(cfg["data"]["train_file"])
    write_rows(cfg["data"]["train_file"], [rows[0], rows[8], rows[16]])
    cfg["rollout"]["max_new_tokens"] = 2
    return cfg


def test_weight_validation():
    teachers = [{"id": "a"}, {"id": "b"}]
    assert normalized_weights(teachers, {"a": 1, "b": 3}) == {"a": .25, "b": .75}
    for invalid in ({"a": 1}, {"a": -1, "b": 2}, {"a": 0, "b": 0}, {"a": float("nan"), "b": 1}):
        with pytest.raises(ValueError):
            normalized_weights(teachers, invalid)


def test_merge_matches_average_of_individually_merged_models(fixture, tmp_path):
    from peft import PeftModel, LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from opd.models import adapter_path
    cfg = fixture
    # Different ranks must combine actual deltas, not averaged A/B factors.
    extra = get_peft_model(AutoModelForCausalLM.from_pretrained(cfg["model"]["base_model"]),
        LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=6, target_modules=["c_attn"]))
    with torch.no_grad():
        for name, parameter in extra.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=.1)
    extra.save_pretrained(tmp_path / "rank2")
    cfg["teachers"][-1]["adapter"] = str(tmp_path / "rank2")
    supplied = {t["id"]: i + 1 for i, t in enumerate(cfg["teachers"])}
    weights = normalized_weights(cfg["teachers"], supplied)
    expected = None
    for teacher in cfg["teachers"]:
        base = AutoModelForCausalLM.from_pretrained(cfg["model"]["base_model"])
        standalone = PeftModel.from_pretrained(base, adapter_path(teacher["adapter"])).merge_and_unload()
        state = {k: v.detach().clone() * weights[teacher["id"]] for k, v in standalone.state_dict().items()}
        if expected is None:
            expected = state
        else:
            for key in state:
                expected[key] += state[key]
    output = tmp_path / "merged"
    merge(cfg, output, supplied)
    assert complete_export(output)
    actual = AutoModelForCausalLM.from_pretrained(output).state_dict()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)
    assert merge(cfg, output, supplied, resume=True) == str(output)
    with pytest.raises(ValueError, match="different inputs"):
        merge(cfg, output, supplied, scale=.5, resume=True)


def test_demos_keep_mixture_and_exact_tokens(fixture, tmp_path):
    cfg = fixture
    cfg["routing"]["tasks"]["coser"]["teachers"] = {"lifechoices": .5, "coser": .5}
    output = tmp_path / "demos.jsonl"
    report = build(cfg, output, candidates=2)
    rows = load_data(output, require_response=True)
    assert report["prompts"] == 3 and len(rows) == 6
    assert all("demo_teacher" not in r and r["trajectory_teacher"] for r in rows)
    from opd.data import tokenizer_signature, warmup_response
    from opd.models import load_tokenizer
    tokenizer = load_tokenizer(cfg)
    for row in rows:
        assert warmup_response(row, tokenizer, cfg, tokenizer_signature(tokenizer, cfg)) == row["response_token_ids"]
    offline = training_config(cfg, "dimension_offpolicy", output)
    sft = training_config(cfg, "task_sft", output)
    assert offline["train"]["trajectory_source"] == "teacher"
    assert offline["routing"] == cfg["routing"]
    assert sft["train"]["stage"] == "warmup"


@pytest.mark.parametrize("method", ["dimension_offpolicy", "task_sft", "merge_dimensions", "merge_tasks"])
def test_four_end_to_end_cli_paths(fixture, tmp_path, monkeypatch, method):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    config = tmp_path / "input.yaml"
    config.write_text(yaml.safe_dump(fixture))
    output = tmp_path / method
    cmd = [sys.executable, "-m", "fusion_ablation", "run", "--method", method,
        "--config", str(config), "--output", str(output), "--num-processes", "1",
        "--global-batch", "1", "--steps", "1", "--save-every", "1", "--no-evaluate"]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert complete_export(output / "full_model")
    if not method.startswith("merge_"):
        cfg = load_config(output / "configs/final.yaml")
        assert cfg["qgpi"]["enabled"] is False
        assert cfg["train"]["trajectory_source"] == "teacher"
        from transformers import AutoModelForCausalLM
        initial = AutoModelForCausalLM.from_pretrained(fixture["model"]["base_model"]).state_dict()
        final = AutoModelForCausalLM.from_pretrained(output / "full_model").state_dict()
        assert any(not torch.equal(initial[k], final[k]) for k in initial)


def test_hierarchy_source_reuses_dimensions_without_final_student(fixture, tmp_path):
    from opd.experiments import Runner
    source = tmp_path / "hierarchy"
    (source / "configs").mkdir(parents=True)
    (source / "configs/preflight.yaml").write_text(yaml.safe_dump(fixture))
    plan = {"workflow": "hierarchy_five", "seeds": [17], "dimensions": ["F", "S"], "fusion": {}}
    (source / "hierarchy_manifest.json").write_text(json.dumps({"plan": plan}))
    data = source / "hierarchy_data/seed_17"
    rows = load_data(fixture["data"]["train_file"])
    write_rows(data / "train.jsonl", rows)
    write_rows(data / "validation.jsonl", rows)
    state = {}
    for dim in plan["dimensions"]:
        stage = f"seed_17/dimension_{dim}/offline_forward"
        path = source / "runs" / stage / "final"
        path.mkdir(parents=True)
        (path / "opd_metadata.json").write_text(json.dumps({"dimension": dim, "student_mode": "lora",
            "base_model": fixture["model"]["base_model"]}))
        state[stage] = {"status": "complete"}
    (source / "stages.json").write_text(json.dumps(state))
    runner = Runner({"output_dir": str(tmp_path / "new"), "evaluation_split": "validation"}, dry_run=True)
    cfg, _ = source_config(source, "dimension_offpolicy", 17, runner)
    assert [t["id"] for t in cfg["teachers"]] == ["dim_F", "dim_S"]
    assert cfg["routing"]["tasks"]["coser__FS"]["teachers"] == {"dim_F": .5, "dim_S": .5}
    direct, _ = source_config(source, "task_sft", 17, runner)
    assert direct["teachers"] == fixture["teachers"]


def test_dry_run_resume_and_export_evaluation(fixture, tmp_path, monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    config = tmp_path / "input.yaml"
    config.write_text(yaml.safe_dump(fixture))
    output = tmp_path / "planned"
    cmd = [sys.executable, "-m", "fusion_ablation", "run", "--method", "merge_tasks",
        "--config", str(config), "--output", str(output), "--num-processes", "1", "--global-batch", "1"]
    for flags in (["--dry-run"], ["--resume"], ["--resume"]):
        completed = subprocess.run(cmd + flags, capture_output=True, text=True, timeout=180)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        if "--dry-run" in flags:
            assert not (output / "full_model").exists()
    assert complete_export(output / "full_model")
    assert (output / "evaluation/final.jsonl.summary.json").is_file()
