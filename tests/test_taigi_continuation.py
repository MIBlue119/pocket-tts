"""Bounded continuation must preserve identity and require measured progress."""

import json
from pathlib import Path

import pytest

from training.scripts import continue_taigi_training as continuation


def metrics(step: int, loss: float = 1.0, similarity: float = 0.0) -> dict:
    return dict(
        step=step,
        loss=loss,
        finite=True,
        healthy=12,
        capped=0,
        repetitive=0,
        similarities=[similarity] * 12,
        prompts=12,
    )


def test_bridge_requires_undercoverage_or_directional_progress():
    baseline = metrics(0)
    current = metrics(200, 1.02)
    assert continuation.gate(baseline, current, baseline, 10000, 8)[0]
    assert not continuation.gate(baseline, current, baseline, 100, 8)[0]
    current["capped"] = 1
    assert not continuation.gate(baseline, current, baseline, 10000, 8)[0]


def test_later_gate_needs_loss_and_text_signal():
    previous = metrics(200, 1.02)
    candidate = metrics(500, 0.98)
    baseline = metrics(0)
    assert not continuation.gate(previous, candidate, baseline, 10000, 8)[0]
    candidate["similarities"] = [0.1] * 12
    assert continuation.gate(previous, candidate, baseline, 10000, 8)[0]
    candidate["finite"] = False
    assert not continuation.gate(previous, candidate, baseline, 10000, 8)[0]


def test_candidate_selection_does_not_just_pick_last():
    first = metrics(200, 0.9, 0.2)
    last = metrics(500, 1.1, 0.3)
    assert continuation.select_candidate([first, last], metrics(0)) is first


def test_fork_rejects_changed_parent_before_creating_child(tmp_path: Path):
    parent = tmp_path / "parent"
    parent.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("run_dir: " + str(parent) + "\n")
    (parent / "run_manifest.json").write_text(
        json.dumps({"identity": {"config_sha256": "wrong", "inputs": {}}})
    )
    with pytest.raises(ValueError, match="config.*identity"):
        continuation.verify_parent(config)


def test_absolute_reserve_stops_admission(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(continuation.time, "time", lambda: 100)
    with pytest.raises(TimeoutError):
        continuation.remaining(200, 100)
    assert continuation.remaining(200, 90) == 10


def parent_fixture(tmp_path: Path) -> Path:
    import torch
    import yaml
    from training.scripts.run_experiment import identity

    run = tmp_path / "parent-training"
    run.mkdir()
    train, valid, tokenizer = [
        tmp_path / name for name in ("train.jsonl", "valid.jsonl", "tokenizer.json")
    ]
    train.write_text('{"transcript":"訓練"}\n')
    valid.write_text('{"transcript":"保留"}\n')
    tokenizer.write_text("{}")
    config = {
        "run_dir": str(run),
        "max_steps": 200,
        "batch_size": 1,
        "grad_accum_steps": 8,
        "data": {"train_jsonl": str(train), "valid_jsonl": str(valid)},
        "model_overrides": {"flow_lm.lookup_table.tokenizer_path": str(tokenizer)},
        "optim": {"lr": 3e-5, "warmup_steps": 100},
    }
    path = tmp_path / "parent.yaml"
    path.write_text(yaml.safe_dump(config))
    current = identity(path, config, [str(tokenizer)])
    (run / "run_manifest.json").write_text(json.dumps({"identity": current}))
    torch.save(
        {
            "step": 200,
            "model": {"weight": torch.tensor([2.0])},
            "ema": {"weight": torch.tensor([1.0])},
        },
        run / "checkpoint_00000200.pt",
    )
    torch.save(
        {
            "optimizer": {
                "state": {0: {"exp_avg": torch.tensor([0.5])}},
                "param_groups": [{"lr": 3e-5}],
            }
        },
        run / "optim_00000200.pt",
    )
    return path


def test_real_fork_preserves_pair_and_mints_only_child_identity(tmp_path: Path):
    import torch
    from training.scripts.run_experiment import identity

    parent = parent_fixture(tmp_path)
    original, original_identity = continuation.verify_parent(parent)
    original_manifest = Path(original["run_dir"], "run_manifest.json").read_bytes()
    child = tmp_path / "child"
    path = continuation.fork_run(parent, child, 500)
    config, child_identity = continuation.verify_parent(path)
    assert {k: v for k, v in config.items() if k not in {"run_dir", "max_steps"}} == {
        k: v for k, v in original.items() if k not in {"run_dir", "max_steps"}
    }
    assert config["max_steps"] == 500
    for name in ("checkpoint_00000200.pt", "optim_00000200.pt"):
        assert (
            Path(config["run_dir"], name).read_bytes()
            == Path(original["run_dir"], name).read_bytes()
        )
    payload = torch.load(Path(config["run_dir"], "checkpoint_00000200.pt"), weights_only=True)
    assert payload["step"] == 200 and payload["ema"]["weight"].item() == 1
    assert child_identity == identity(
        path,
        config,
        [
            config["model_overrides"]["flow_lm.lookup_table.tokenizer_path"],
            str(child / "lineage.json"),
        ],
    )
    assert Path(original["run_dir"], "run_manifest.json").read_bytes() == original_manifest
    assert original_identity != child_identity


def test_changed_tokenizer_and_nonfinite_pair_refuse_fork(tmp_path: Path):
    import torch

    config_path = parent_fixture(tmp_path)
    config, _ = continuation.verify_parent(config_path)
    tokenizer = Path(config["model_overrides"]["flow_lm.lookup_table.tokenizer_path"])
    tokenizer.write_text("changed")
    with pytest.raises(ValueError, match="input identity"):
        continuation.fork_run(config_path, tmp_path / "refused", 500)
    assert not (tmp_path / "refused").exists()
    tokenizer.write_text("{}")
    checkpoint = Path(config["run_dir"], "checkpoint_00000200.pt")
    torch.save({"step": 200, "model": {"weight": torch.tensor([float("nan")])}}, checkpoint)
    with pytest.raises(ValueError, match="nonfinite"):
        continuation.fork_run(config_path, tmp_path / "refused", 500)


def test_wait_does_not_complete_on_poll_error_or_live_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import argparse
    import subprocess

    args = argparse.Namespace(
        parent_unit="parent.service",
        parent_job=tmp_path,
        absolute_deadline_epoch=10000,
        poll_seconds=1,
    )
    terminal = {
        "service": {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "Restart": "no",
            "CGroupProcessesCount": "0",
            "Result": "success",
            "ExecMainStartTimestampMonotonic": "100",
        },
        "job": {"state": "complete"},
    }
    responses = [OSError("temporary"), {"service": {"ActiveState": "active"}}, terminal]
    calls = []

    def run(*unused: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        value = responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return subprocess.CompletedProcess([], 0, json.dumps(value))

    monkeypatch.setattr(continuation.subprocess, "run", run)
    monkeypatch.setattr(continuation.time, "time", lambda: 1)
    monkeypatch.setattr(continuation.time, "sleep", lambda _: calls.append("wait"))
    continuation.wait_parent(args, {}, tmp_path / "report.json")
    assert calls == ["wait", "wait"]


def test_nonfinite_candidates_are_never_selected():
    candidate = metrics(200)
    candidate["finite"] = False
    with pytest.raises(ValueError, match="no finite"):
        continuation.select_candidate([candidate], metrics(0))


def test_export_is_raw_and_selected_contract_is_hash_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import torch
    import safetensors.torch
    from types import SimpleNamespace
    from training import args as args_module
    from training.modules import builders

    config_path = parent_fixture(tmp_path)
    config, _ = continuation.verify_parent(config_path)
    model = torch.nn.Module()
    model.flow_lm = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.flow_lm.weight.fill_(2)
        model.flow_lm.bias.fill_(3)
    checkpoint = Path(config["run_dir"], "checkpoint_00000200.pt")
    torch.save(
        {
            "step": 200,
            "model": model.state_dict(),
            "ema": {"flow_lm.weight": torch.tensor([[999.0]])},
        },
        checkpoint,
    )
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps({"state": "complete", "checkpoint_sha256": continuation.digest(checkpoint)})
    )
    monkeypatch.setattr(args_module, "load_args", lambda _: SimpleNamespace(seed=17))
    runtime = SimpleNamespace(model_dump=lambda **_: {"flow_lm": {"lookup_table": {"n_bins": 2}}})
    monkeypatch.setattr(builders, "build_models", lambda _: (model, torch.nn.Linear(1, 1), runtime))
    output = tmp_path / "continuation"
    output.mkdir()
    candidate = {
        **metrics(200),
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "evaluation": str(evaluation),
    }
    continuation.export_selection(candidate, metrics(0), output, [])
    selected = continuation.read(output / "selected.json")
    assert selected["state"] == "complete" and selected["quality_verified"] is False
    assert selected["candidate_id"] == "step-200-raw"
    assert set(selected["files"]) == {"weights", "config", "tokenizer", "evaluation", "provenance"}
    for record in selected["files"].values():
        assert Path(record["path"]).is_relative_to(output)
        assert continuation.digest(record["path"]) == record["sha256"]
    exported = safetensors.torch.load_file(selected["files"]["weights"]["path"])
    assert exported["flow_lm.weight"].item() == 2
    provenance = continuation.read(Path(selected["files"]["provenance"]["path"]))
    assert provenance["weights_variant"] == "raw"
    assert provenance["source_checkpoint_sha256"] == continuation.digest(checkpoint)


def test_evaluation_identity_and_hashes_are_required(tmp_path: Path):
    config_path = parent_fixture(tmp_path)
    config, _ = continuation.verify_parent(config_path)
    checkpoint = Path(config["run_dir"], "checkpoint_00000200.pt")
    prompts = tmp_path / "prompts.jsonl"
    rows = [{"id": str(i), "text": f"句子{i}"} for i in range(12)]
    prompts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    audio = tmp_path / "output.wav"
    audio.write_bytes(b"checked bytes")
    report = {
        "state": "complete",
        "step": 200,
        "weights": "raw_checkpoint",
        "config_sha256": continuation.digest(config_path),
        "checkpoint_sha256": continuation.digest(checkpoint),
        "tokenizer_sha256": continuation.digest(
            config["model_overrides"]["flow_lm.lookup_table.tokenizer_path"]
        ),
        "asr": {
            "model": continuation.BREEZE_MODEL,
            "revision": continuation.BREEZE_REVISION,
            "adapter": continuation.BREEZE_ADAPTER_VERSION,
        },
        "prompts": [
            {
                **r,
                "status": "audio",
                "generated_frames": 20,
                "audio_path": str(audio),
                "audio_sha256": continuation.digest(audio),
                "character_sequence_similarity": 0.1,
            }
            for r in rows
        ],
    }
    evaluation = tmp_path / "report.json"
    evaluation.write_text(json.dumps(report))
    records = [
        {"type": "train", "step": 200, "grad_norm": 1.0, "metrics": {"loss": 0.5}},
        {"type": "valid", "step": 200, "metrics": {"loss": 0.7}},
    ]
    summary = continuation.summarize(evaluation, records, 200, config_path, checkpoint, prompts)
    assert summary["healthy"] == 12 and summary["finite"]
    report["weights"] = "ema"
    evaluation.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="evaluation identity"):
        continuation.summarize(evaluation, records, 200, config_path, checkpoint, prompts)
    report["weights"] = "raw_checkpoint"
    report["tokenizer_sha256"] = "different"
    evaluation.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="evaluation identity"):
        continuation.summarize(evaluation, records, 200, config_path, checkpoint, prompts)


def test_deployment_receipt_cannot_misidentify_source(tmp_path: Path):
    receipt = tmp_path / "deployment.json"
    receipt.write_text(json.dumps({"source_commit": "a" * 40, "controller_sha256": "wrong"}))
    with pytest.raises(ValueError, match="deployment receipt"):
        continuation.source_identity(receipt)


def test_gate_halt_finishes_operation_and_exports_prior_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    parent_config = parent_fixture(tmp_path)
    config, _ = continuation.verify_parent(parent_config)
    parent_job = tmp_path / "job-200"
    parent_job.mkdir()
    (parent_job / "training.yaml").write_bytes(parent_config.read_bytes())
    (parent_job / "prompts.jsonl").write_text('{"id":"fixed","text":"固定"}\n')
    Path(config["run_dir"], "progress.jsonl").write_text("{}\n")
    args = continuation.parser().parse_args([str(parent_job), str(tmp_path / "continuation")])
    monkeypatch.setenv("INVOCATION_ID", "test")
    monkeypatch.setattr(continuation, "wait_parent", lambda *_: None)
    monkeypatch.setattr(continuation, "source_identity", lambda *_: {"test": True})

    def summary(
        evaluation: Path,
        records: list,
        step: int,
        source: Path,
        checkpoint: Path | None,
        prompts: Path,
    ) -> dict:
        return {
            **metrics(step),
            "evaluation": str(evaluation),
            "config": str(source),
            "checkpoint": str(checkpoint) if checkpoint else None,
        }

    monkeypatch.setattr(continuation, "summarize", summary)
    monkeypatch.setattr(continuation, "command", lambda *_: pytest.fail("gate must stop training"))
    exported = []
    monkeypatch.setattr(
        continuation, "export_selection", lambda chosen, *_: exported.append(chosen["step"])
    )
    result = continuation.run(args)
    assert result["state"] == "complete" and result["quality_verified"] is False
    assert result["gates"][0]["admit"] is False
    assert exported == [200]
    assert continuation.read(args.output / "job.json")["selected_step"] == 200
