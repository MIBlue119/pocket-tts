"""Longer exploration stays bounded without treating a pilot quality gate as authorization."""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from training.scripts import extend_taigi_training as extension
from training.scripts.extend_taigi_training import wait_first


def setup_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[argparse.Namespace, list[list[str]]]:
    parent, first = tmp_path / "parent200", tmp_path / "first1000"
    for directory in (parent, first):
        directory.mkdir()
        (directory / "training.yaml").write_text("fixed config")
    (parent / "prompts.jsonl").write_text("fixed12prompts")
    args = extension.parser().parse_args(
        [
            str(first),
            str(parent),
            str(tmp_path / "longer"),
            "--absolute-deadline-epoch",
            "20000",
            "--owner-authorized-longer-training",
        ]
    )
    monkeypatch.setenv("INVOCATION_ID", "unit")
    monkeypatch.setattr(extension.time, "time", lambda: 1000)
    monkeypatch.setattr(extension, "source_record", lambda: {"hash": "source"})
    monkeypatch.setattr(extension, "wait_first", lambda *_: None)
    monkeypatch.setattr(
        extension.continuation,
        "verify_parent",
        lambda p: (
            {
                "run_dir": str(p.parent / "training"),
                "model_overrides": {"flow_lm.lookup_table.tokenizer_path": "fixed-tokenizer"},
            },
            {},
        ),
    )

    def pair(path: Path) -> tuple[Path, Path, int]:
        step = (
            200
            if path.parent == parent
            else 1000
            if path.parent == first
            else int(path.parent.name.split("-")[-1])
        )
        return path / "checkpoint.pt", path / "optim.pt", step

    monkeypatch.setattr(extension.continuation, "verify_pair", pair)
    monkeypatch.setattr(
        extension,
        "validate_first",
        lambda *_: (first / "training.yaml", first / "training/checkpoint.pt"),
    )
    monkeypatch.setattr(extension.continuation, "progress", lambda *_: [])

    def summary(
        evaluation: Path,
        records: list[dict[str, Any]],
        step: int,
        config: Path,
        checkpoint: Path | None,
        prompts: Path,
    ) -> dict[str, Any]:
        return {
            "step": step,
            "loss": extension.INITIAL_BASELINE_LOSS if step == 0 else 0.55,
            "finite": True,
            "similarities": [0.0] * 12,
            "healthy": 6,
            "capped": 2,
            "repetitive": 0,
            "config": str(config),
            "checkpoint": str(checkpoint),
            "evaluation": str(evaluation),
        }

    monkeypatch.setattr(extension.continuation, "summarize", summary)

    def evaluate(
        config: Path,
        checkpoint: Path,
        prompts: Path,
        output: Path,
        step: int,
        unused: argparse.Namespace,
    ) -> dict[str, Any]:
        return summary(output / "report.json", [], step, config, checkpoint, prompts)

    monkeypatch.setattr(extension, "evaluate", evaluate)

    def fork(parent_config: Path, child: Path, endpoint: int) -> Path:
        child.mkdir()
        (child / "training.yaml").write_text("unchanged recipe")
        return child / "training.yaml"

    monkeypatch.setattr(extension.continuation, "fork_run", fork)
    calls = []

    def command(argv: list[str], seconds: float, log: Path) -> int:
        calls.append(argv)
        assert 0 < seconds <= args.absolute_deadline_epoch - 1000
        if "export_selection" in argv[2]:
            request = json.loads(Path(argv[3]).read_text())
            (args.output / "selected.json").write_text(
                json.dumps(
                    {
                        "state": "complete",
                        "candidate_id": f"step-{request['selection']['step']}-raw",
                        "files": {},
                    }
                )
            )
        return 0

    monkeypatch.setattr(extension, "command", command)
    return args, calls


def test_explicit_owner_authorization_and_deadline_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls = setup_run(tmp_path, monkeypatch)
    args.owner_authorized_longer_training = False
    with pytest.raises(ValueError, match="owner-authorized"):
        extension.run(args)
    assert not args.output.exists() and not calls
    args.owner_authorized_longer_training = True
    args.absolute_deadline_epoch = 2700
    with pytest.raises(TimeoutError):
        extension.run(args)
    assert not args.output.exists() and not calls


def test_authorized_longer_run_has_no_old_generation_admission_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls = setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(extension.continuation, "gate", lambda *_: pytest.fail())
    result = extension.run(args)
    assert result["state"] == "complete" and result["quality_verified"] is False
    assert [c["step"] for c in result["candidates"]] == [200, 1000, 3000, 6000]
    assert [g["endpoint"] for g in result["gates"]] == [3000, 6000]
    assert sum("training.scripts.run_experiment" in c for c in calls) == 2
    for argv in calls:
        if "training.scripts.run_experiment" in argv:
            assert argv[argv.index("--wandb") + 1] == "offline"
    assert result["selected_step"] == 200  # Equal losses do not blindly choose the last update.


def test_divergent_stage_stops_without_another_training_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls = setup_run(tmp_path, monkeypatch)
    original = extension.evaluate

    def divergent(
        config: Path,
        checkpoint: Path,
        prompts: Path,
        output: Path,
        step: int,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        candidate = original(config, checkpoint, prompts, output, step, args)
        candidate["loss"] = 1.6 * extension.INITIAL_BASELINE_LOSS
        return candidate

    monkeypatch.setattr(extension, "evaluate", divergent)
    result = extension.run(args)
    assert result["state"] == "complete" and result["selected_step"] == 200
    assert "validation_diverged" in result["halt_reason"]
    assert not any("training.scripts.run_experiment" in c for c in calls)


@pytest.mark.parametrize(
    "failure",
    [
        InterruptedError(),
        subprocess.TimeoutExpired("runner", 10),
        KeyboardInterrupt(),
        SystemExit(),
    ],
)
def test_outer_cancellation_or_timeout_never_starts_export_or_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    args, calls = setup_run(tmp_path, monkeypatch)

    def interrupted(*_args: object) -> None:
        raise failure

    monkeypatch.setattr(extension, "evaluate", interrupted)
    result = extension.run(args)
    assert result["state"] == "failed"
    assert "cleanup_boundary" in result
    assert not calls


def test_live_first_service_wait_failure_cannot_trigger_fallback_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls = setup_run(tmp_path, monkeypatch)

    def fail_wait(*_args: object) -> None:
        raise TimeoutError("first service still running")

    monkeypatch.setattr(extension, "wait_first", fail_wait)
    result = extension.run(args)
    assert result["state"] == "failed"
    assert not calls and not (args.output / "selected.json").exists()


def test_supervised_numeric_timeout_preserves_prior_evaluated_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls = setup_run(tmp_path, monkeypatch)
    original = extension.command

    def timed_out(argv: list[str], seconds: float, log: Path) -> int:
        if "training.scripts.run_experiment" in argv:
            return 124
        return original(argv, seconds, log)

    monkeypatch.setattr(extension, "command", timed_out)
    result = extension.run(args)
    assert result["state"] == "complete"
    assert "partial_child_retained" in result["halt_reason"]
    assert [c["step"] for c in result["candidates"]] == [200, 1000]
    assert result["selected_step"] in (200, 1000)


def test_missing_diagnostics_stops_later_exploration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, calls = setup_run(tmp_path, monkeypatch)
    original = extension.evaluate

    def missing(
        config: Path,
        checkpoint: Path,
        prompts: Path,
        output: Path,
        step: int,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        candidate = original(config, checkpoint, prompts, output, step, args)
        candidate["similarities"] = []
        return candidate

    monkeypatch.setattr(extension, "evaluate", missing)
    result = extension.run(args)
    assert result["state"] == "failed"
    assert not any("training.scripts.run_experiment" in c for c in calls)


def test_wait_uses_user_unit_and_allows_missing_job_json_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = setup_run(tmp_path, monkeypatch)
    terminal = {
        "service": {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "Restart": "no",
            "Result": "success",
            "CGroupProcessesCount": "0",
            "ExecMainStartTimestampMonotonic": "1",
        },
        "job": None,
    }

    def poll(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert '["systemctl", "--user", "show",' in argv[2]
        return subprocess.CompletedProcess(argv, 0, json.dumps(terminal))

    monkeypatch.setattr(extension.subprocess, "run", poll)
    wait_first(args, {}, tmp_path / "poll.json")
    assert json.loads((tmp_path / "poll.json").read_text())["first_service_status"]["job"] is None


@pytest.mark.parametrize(
    "failure", [None, "missing", "running", "failed", "identity", "timestamps"]
)
def test_collected_unit_requires_latest_durable_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    args, _ = setup_run(tmp_path, monkeypatch)
    run = args.first_job / "training"
    attempts = run / "attempts"
    attempts.mkdir(parents=True)
    identity = {"config_sha256": "verified", "inputs": {}}
    monkeypatch.setattr(
        extension.continuation, "verify_parent", lambda _: ({"run_dir": str(run)}, identity)
    )
    record = {
        "state": "succeeded",
        "exit_code": 0,
        "child_exit_code": 0,
        "started_at": 100,
        "ended_at": 200,
        "identity": identity,
    }
    if failure != "missing":
        (attempts / "ffffffff.json").write_text(json.dumps(record))
    if failure in {"running", "failed"}:
        (attempts / "00000000.json").write_text(
            json.dumps(
                {**record, "started_at": 300, "ended_at": 400, "state": failure, "exit_code": 1}
            )
        )
    elif failure == "identity":
        record["identity"] = {"wrong": True}
        (attempts / "ffffffff.json").write_text(json.dumps(record))
    elif failure == "timestamps":
        record["ended_at"] = 50
        (attempts / "ffffffff.json").write_text(json.dumps(record))
    terminal = {
        "job": None,
        "service": {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "Restart": "no",
            "Result": "success",
            "MainPID": "0",
            "ControlPID": "0",
            "CGroupProcessesCount": "0",
            "InvocationID": "",
            "ExecMainStartTimestampMonotonic": "0",
        },
    }
    monkeypatch.setattr(
        extension.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a, 0, json.dumps(terminal)),
    )
    monkeypatch.setattr(extension.time, "sleep", lambda _: pytest.fail())
    report: dict[str, Any] = {}
    if failure:
        with pytest.raises(ValueError):
            wait_first(args, report, tmp_path / "poll.json")
    else:
        wait_first(args, report, tmp_path / "poll.json")
        assert report["first_completion_receipt"]["path"] == str(attempts / "ffffffff.json")
        assert report["first_completion_receipt"]["sha256"] == extension.digest(
            attempts / "ffffffff.json"
        )


@pytest.mark.parametrize(
    "mismatch", [None, "checkpoint_hash", "optimizer_path", "partial_step", "recipe"]
)
def test_first_stage_requires_exact1000_and_verified_original200_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str | None
) -> None:
    parent, first = tmp_path / "parent", tmp_path / "first"
    for directory in (parent, first):
        directory.mkdir()
        (directory / "training.yaml").write_text(directory.name)
        (directory / "checkpoint.pt").write_text(directory.name + "raw")
        (directory / "optim.pt").write_text(directory.name + "optimizer")
    parent_config = parent / "training.yaml"
    identity = {"verified": "source200"}
    lineage = {
        "parent_step": 200,
        "endpoint": 1000,
        "parent_identity": identity,
        "parent_config": {"sha256": extension.digest(parent_config)},
        "checkpoint": continuation_record(parent / "checkpoint.pt"),
        "optimizer": continuation_record(parent / "optim.pt"),
    }
    if mismatch == "checkpoint_hash":
        lineage["checkpoint"]["sha256"] = "bad"
    if mismatch == "optimizer_path":
        lineage["optimizer"]["path"] = str(first / "optim.pt")
    (first / "lineage.json").write_text(json.dumps(lineage))

    def config(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
        return {
            "run_dir": str(path.parent),
            "max_steps": 200 if path.parent == parent else 1000,
            "optim": {"lr": 1e-4 if mismatch == "recipe" and path.parent == first else 3e-5},
        }, identity

    def pair(path: Path) -> tuple[Path, Path, int]:
        return (
            path / "checkpoint.pt",
            path / "optim.pt",
            200 if path == parent else 999 if mismatch == "partial_step" else 1000,
        )

    monkeypatch.setattr(extension.continuation, "verify_parent", config)
    monkeypatch.setattr(extension.continuation, "verify_pair", pair)
    if mismatch:
        with pytest.raises(ValueError, match="chain|exactly1000|hyperparameters"):
            extension.validate_first(parent_config, first)
    else:
        assert extension.validate_first(parent_config, first) == (
            first / "training.yaml",
            first / "checkpoint.pt",
        )


def continuation_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": extension.digest(path)}


@pytest.mark.parametrize("publish_exit", [0, 1])
def test_publication_result_preserves_completed_local_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publish_exit: int
) -> None:
    args, calls = setup_run(tmp_path, monkeypatch)
    args.publish_repo = "owner/private-model"
    original = extension.command

    def command(argv: list[str], seconds: float, log: Path) -> int:
        if "training.scripts.publish_taigi_model" in argv:
            calls.append(argv)
            assert seconds <= 1200
            assert argv[argv.index("--repo-id") + 1] == args.publish_repo
            assert "--publish" in argv
            if publish_exit == 0:
                (args.output / "package.publication.json").write_text(
                    json.dumps({"private": True, "repo_id": args.publish_repo})
                )
            return publish_exit
        code = original(argv, seconds, log)
        if "export_selection" in argv[2]:
            selected = json.loads((args.output / "selected.json").read_text())
            directory = args.output / "selection"
            directory.mkdir()
            selected["files"] = {}
            for key in ("weights", "config", "tokenizer", "evaluation", "provenance"):
                artifact = directory / key
                artifact.write_text("explicit selected artifact")
                selected["files"][key] = continuation_record(artifact)
            (args.output / "selected.json").write_text(json.dumps(selected))
        return code

    monkeypatch.setattr(extension, "command", command)
    result = extension.run(args)
    assert result["state"] == "complete" and result["selected_step"] == 200
    assert result["publication"]["state"] == ("complete" if publish_exit == 0 else "failed")
    assert (args.output / "selected.json").is_file()
    assert any("training.scripts.publish_taigi_model" in argv for argv in calls)
