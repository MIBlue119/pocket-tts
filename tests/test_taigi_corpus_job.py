import json
from pathlib import Path

import pytest

from training.scripts import run_taigi_corpus_job as job


def corpus_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, states: list[str]
) -> tuple[Path, Path]:
    from training.taigi import corpus

    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text("frozen")
    output = tmp_path / "corpus"
    output.mkdir()
    queue = [{"key": str(i)} for i in range(len(states))]
    monkeypatch.setattr(corpus, "normalize_inventory", lambda _: queue)
    monkeypatch.setattr(corpus, "worker_artifacts", lambda *_: {})
    (output / "plan.json").write_text(
        json.dumps({"inventory_sha256": job.digest(inventory), "queue": queue, "config": {}})
    )
    (output / "state.json").write_text(
        json.dumps({"videos": {str(i): {"state": state} for i, state in enumerate(states)}})
    )
    for i, state in enumerate(states):
        if state == "complete":
            stage = output / "videos" / str(i) / "stage"
            stage.mkdir(parents=True)
            (stage / "result.json").write_text(
                json.dumps({"state": state, "release_path": str(tmp_path / "release")})
            )
    return inventory, output


@pytest.mark.parametrize("state", ["pending", "running", "interrupted"])
def test_partial_queue_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    inventory, output = corpus_fixture(tmp_path, monkeypatch, ["complete", state])
    with pytest.raises(ValueError, match="partial corpus"):
        job.completed_releases(inventory, output)


def test_failed_unavailable_video_remains_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory, output = corpus_fixture(
        tmp_path, monkeypatch, ["complete", "failed", "no_candidates"]
    )
    releases, counts = job.completed_releases(inventory, output)
    assert len(releases) == 1
    assert counts == {"complete": 1, "failed": 1, "no_candidates": 1}


def test_deadline_reserves_backup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(job.time, "monotonic", lambda: 50)
    assert job.remaining(100, 20) == 30
    with pytest.raises(TimeoutError):
        job.remaining(100, 50)


def test_failed_stage_is_not_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inventory, output = corpus_fixture(tmp_path, monkeypatch, ["complete"])
    args = job.parser().parse_args([str(inventory), str(output), str(tmp_path / "job")])
    monkeypatch.setenv("INVOCATION_ID", "test")
    monkeypatch.setattr(job, "command", lambda *_: 2)
    with pytest.raises(RuntimeError, match="corpus subprocess failed"):
        job.run(args)
    report = json.loads((args.job / "job.json").read_text())
    assert report["state"] == "failed"
    assert report["stage"] == "corpus"


def test_tokenizer_only_receives_train_and_recipe_is_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from training.taigi import merge

    inventory, output = corpus_fixture(tmp_path, monkeypatch, ["complete", "failed"])
    release = tmp_path / "release"
    release.mkdir()
    (release / "train.jsonl").write_text('{"transcript": "訓練"}\n')
    (release / "valid.jsonl").write_text('{"transcript": "保留"}\n')
    monkeypatch.setattr(merge, "merge_releases", lambda *_args, **_kwargs: release)
    monkeypatch.setenv("INVOCATION_ID", "test")
    args = job.parser().parse_args([str(inventory), str(output), str(tmp_path / "job")])
    calls = []

    def execute(argv: list[str], seconds: float, log: Path) -> int:
        calls.append((argv, seconds))
        assert seconds > 0
        if "training.taigi.corpus" in argv:
            return 1
        if "training.scripts.convert_tokenizer" in argv:
            (args.job / "tokenizer.json").write_text(
                json.dumps({"model": {"vocab": [["a", 0], ["b", 0]]}})
            )
        if "training.scripts.run_experiment" in argv:
            training = args.job / "training"
            training.mkdir()
            (training / "checkpoint_000100.pt").write_text("raw")
            (training / "optim_000100.pt").write_text("optimizer")
            return 124
        return 0

    monkeypatch.setattr(job, "command", execute)
    report = job.run(args)
    tokenizer_argv = next(argv for argv, _ in calls if "training.scripts.train_tokenizer" in argv)
    assert str(release / "train.jsonl") in tokenizer_argv
    assert str(release / "valid.jsonl") not in tokenizer_argv
    assert tokenizer_argv[-1] == "4096"
    assert report["tokenizer"]["actual_vocab"] == 2
    assert report["state"] == "complete"
    assert report["video_outcomes"]["failed"] == 1
    config = job.yaml.safe_load((args.job / "training.yaml").read_text())
    assert config["reset_text_embedding"] is True
    assert config["max_steps"] == 2000
    assert config["num_valid_batches"] == 1
    assert config["sample_sentences"] == []
    assert config["grad_accum_steps"] == 8
    assert len(job.rows(args.job / "prompts.jsonl")) == 12
    training_argv, limit = next(
        call for call in calls if "training.scripts.run_experiment" in call[0]
    )
    assert float(training_argv[training_argv.index("--max-seconds") + 1]) <= args.training_seconds
    assert limit <= args.training_seconds + 60
    with pytest.raises(FileExistsError):
        job.run(args)
