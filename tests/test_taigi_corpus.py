import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from training.taigi import corpus


def arguments(tmp_path: Path, entries: list[dict[str, Any]]) -> argparse.Namespace:
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"sources": [{"playlist_id": "first", "entries": entries}]}))
    return corpus.parser().parse_args(
        [
            "run",
            str(inventory),
            str(tmp_path / "out"),
            "--enhancer-python",
            "/usr/bin/python3",
            "--min-free-gib",
            "0",
        ]
    )


def fake_execute(calls: list[list[str]]) -> corpus.Executor:
    def execute(command: list[str], log: Path, deadline: float) -> None:
        calls.append(command)
        if "yt_dlp" in command:
            template = Path(command[command.index("-o") + 1])
            if "--skip-download" in command:
                template.with_name("source.info.json").write_text(
                    json.dumps({"id": command[-1][-11:], "duration": 3200, "license": "unknown"})
                )
            else:
                sf.write(template.with_name("source.wav"), np.zeros(1600), 16000)
                template.with_name("source.webm").write_bytes(b"media")
        else:
            spec = json.loads(Path(command[-1]).read_text())
            stage = Path(spec["output_dir"])
            stage.mkdir(parents=True, exist_ok=True)
            _, _, identity = corpus._context(spec)
            if command[-2] == "enhance":
                candidates = stage / "candidates" / "candidates.jsonl"
                candidates.parent.mkdir(exist_ok=True)
                candidates.write_text("")
                corpus.atomic_json(
                    stage / "enhancement-report.json",
                    dict(
                        state="complete",
                        **identity,
                        total_frames=1600,
                        covered_frames=1600,
                        sample_rate=16000,
                        artifacts={str(candidates): corpus.sha256_file(candidates)},
                    ),
                )
            else:
                corpus.atomic_json(
                    stage / "result.json",
                    dict(
                        state="no_candidates",
                        **identity,
                        release_path=None,
                        candidate_count=0,
                        artifacts={
                            str(stage / "enhancement-report.json"): corpus.sha256_file(
                                stage / "enhancement-report.json"
                            )
                        },
                    ),
                )

    return execute


def test_long_source_resume_and_integrity(tmp_path: Path) -> None:
    args = arguments(tmp_path, [{"id": "abcdefghijk", "duration": 3200}])
    calls = []
    assert corpus.run(args, execute_fn=fake_execute(calls))["counts"]["no_candidates"] == 1
    assert len(calls) == 4
    corpus.run(args, execute_fn=fake_execute(calls))
    assert len(calls) == 4
    (args.output / "videos/abcdefghijk/source.wav").write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity"):
        corpus.run(args, execute_fn=fake_execute(calls))
    assert len(calls) == 4


def test_all_inventory_membership_and_union(tmp_path: Path) -> None:
    entries = [{"id": f"v{i:010d}", "duration": None if i == 0 else 3200} for i in range(290)]
    args = arguments(tmp_path, entries)
    inventory = json.loads(args.inventory.read_text())
    inventory["sources"].append(
        {
            "playlist_id": "second",
            "entries": [entries[0], {"id": "other000001", "availability": "private"}],
        }
    )
    args.inventory.write_text(json.dumps(inventory))
    queue = corpus.normalize_inventory(args.inventory)
    assert {row["video_id"] for row in queue} == {row["id"] for row in entries} | {"other000001"}
    assert len(queue) == 291
    first = next(row for row in queue if row["video_id"] == entries[0]["id"])
    assert first["playlists"] == ["first", "second"]
    assert len({row["group_id"] for row in queue}) == 1
    assert first["duration"] is None


def test_disk_pause_keeps_every_item(tmp_path: Path) -> None:
    args = arguments(
        tmp_path,
        [{"id": "abcdefghijk", "duration": None}, {"id": "other000001", "availability": "private"}],
    )
    args.min_free_gib = 1e12
    report = corpus.run(args, execute_fn=fake_execute([]))
    assert report["total"] == 2
    assert report["counts"]["pending"] == 2
    assert report["reason"] == "disk_reserve"


def test_stage_resume_does_not_repeat_download_or_enhancement(tmp_path: Path) -> None:
    args = arguments(tmp_path, [{"id": "abcdefghijk", "duration": None}])
    calls = []

    def interrupt_transcription(command: list[str], log: Path, deadline: float) -> None:
        if command[-2] == "transcribe":
            raise corpus.Cancelled("pause")
        fake_execute(calls)(command, log, deadline)

    assert corpus.run(args, execute_fn=interrupt_transcription)["reason"] == "cancelled"
    assert len(calls) == 3
    report = corpus.run(args, execute_fn=fake_execute(calls))
    assert report["counts"]["no_candidates"] == 1
    assert len(calls) == 4
    assert calls[-1][-2] == "transcribe"


def test_unavailable_failures_are_bounded_and_retained(tmp_path: Path) -> None:
    args = arguments(tmp_path, [{"id": "abcdefghijk", "availability": "private"}, {"id": None}])
    calls = []

    def unavailable(command: list[str], log: Path, deadline: float) -> None:
        calls.append(command)
        raise RuntimeError("video unavailable")

    result = corpus.run(args, execute_fn=unavailable)
    assert result["total"] == 2
    assert result["counts"]["failed"] == 2
    assert len(calls) == args.max_attempts
    corpus.run(args, execute_fn=unavailable)
    assert len(calls) == args.max_attempts


def test_raw_promotion_recovers_without_download(tmp_path: Path) -> None:
    args = arguments(tmp_path, [{"id": "abcdefghijk"}])
    calls = []

    def interrupt_enhancement(command: list[str], log: Path, deadline: float) -> None:
        if command[-2] == "enhance":
            raise corpus.Cancelled("pause")
        fake_execute(calls)(command, log, deadline)

    corpus.run(args, execute_fn=interrupt_enhancement)
    state_path = args.output / "state.json"
    state = json.loads(state_path.read_text())
    entry = state["videos"]["abcdefghijk"]
    entry["stages"]["acquire"]["state"] = "interrupted"
    state_path.write_text(json.dumps(state))
    (args.output / "videos/abcdefghijk/source-report.json").unlink()
    result = corpus.run(args, execute_fn=fake_execute(calls))
    assert result["counts"]["no_candidates"] == 1
    assert len(calls) == 4


def test_completed_enhancement_corruption_refuses_resume(tmp_path: Path) -> None:
    args = arguments(tmp_path, [{"id": "abcdefghijk"}])
    calls = []

    def interrupt_transcription(command: list[str], log: Path, deadline: float) -> None:
        if command[-2] == "transcribe":
            raise corpus.Cancelled("pause")
        fake_execute(calls)(command, log, deadline)

    corpus.run(args, execute_fn=interrupt_transcription)
    (args.output / "videos/abcdefghijk/stage/candidates/candidates.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="integrity"):
        corpus.run(args, execute_fn=fake_execute(calls))
    assert len(calls) == 3


def test_deadline_and_plan_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = arguments(tmp_path, [{"id": "abcdefghijk"}])
    ticks = iter([0, args.max_seconds + 1])
    with monkeypatch.context() as patch:
        patch.setattr(corpus.time, "monotonic", lambda: next(ticks))
        result = corpus.run(args, execute_fn=fake_execute([]))
    assert result["reason"] == "deadline"
    assert result["counts"] == {"pending": 1}
    args.device = "cpu"
    with pytest.raises(ValueError, match="plan"):
        corpus.run(args, execute_fn=fake_execute([]))


def test_lock_refuses_second_writer(tmp_path: Path) -> None:
    args = arguments(tmp_path, [{"id": "abcdefghijk"}])
    args.output.mkdir()
    with (args.output / ".lock").open("a") as lock:
        corpus.fcntl.flock(lock, corpus.fcntl.LOCK_EX | corpus.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            corpus.run(args, execute_fn=fake_execute([]))


def test_changed_code_refuses_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = arguments(tmp_path, [{"id": "abcdefghijk"}])
    calls = []
    corpus.run(args, execute_fn=fake_execute(calls))
    original_hash = corpus.sha256_file
    monkeypatch.setattr(
        corpus,
        "sha256_file",
        lambda path: "changed" if Path(path).name == "corpus.py" else original_hash(path),
    )
    with pytest.raises(ValueError, match="plan"):
        corpus.run(args, execute_fn=fake_execute(calls))
    assert len(calls) == 4


def test_recovered_worker_report_rejects_wrong_group(tmp_path: Path) -> None:
    args = arguments(tmp_path, [{"id": "abcdefghijk"}])
    corpus.run(args, execute_fn=fake_execute([]))
    video_dir = args.output / "videos/abcdefghijk"
    report_path = video_dir / "stage/enhancement-report.json"
    report = json.loads(report_path.read_text())
    report["group_id"] = "playlist:wrong"
    corpus.atomic_json(report_path, report)
    with pytest.raises(ValueError, match="identity"):
        corpus.worker_artifacts(video_dir, "enhance")


def test_code_change_during_run_prevents_next_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = arguments(tmp_path, [{"id": "abcdefghijk"}])
    calls = []
    original_hash = corpus.sha256_file

    def change_after_download(command: list[str], log: Path, deadline: float) -> None:
        fake_execute(calls)(command, log, deadline)
        if "--keep-video" in command:
            monkeypatch.setattr(
                corpus,
                "sha256_file",
                lambda path: "changed" if Path(path).name == "corpus.py" else original_hash(path),
            )

    with pytest.raises(ValueError, match="plan code changed"):
        corpus.run(args, execute_fn=change_after_download)
    assert len(calls) == 2
    assert corpus.status(args.output)["reason"] == "integrity_failure"


def test_inventory_round_robin_preserves_all_memberships(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.json"
    sources: list[dict[str, Any]] = [
        {
            "playlist_id": f"show-{group}",
            "entries": [{"id": f"g{group}e{episode:08d}"} for episode in range(3)],
        }
        for group in range(6)
    ]
    # Shared membership must neither duplicate a video nor consume a scheduling turn.
    sources[1]["entries"].insert(0, sources[0]["entries"][0])
    inventory.write_text(json.dumps({"sources": sources}))
    queue = corpus.normalize_inventory(inventory)
    expected = [f"g{group}e{episode:08d}" for episode in range(3) for group in range(6)]
    assert [row["video_id"] for row in queue] == expected
    assert {row["playlists"][0] for row in queue[:6]} == {f"show-{i}" for i in range(6)}
    assert queue[0]["playlists"] == ["show-0", "show-1"]
    assert queue[0]["group_id"] == queue[1]["group_id"]
    assert corpus.normalize_inventory(inventory) == queue
