import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import BinaryIO, cast

import pytest

from training.scripts import watch_taigi_cloud_job as watch


def arguments(tmp_path: Path) -> argparse.Namespace:
    return watch.parser().parse_args(
        [
            "--host",
            "tester@cloud",
            "--remote-output",
            "/outputs/full job",
            "--local-output",
            str(tmp_path / "mirror"),
            "--remote-job-json",
            "/outputs/job/job.json",
            "--remote-systemd-unit",
            "taigi.service",
            "--absolute-deadline-epoch",
            str(time.time() + 1000),
            "--once",
        ]
    )


def terminal() -> dict[str, object]:
    return {
        "service": {
            "ActiveState": "inactive",
            "SubState": "dead",
            "LoadState": "loaded",
            "Restart": "no",
            "Result": "success",
            "InvocationID": "known-run",
            "CGroupProcessesCount": "0",
            "ExecMainStartTimestampMonotonic": "100",
            "ExecMainExitTimestampMonotonic": "200",
        },
        "job": {"state": "complete"},
    }


def test_verified_final_backup_invokes_only_explicit_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = arguments(tmp_path)
    helper = tmp_path / "stop-helper"
    helper.write_text("#!/bin/sh\nexit 0\n")
    helper.chmod(0o700)
    args.stop_command = helper
    calls = []
    payload = b"checkpoint bytes"

    def execute(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        assert cast(float, kwargs["timeout"]) > 0
        if argv[0] == "ssh" and watch.HASH_SCRIPT in argv[-1]:
            cast(BinaryIO, kwargs["stdout"]).write(
                (
                    json.dumps(
                        {
                            "path": "checkpoint.pt",
                            "size": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    )
                    + "\n"
                ).encode()
            )
            return subprocess.CompletedProcess(argv, 0)
        if argv[0] == "ssh":
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(terminal()))
        if argv[0] == "rsync":
            (args.local_output / "checkpoint.pt").write_bytes(payload)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(watch.subprocess, "run", execute)
    receipt = watch.run(args)
    assert receipt["backup"] == "verified"
    assert receipt["verified_files"] == 1
    assert calls[-1][0] == [str(helper.resolve()), "--stop-now"]
    rsync = next(argv for argv, _ in calls if argv[0] == "rsync")
    assert "--partial" in rsync and "--delete" not in rsync
    assert "tester@cloud:/outputs/full job/" in rsync
    saved = json.loads((tmp_path / "mirror.watch.json").read_text())
    assert saved["stop"] == "succeeded"


def test_transient_ssh_error_never_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = arguments(tmp_path)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired("ssh", 1)

    monkeypatch.setattr(watch.subprocess, "run", fail)
    receipt = watch.run(args)
    assert receipt["state"] == "watching"
    assert receipt["backup"] == "incomplete"
    assert "stop" not in receipt


def test_deadline_stops_even_without_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = arguments(tmp_path)
    args.absolute_deadline_epoch = time.time() - 1
    helper = tmp_path / "stop-helper"
    helper.write_text("helper")
    helper.chmod(0o700)
    args.stop_command = helper
    calls = []

    def execute(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(watch.subprocess, "run", execute)
    receipt = watch.run(args)
    assert receipt["state"] == "deadline"
    assert receipt["backup"] == "incomplete"
    assert calls == [[str(helper), "--stop-now"]]
    assert args.local_output.is_dir()


def test_hash_mismatch_does_not_stop_before_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = arguments(tmp_path)
    monkeypatch.setattr(watch, "poll", lambda _: terminal())
    monkeypatch.setattr(watch, "mirror", lambda _: None)

    def hashes(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        cast(BinaryIO, kwargs["stdout"]).write(b'{"path":"missing.pt","size":3,"sha256":"wrong"}\n')
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(watch.subprocess, "run", hashes)
    receipt = watch.run(args)
    assert receipt["backup"] == "incomplete"
    assert receipt["state"] == "watching"
    assert "missing or size mismatch" in receipt["last_error"]
    assert "stop" not in receipt


def test_operation_timeout_reserves_absolute_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = arguments(tmp_path)
    args.absolute_deadline_epoch = 100
    monkeypatch.setattr(watch.time, "time", lambda: 97)
    assert watch.time_left(args) == 3
    monkeypatch.setattr(watch.time, "time", lambda: 100)
    with pytest.raises(TimeoutError):
        watch.time_left(args)


def test_running_service_cannot_trigger_final_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = arguments(tmp_path)
    monkeypatch.setattr(
        watch,
        "poll",
        lambda _: {
            "service": {"ActiveState": "active", "SubState": "running"},
            "job": {"state": "complete"},
        },
    )
    monkeypatch.setattr(watch, "mirror", lambda _: None)
    monkeypatch.setattr(watch, "verify", lambda *_: pytest.fail())
    receipt = watch.run(args)
    assert receipt["backup"] == "incomplete"
    assert "stop" not in receipt


def test_real_streaming_hash_checks_local_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    args = arguments(tmp_path)
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "nested").mkdir()
    (remote / "nested" / "weights.pt").write_bytes(b"frozen checkpoint")
    (remote / "metadata.json").write_text('{"rows": 12}')
    (remote / "link").symlink_to(remote / "metadata.json")
    args.local_output.mkdir()
    (args.local_output / "nested").mkdir()
    (args.local_output / "nested" / "weights.pt").write_bytes(b"frozen checkpoint")
    (args.local_output / "metadata.json").write_text('{"rows": 12}')
    monkeypatch.setattr(
        watch, "ssh", lambda *_: [sys.executable, "-c", watch.HASH_SCRIPT, str(remote)]
    )
    manifest = tmp_path / "checksums.jsonl"
    assert watch.verify(args, manifest) == 2
    (args.local_output / "nested" / "weights.pt").write_bytes(b"broken checkpoint")
    with pytest.raises(ValueError, match="checksum mismatch"):
        watch.verify(args, manifest)


@pytest.mark.parametrize("active,job_state", [("failed", "running"), ("inactive", None)])
def test_ended_invocation_with_stale_job_stops_after_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, active: str, job_state: str | None
) -> None:
    args = arguments(tmp_path)
    status = terminal()
    service = cast(dict[str, str], status["service"])
    service.update(ActiveState=active, Result="oom-kill")
    status["job"] = {"state": job_state} if job_state else None
    monkeypatch.setattr(watch, "poll", lambda _: status)
    monkeypatch.setattr(watch, "mirror", lambda _: None)
    monkeypatch.setattr(watch, "verify", lambda *_: 3)
    receipt = watch.run(args)
    assert receipt["backup"] == "verified"
    assert receipt["job_outcome"] == "interrupted"
    assert receipt["job_incomplete"] is True
    assert receipt["stop"] == "stop_not_configured"


@pytest.mark.parametrize(
    "load,start,restart",
    [("not-found", "0", "no"), ("loaded", "0", "no"), ("loaded", "100", "always")],
)
def test_unstarted_missing_or_restarting_unit_cannot_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load: str, start: str, restart: str
) -> None:
    args = arguments(tmp_path)
    status = terminal()
    service = cast(dict[str, str], status["service"])
    service.update(
        LoadState=load, ExecMainStartTimestampMonotonic=start, InvocationID="", Restart=restart
    )
    monkeypatch.setattr(watch, "poll", lambda _: status)
    monkeypatch.setattr(watch, "mirror", lambda _: None)
    monkeypatch.setattr(watch, "verify", lambda *_: pytest.fail())
    receipt = watch.run(args)
    assert receipt["backup"] == "incomplete"
    assert "stop" not in receipt


def test_lingering_cgroup_writer_prevents_final_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = arguments(tmp_path)
    status = terminal()
    cast(dict[str, str], status["service"])["CGroupProcessesCount"] = "1"
    monkeypatch.setattr(watch, "poll", lambda _: status)
    monkeypatch.setattr(watch, "mirror", lambda _: None)
    monkeypatch.setattr(watch, "verify", lambda *_: pytest.fail())
    receipt = watch.run(args)
    assert receipt["backup"] == "incomplete"
    assert "stop" not in receipt


def test_changed_invocation_during_mirror_prevents_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = arguments(tmp_path)
    before, after = terminal(), terminal()
    cast(dict[str, str], after["service"])["InvocationID"] = "different-run"
    statuses = iter([before, after])
    monkeypatch.setattr(watch, "poll", lambda _: next(statuses))
    monkeypatch.setattr(watch, "mirror", lambda _: None)
    monkeypatch.setattr(watch, "verify", lambda *_: pytest.fail())
    receipt = watch.run(args)
    assert receipt["backup"] == "incomplete"
    assert "invocation changed" in receipt["last_error"]
    assert "stop" not in receipt
