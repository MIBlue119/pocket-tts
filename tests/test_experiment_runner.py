import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import yaml

from training.scripts import run_experiment as runner


def setup_run(tmp_path: Path) -> tuple[argparse.Namespace, Path, Path]:
    manifest = tmp_path / "train.jsonl"
    manifest.write_text('{"text": "private transcript"}\n')
    config = tmp_path / "config.yaml"
    run_dir = tmp_path / "run"
    config.write_text(
        yaml.safe_dump(
            {
                "run_dir": str(run_dir),
                "data": {
                    "train_jsonl": str(manifest),
                    "precompute": False,
                    "sampling_mode": "sentence",
                },
            }
        )
    )
    args = argparse.Namespace(
        config=str(config),
        max_seconds=3,
        grace_seconds=0.2,
        wandb="disabled",
        entity="weirenlan",
        project="pocket-tts-taigi",
        provenance=[],
    )
    return args, run_dir, manifest


def child(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def attempt(run_dir: Path) -> dict:
    return json.loads(next((run_dir / "attempts").glob("*.json")).read_text())


def test_success_metrics_and_provenance(tmp_path: Path) -> None:
    args, run_dir, manifest = setup_run(tmp_path)
    events = []

    class Events:
        def put_nowait(self: "Events", event: dict) -> None:
            events.append(event)

    run_dir.mkdir()
    code = (
        f"from pathlib import Path; import json; "
        f"p=Path({str(run_dir / 'progress.jsonl')!r}); "
        "p.write_text(json.dumps({'type':'train','step':1,'metrics':"
        "{'loss':2.0,'transcript':'secret','unknown':3}})+'\\n')"
    )
    assert runner.supervise(child(code), run_dir, 3, 0.2, Events()) == (0, "completed")
    assert events == [{"train/loss": 2.0, "trainer_step": 1}]
    (run_dir / "progress.jsonl").unlink()
    assert runner.run(args, child("raise SystemExit(0)")) == 0
    result = attempt(run_dir)
    assert result["state"] == "succeeded"
    assert len(result["git_sha"]) == 40
    assert result["identity"]["inputs"]["train_jsonl"]["sha256"] == runner.digest(manifest)
    assert result["ended_at"] >= result["started_at"]


def test_child_exit_propagates(tmp_path: Path) -> None:
    args, run_dir, _ = setup_run(tmp_path)
    assert runner.run(args, child("raise SystemExit(7)")) == 7
    assert attempt(run_dir)["state"] == "failed"


def test_timeout_graceful_shutdown(tmp_path: Path) -> None:
    args, run_dir, _ = setup_run(tmp_path)
    args.max_seconds = 0.4
    marker = tmp_path / "terminated"
    script = (
        "import signal,time; from pathlib import Path; "
        f"signal.signal(signal.SIGTERM, lambda *a: (Path({str(marker)!r}).write_text('yes'), exit(0))); "
        "time.sleep(20)"
    )
    assert runner.run(args, child(script)) == 124
    assert marker.read_text() == "yes"
    assert attempt(run_dir)["state"] == "timeout"
    assert attempt(run_dir)["child_exit_code"] == 0


def test_timeout_kills_stubborn_child_and_group(tmp_path: Path) -> None:
    args, run_dir, _ = setup_run(tmp_path)
    args.max_seconds = 0.5
    pidfile = tmp_path / "pid"
    script = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pid=os.fork(); "
        f"Path({str(pidfile)!r}).write_text(str(os.getpid())) if pid == 0 else None; "
        "time.sleep(20)"
    )
    start = time.monotonic()
    assert runner.run(args, child(script)) == 124
    assert time.monotonic() - start < 4
    assert attempt(run_dir)["child_exit_code"] == -signal.SIGKILL
    pid = int(pidfile.read_text())
    stat = Path(f"/proc/{pid}/stat")
    assert not stat.exists() or stat.read_text().split()[2] == "Z"


@pytest.mark.parametrize("change", ["config", "data", "provenance"])
def test_reject_changed_identity(tmp_path: Path, change: str) -> None:
    args, run_dir, manifest = setup_run(tmp_path)
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}")
    args.provenance = [str(provenance)]
    assert runner.run(args, child("pass")) == 0
    target = {"config": Path(args.config), "data": manifest, "provenance": provenance}[change]
    with target.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="resume rejected"):
        runner.run(args, child("pass"))


def test_reject_missing_latest_optimizer_and_snapshot(tmp_path: Path) -> None:
    args, run_dir, _ = setup_run(tmp_path)
    assert runner.run(args, child("pass")) == 0
    (run_dir / "checkpoint_00000001.pt").write_bytes(b"model")
    (run_dir / "optim_00000001.pt").write_bytes(b"optimizer")
    (run_dir / "checkpoint_00000002.pt").write_bytes(b"model2")
    with pytest.raises(ValueError, match="optimizer"):
        runner.run(args, child("pass"))
    (run_dir / "optim_00000002.pt").write_bytes(b"optimizer2")
    assert runner.run(args, child("pass")) == 0
    destination = tmp_path / "snapshot"
    runner.snapshot(run_dir, destination)
    assert (destination / "READY").exists()
    checksums = json.loads((destination / "checksums.json").read_text())
    assert "checkpoint_00000002.pt" in checksums
    assert "optim_00000002.pt" in checksums
    assert all(runner.digest(destination / name) == value for name, value in checksums.items())


def test_lock_excludes_concurrent_launcher(tmp_path: Path) -> None:
    args, run_dir, _ = setup_run(tmp_path)
    with runner.run_lock(run_dir):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "training.scripts.run_experiment",
                "run",
                args.config,
                "--max-seconds",
                "1",
            ],
            cwd=runner.ROOT,
            capture_output=True,
            text=True,
        )
    assert result.returncode == 2
    assert "already locked" in result.stderr
    assert not (run_dir / "attempts").exists()


def test_signal_to_launcher_stops_child(tmp_path: Path) -> None:
    args, run_dir, _ = setup_run(tmp_path)
    run_dir.mkdir()
    marker = tmp_path / "ready"
    code = (
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, lambda *a: exit(0)); "
        f"Path({str(marker)!r}).touch(); time.sleep(20)"
    )
    wrapper = (
        "from pathlib import Path; from training.scripts.run_experiment import supervise; "
        f"print(supervise({child(code)!r}, Path({str(run_dir)!r}), 20, .2), flush=True)"
    )
    process = subprocess.Popen(child(wrapper), cwd=runner.ROOT, stdout=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 3
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()
        os.kill(process.pid, signal.SIGTERM)
        output, _ = process.communicate(timeout=3)
        assert "(0, 'interrupted')" in output
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_optional_wandb_failure_does_not_fail_training(tmp_path: Path) -> None:
    args, run_dir, _ = setup_run(tmp_path)
    args.wandb = "offline"
    assert runner.run(args, child("pass")) == 0
    assert attempt(run_dir)["state"] == "succeeded"


def test_wandb_payload_is_numeric_and_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import queue
    import types

    captured = {}
    events = queue.Queue()
    events.put({"train/loss": 0.5, "trainer_step": 2})
    events.put(None)

    class Run:
        def log(self: "Run", value: dict) -> None:
            captured["metrics"] = value

        def finish(self: "Run") -> None:
            captured["finished"] = True

    def init(**kwargs: object) -> Run:
        captured.update(kwargs)
        return Run()

    fake = types.SimpleNamespace(init=init, Settings=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    runner.wandb_worker(
        events,
        {
            "entity": "weirenlan",
            "project": "pocket-tts-taigi",
            "mode": "offline",
            "attempt": "unique-attempt",
            "directory": str(tmp_path),
        },
        {"config_sha256": "abc", "git_sha": "def"},
    )
    assert captured["config"] == {"config_sha256": "abc", "git_sha": "def"}
    assert captured["settings"]["x_disable_meta"]
    assert captured["settings"]["x_disable_stats"]
    assert captured["settings"]["disable_code"]
    assert captured["settings"]["console"] == "off"
    assert captured["resume"] == "never"
    assert captured["finished"]


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 0, -1])
def test_reject_unbounded_duration(tmp_path: Path, value: float) -> None:
    args, _, _ = setup_run(tmp_path)
    args.max_seconds = value
    with pytest.raises(ValueError, match="max-seconds"):
        runner.run(args, child("pass"))
