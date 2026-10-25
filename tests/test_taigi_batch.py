import argparse
import json
from pathlib import Path
from collections.abc import Callable

import pytest

from training.taigi import batch


def setup(tmp_path: Path) -> argparse.Namespace:
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(
        json.dumps(dict(url="https://youtu.be/abcdefghijk", group_id="one", start=0, duration=20))
        + "\n"
    )
    args = batch.parser().parse_args(
        [
            str(jobs),
            str(tmp_path / "out"),
            "--enhancer-python",
            "/usr/bin/python3",
            "--min-free-gib",
            "0",
        ]
    )
    return args


def worker(calls: list[list[str]]) -> Callable:
    def execute(command: list[str], log: Path, deadline: float) -> None:
        calls.append(command)
        out = Path(command[4])
        release = out / "releases" / "release"
        release.mkdir(parents=True)
        audio = release / "audio.wav"
        audio.write_bytes(b"audio")
        row = dict(
            published_audio_path=str(audio),
            audio_sha256=batch.sha256_file(audio),
            start=0,
            duration=1,
            input_text_zh="text",
            split="train",
        )
        (release / "metadata.jsonl").write_text(json.dumps(row) + "\n")
        (release / "train.jsonl").write_text(
            json.dumps(dict(path=str(audio), start=0, duration=1, transcript="text")) + "\n"
        )
        (release / "valid.jsonl").write_text("")
        (release / "quarantine.jsonl").write_text("")
        (release / "report.json").write_text(json.dumps(dict(accepted_weak_label_count=1)))
        (out / "run-report.json").write_text(
            json.dumps(dict(state="weak_labels_ready", release=str(release)))
        )

    return execute


def test_resume_and_corruption(tmp_path: Path) -> None:
    args = setup(tmp_path)
    calls = []
    assert batch.run_batch(args, execute_fn=worker(calls))["state"] == "complete"
    assert batch.run_batch(args, execute_fn=worker(calls))["state"] == "complete"
    assert len(calls) == 1
    next(args.output.glob("attempts/*/*/releases/release/audio.wav")).write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity|hash"):
        batch.run_batch(args, execute_fn=worker(calls))
    assert len(calls) == 1


def test_changed_plan_and_conflict(tmp_path: Path) -> None:
    args = setup(tmp_path)
    batch.run_batch(args, execute_fn=worker([]))
    args.device = "cpu"
    with pytest.raises(ValueError, match="plan"):
        batch.run_batch(args, execute_fn=worker([]))
    row = json.loads(args.jobs.read_text())
    args.jobs.write_text(args.jobs.read_text() + json.dumps({**row, "group_id": "other"}) + "\n")
    with pytest.raises(ValueError, match="conflicting"):
        batch.run_batch(args, execute_fn=worker([]))


def test_attempt_cap(tmp_path: Path) -> None:
    args = setup(tmp_path)
    calls = []

    def fail(*unused: object) -> None:
        calls.append(1)
        raise RuntimeError("failed worker")

    assert batch.run_batch(args, execute_fn=fail)["state"] == "incomplete"
    assert batch.run_batch(args, execute_fn=fail)["state"] == "incomplete"
    assert len(calls) == args.max_attempts


def test_disk_and_deadline_admission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = setup(tmp_path)
    args.min_free_gib = 1e12
    assert batch.run_batch(args, execute_fn=worker([]))["reason"] == "disk_reserve"
    args.output = tmp_path / "deadline"
    args.min_free_gib = 0
    ticks = iter([0, 3601])
    monkeypatch.setattr(batch.time, "monotonic", lambda: next(ticks))
    assert batch.run_batch(args, execute_fn=worker([]))["reason"] == "deadline"


def test_real_wrapper_timeout_cleans_stage(tmp_path: Path) -> None:
    import sys
    import time

    child_pid = tmp_path / "child.pid"
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text("""import os, signal, subprocess, sys, time
from pathlib import Path
from training.taigi.build_video import execute

def cancel(signum, frame):
    raise SystemExit(128+signum)
signal.signal(signal.SIGTERM, cancel)
execute([sys.executable, '-c', "import os,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)".replace('import os,time;', 'import os,time,sys;'), sys.argv[1]], Path(sys.argv[2]), time.monotonic()+60)
""")
    with pytest.raises(batch.subprocess.TimeoutExpired):
        batch.execute(
            [sys.executable, str(wrapper), str(child_pid), str(tmp_path / "stage.log")],
            tmp_path / "wrapper.log",
            time.monotonic() + 2,
        )
    assert child_pid.exists(), (tmp_path / "wrapper.log").read_text()
    assert batch.process_identity(int(child_pid.read_text())) is None
    record = json.loads((tmp_path / "wrapper.pid.json").read_text())
    assert batch.process_identity(record["pid"]) is None


def test_enhancer_python_preserves_venv_path(tmp_path: Path) -> None:
    import subprocess
    import sys

    args = setup(tmp_path)
    venv = tmp_path / "enhancer"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    args.enhancer_python = venv / "bin" / "python"
    calls = []
    batch.run_batch(args, execute_fn=worker(calls))
    command = calls[0]
    interpreter = command[command.index("--enhancer-python") + 1]
    prefix = subprocess.check_output(
        [interpreter, "-c", "import sys; print(sys.prefix)"], text=True
    ).strip()
    assert prefix == str(venv)


def test_lock_refuses_second_writer(tmp_path: Path) -> None:
    args = setup(tmp_path)
    args.output.mkdir()
    with (args.output / ".lock").open("a") as lock:
        batch.fcntl.flock(lock, batch.fcntl.LOCK_EX | batch.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            batch.run_batch(args, execute_fn=worker([]))


def test_real_signal_persists_interrupted_attempt(tmp_path: Path) -> None:
    import os
    import signal
    import subprocess
    import sys
    import time

    args = setup(tmp_path)
    child_pid = tmp_path / "child.pid"
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text("""import signal, sys, time
from pathlib import Path
from training.taigi.build_video import execute

def cancel(signum, frame):
    raise SystemExit(128+signum)
signal.signal(signal.SIGTERM, cancel)
execute([sys.executable, '-c', "import os,time,sys; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)", sys.argv[1]], Path(sys.argv[2]), time.monotonic()+60)
""")
    runner = tmp_path / "runner.py"
    runner.write_text(
        """import sys
from training.taigi import batch
args = batch.parser().parse_args(sys.argv[1:4] + ['--enhancer-python', sys.executable, '--min-free-gib', '0'])
def execute(command, log, deadline):
    batch.execute([sys.executable, sys.argv[4], sys.argv[5], sys.argv[6]], log, deadline)
batch.run_batch(args, execute_fn=execute)
""".replace("sys.argv[1:4]", "sys.argv[1:3]").replace(
            "sys.argv[4], sys.argv[5], sys.argv[6]", "sys.argv[3], sys.argv[4], sys.argv[5]"
        )
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(runner),
            str(args.jobs),
            str(args.output),
            str(wrapper),
            str(child_pid),
            str(tmp_path / "stage.log"),
        ],
        env={**os.environ, "PYTHONPATH": str(batch.ROOT)},
    )
    try:
        deadline = time.monotonic() + 10
        while not child_pid.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert child_pid.exists()
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=10)
        assert batch.process_identity(int(child_pid.read_text())) is None
        state = json.loads((args.output / "state.json").read_text())
        assert next(iter(state["jobs"].values()))["attempts"][0]["state"] == "interrupted"
        assert json.loads((args.output / "batch-report.json").read_text())["reason"] == "cancelled"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_stale_attempt_is_preserved_and_retried(tmp_path: Path) -> None:
    args = setup(tmp_path)

    def fail(*unused: object) -> None:
        raise batch.Cancelled("interrupted")

    batch.run_batch(args, execute_fn=fail)
    path = args.output / "state.json"
    state = json.loads(path.read_text())
    entry = next(iter(state["jobs"].values()))
    entry["state"] = entry["attempts"][0]["state"] = "running"
    path.write_text(json.dumps(state))
    calls = []
    result = batch.run_batch(args, execute_fn=worker(calls))
    entry = next(iter(result["jobs"].values()))
    assert entry["attempts"][0]["state"] == "interrupted"
    assert entry["attempts"][1]["state"] == "complete"
    assert len(calls) == 1
