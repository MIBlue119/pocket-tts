import json
from argparse import Namespace
from pathlib import Path

import pytest

from training.scripts.publish_taigi_when_ready import ready_selection


def test_waits_for_verified_backup(tmp_path: Path):
    args = Namespace(production=tmp_path / "production", receipt=tmp_path / "receipt.json")
    args.receipt.write_text(json.dumps({"state": "watching", "backup": "incomplete"}))
    assert ready_selection(args) is None


def test_rejects_path_outside_verified_production(tmp_path: Path):
    production = tmp_path / "production"
    (production / "continuation").mkdir(parents=True)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"state": "complete", "backup": "verified", "local_output": str(production)})
    )
    selection = {
        "state": "complete",
        "candidate_id": "step-200-raw",
        "files": {
            name: {"path": str(tmp_path / "outside"), "sha256": "0" * 64}
            for name in ("weights", "config", "tokenizer", "evaluation", "provenance")
        },
    }
    (production / "continuation/selected.json").write_text(json.dumps(selection))
    args = Namespace(production=production, receipt=receipt)
    with pytest.raises(ValueError, match="escapes"):
        ready_selection(args)


def test_verified_selection_and_tamper(tmp_path: Path):
    import hashlib

    production = tmp_path / "production"
    (production / "continuation").mkdir(parents=True)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"state": "complete", "backup": "verified", "local_output": str(production)})
    )
    files = {}
    for name in ("weights", "config", "tokenizer", "evaluation", "provenance"):
        p = production / name
        p.write_text(name)
        files[name] = {"path": str(p), "sha256": hashlib.sha256(name.encode()).hexdigest()}
    selection = {"state": "complete", "candidate_id": "step-200-raw", "files": files}
    (production / "continuation/selected.json").write_text(json.dumps(selection))
    args = Namespace(production=production, receipt=receipt)
    assert ready_selection(args) == selection
    (production / "weights").write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        ready_selection(args)


def test_publisher_timeout_stops_child_group(tmp_path: Path):
    import subprocess
    import sys
    import time

    from training.scripts.publish_taigi_when_ready import run_publisher

    pidfile = tmp_path / "child.pid"
    script = (
        "import subprocess,time,pathlib; "
        'p=subprocess.Popen(["sleep","120"]); '
        f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); "
        "time.sleep(120)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        run_publisher([sys.executable, "-c", script], 0.5, tmp_path / "log")
    pid = int(pidfile.read_text())
    for _ in range(20):
        stat = Path(f"/proc/{pid}/stat")
        if not stat.exists() or stat.read_text().split()[2] == "Z":
            break
        time.sleep(0.05)
    else:
        pytest.fail("publisher descendant survived timeout")
