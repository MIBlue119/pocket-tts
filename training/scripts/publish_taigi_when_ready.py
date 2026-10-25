"""Publish the selected candidate locally, only after verified cloud backup.

HF credentials remain on the local machine. This process neither provisions nor
stops compute. The separate cloud watchdog keeps its original accounting clock.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from training.scripts.run_experiment import digest, write_json

FILES = ("weights", "config", "tokenizer", "evaluation", "provenance")


def ready_selection(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.receipt.exists():
        return None
    receipt = json.loads(args.receipt.read_text())
    if receipt.get("state") != "complete" or receipt.get("backup") != "verified":
        if receipt.get("state") == "deadline":
            raise ValueError("cloud deadline reached without verified backup")
        return None
    root = args.production.resolve(strict=True)
    if Path(receipt.get("local_output", "")).resolve() != root:
        raise ValueError("backup receipt refers to another production directory")
    path = root / "continuation/selected.json"
    if not path.exists():
        raise ValueError("completed backup contains no selected candidate")
    selection = json.loads(path.read_text())
    if selection.get("state") != "complete":
        raise ValueError("selected candidate is incomplete")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}", selection.get("candidate_id", "")):
        raise ValueError("invalid candidate identifier")
    for name in FILES:
        item = selection["files"][name]
        file = Path(item["path"])
        if file.is_symlink() or not file.resolve().is_relative_to(root):
            raise ValueError(f"selected {name} escapes verified production")
        if not file.is_file() or digest(file) != item["sha256"]:
            raise ValueError(f"selected {name} hash mismatch")
    return selection


def run_publisher(command: list[str], timeout: float, log: Path) -> None:
    """Contain inference descendants even if the publication deadline expires."""
    with log.open("ab") as stream:
        child = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = child.wait(timeout=timeout)
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def run(args: argparse.Namespace) -> int:
    if not math.isfinite(args.deadline_epoch) or args.deadline_epoch <= time.time():
        raise ValueError("publication deadline must be in the future")
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output / "delivery.json"
    if receipt.exists() and json.loads(receipt.read_text()).get("state") == "complete":
        return 0
    write_json(receipt, {"state": "waiting_for_verified_backup", "repo_id": args.repo_id})
    try:
        while time.time() < args.deadline_epoch:
            selection = ready_selection(args)
            if selection is not None:
                package = args.output / selection["candidate_id"]
                command = [
                    sys.executable,
                    "-m",
                    "training.scripts.publish_taigi_model",
                    "--output",
                    str(package),
                    "--candidate-id",
                    selection["candidate_id"],
                    "--repo-id",
                    args.repo_id,
                    "--publish",
                ]
                for name in FILES:
                    command.extend(["--" + name, selection["files"][name]["path"]])
                write_json(
                    receipt,
                    {
                        "state": "publishing",
                        "repo_id": args.repo_id,
                        "candidate_id": selection["candidate_id"],
                    },
                )
                run_publisher(
                    command, max(1, args.deadline_epoch - time.time()), args.output / "publish.log"
                )
                write_json(
                    receipt,
                    {
                        "state": "complete",
                        "repo_id": args.repo_id,
                        "candidate_id": selection["candidate_id"],
                        "package": str(package),
                        "quality_verified": False,
                        "ended_at": time.time(),
                    },
                )
                return 0
            time.sleep(min(60, max(0, args.deadline_epoch - time.time())))
        raise TimeoutError("publication deadline reached")
    except Exception as error:
        write_json(
            receipt,
            {
                "state": "failed",
                "repo_id": args.repo_id,
                "error": f"{type(error).__name__}: {error}"[:500],
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
