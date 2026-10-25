"""Bounded, auditable training attempts. No cloud provisioning or artifact uploads."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Generator
from pathlib import Path
from types import FrameType
from typing import Any, Protocol

import yaml

ROOT = Path(__file__).resolve().parents[2]


class EventSource(Protocol):
    def get(self) -> dict[str, float | int] | None: ...  # noqa: ANN101


class EventSink(Protocol):
    def put_nowait(self, event: dict[str, float | int], /) -> None: ...  # noqa: ANN101


METRICS = {"loss", "flow_loss", "eos_loss", "lr", "grad_norm", "it_s", "ce_loss"}


def digest(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@contextlib.contextmanager
def run_lock(run_dir: Path) -> Generator[None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".launcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("run directory is already locked") from None
        yield


def checkpoint_pair(run_dir: Path) -> tuple[Path, Path] | None:
    checkpoints = sorted(
        p for p in run_dir.glob("checkpoint_*.pt") if re.fullmatch(r"checkpoint_\d+\.pt", p.name)
    )
    if not checkpoints:
        return None
    checkpoint = checkpoints[-1]
    optimizer = checkpoint.with_name(checkpoint.name.replace("checkpoint_", "optim_"))
    if not optimizer.is_file() or optimizer.stat().st_size == 0:
        raise ValueError("latest checkpoint lacks a complete optimizer sidecar")
    if checkpoint.stat().st_size == 0:
        raise ValueError("latest checkpoint is empty")
    return checkpoint, optimizer


def identity(config_path: Path, config: dict[str, Any], provenance: list[str]) -> dict[str, Any]:
    inputs = {}
    for key in ("train_jsonl", "valid_jsonl"):
        value = config.get("data", {}).get(key)
        if value:
            path = (ROOT / value).resolve()
            inputs[key] = {"path": str(path), "sha256": digest(path)}
    if "train_jsonl" not in inputs:
        raise ValueError("data.train_jsonl is required")
    for index, value in enumerate(provenance):
        path = Path(value).resolve()
        inputs[f"provenance_{index}"] = {"path": str(path), "sha256": digest(path)}
    model = config.get("model_config")
    if model and (ROOT / model).is_file():
        model_path = (ROOT / model).resolve()
        inputs["model_config"] = {"path": str(model_path), "sha256": digest(model_path)}
    return {"config_sha256": digest(config_path), "inputs": inputs}


def numeric_metrics(record: dict[str, Any]) -> dict[str, float | int]:
    kind = record.get("type")
    if kind not in {"train", "valid", "checkpoint", "start"}:
        return {}
    result = {}
    values = dict(record.get("metrics") or {})
    values.update({key: record[key] for key in METRICS if key in record})
    for key, value in values.items():
        if key in METRICS and type(value) in (int, float) and math.isfinite(value):
            result[f"{kind}/{key}"] = value
    if type(record.get("step")) is int:
        result["trainer_step"] = record["step"]
    return result


def wandb_worker(events: EventSource, settings: dict[str, str], metadata: dict[str, Any]) -> None:
    # Isolate SDK/network stalls from the trainer's wall-clock budget.
    try:
        import wandb

        os.environ["WANDB_CONSOLE"] = "off"
        os.environ["WANDB_DISABLE_CODE"] = "true"
        mode = settings["mode"]
        if mode not in ("online", "offline", "disabled"):
            raise ValueError("unsupported W&B mode")
        run = wandb.init(
            entity=settings["entity"],
            project=settings["project"],
            mode=mode,
            id=settings["attempt"],
            name=settings["attempt"],
            resume="never",
            dir=settings["directory"],
            config=metadata,
            settings=wandb.Settings(
                disable_git=True,
                x_disable_meta=True,
                disable_code=True,
                console="off",
                x_disable_stats=True,
                init_timeout=15,
            ),
        )
        while True:
            event = events.get()
            if event is None:
                break
            run.log(event)
        run.finish()
    except Exception:
        print("W&B unavailable; progress.jsonl remains authoritative", file=sys.stderr)


def signal_group(child: subprocess.Popen[bytes], signum: int) -> None:
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        pass


def supervise(
    command: list[str],
    run_dir: Path,
    max_seconds: float,
    grace_seconds: float,
    events: EventSink | None = None,
) -> tuple[int, str]:
    progress_path = run_dir / "progress.jsonl"
    offset = progress_path.stat().st_size if progress_path.exists() else 0
    pending = b""
    requested_signal = None

    def request_stop(signum: int, frame: FrameType | None) -> None:
        nonlocal requested_signal
        requested_signal = signum

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    child = None
    reason = "completed"
    try:
        with (run_dir / "trainer.log").open("ab") as log:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + max_seconds
            stop_deadline = None
            while True:
                code = child.poll()
                if progress_path.exists():
                    with progress_path.open("rb") as progress:
                        progress.seek(offset)
                        chunk = progress.read()
                        offset = progress.tell()
                    pending += chunk
                    lines = pending.split(b"\n")
                    pending = lines.pop()
                    for line in lines:
                        try:
                            metric = numeric_metrics(json.loads(line))
                        except (ValueError, TypeError, AttributeError):
                            continue
                        if metric and events is not None:
                            try:
                                events.put_nowait(metric)
                            except queue.Full:
                                pass
                if code is not None:
                    # A worker must not outlive the training process.
                    signal_group(child, signal.SIGTERM)
                    signal_group(child, signal.SIGKILL)
                    return code, reason
                if stop_deadline is None and (requested_signal or time.monotonic() >= deadline):
                    reason = "interrupted" if requested_signal else "timeout"
                    signal_group(child, signal.SIGTERM)
                    stop_deadline = time.monotonic() + grace_seconds
                if stop_deadline is not None and time.monotonic() >= stop_deadline:
                    signal_group(child, signal.SIGKILL)
                time.sleep(0.05)
    finally:
        if child is not None and child.poll() is None:
            signal_group(child, signal.SIGKILL)
            child.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run(args: argparse.Namespace, command: list[str] | None = None) -> int:
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    run_dir = (ROOT / config["run_dir"]).resolve()
    if (
        not math.isfinite(args.max_seconds)
        or not math.isfinite(args.grace_seconds)
        or args.max_seconds <= 0
        or args.grace_seconds < 0
    ):
        raise ValueError("max-seconds must be positive and grace-seconds nonnegative")
    with run_lock(run_dir):
        current = identity(config_path, config, args.provenance)
        manifest_path = run_dir / "run_manifest.json"
        pair = checkpoint_pair(run_dir)
        if manifest_path.exists():
            if json.loads(manifest_path.read_text())["identity"] != current:
                raise ValueError("resume rejected: config, input manifests, or provenance changed")
        elif pair or (run_dir / "progress.jsonl").exists():
            raise ValueError("existing training has no launcher identity; use a new run directory")
        else:
            write_json(manifest_path, {"schema_version": 1, "identity": current})
        attempt = uuid.uuid4().hex
        attempt_dir = run_dir / "attempts"
        attempt_dir.mkdir(exist_ok=True)
        path = attempt_dir / f"{attempt}.json"
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT))
        record = {
            "attempt_id": attempt,
            "git_sha": sha,
            "git_dirty": dirty,
            "identity": current,
            "started_at": time.time(),
            "state": "running",
            "max_seconds": args.max_seconds,
            "grace_seconds": args.grace_seconds,
            "resumed_checkpoint": pair[0].name if pair else None,
        }
        write_json(path, record)
        print(f"Run manifest: {manifest_path}\nAttempt: {path}", flush=True)
        worker = events = None
        try:
            if args.wandb != "disabled":
                ctx = multiprocessing.get_context("spawn")
                events = ctx.Queue(maxsize=1000)
                metadata = {"git_sha": sha, "config_sha256": current["config_sha256"]}
                metadata["input_sha256"] = {k: v["sha256"] for k, v in current["inputs"].items()}
                worker = ctx.Process(
                    target=wandb_worker,
                    args=(
                        events,
                        {
                            "entity": args.entity,
                            "project": args.project,
                            "mode": args.wandb,
                            "attempt": attempt,
                            "directory": str(run_dir),
                        },
                        metadata,
                    ),
                    daemon=True,
                )
                worker.start()
            code, reason = supervise(
                command or [sys.executable, "-m", "training.train", str(config_path)],
                run_dir,
                args.max_seconds,
                args.grace_seconds,
                events,
            )
            exit_code = 124 if reason == "timeout" else (130 if reason == "interrupted" else code)
            if exit_code < 0:
                exit_code = 128 - exit_code
            record.update(
                state=reason if reason != "completed" else ("succeeded" if code == 0 else "failed"),
                child_exit_code=code,
                exit_code=exit_code,
            )
            return exit_code
        except BaseException as error:
            record.update(state="launcher_failed", error_type=type(error).__name__, exit_code=1)
            raise
        finally:
            record["ended_at"] = time.time()
            write_json(path, record)
            if worker is not None and worker.pid is not None and events is not None:
                try:
                    events.put_nowait(None)
                except queue.Full:
                    pass
                worker.join(timeout=2)
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=1)
                if worker.is_alive():
                    worker.kill()
                    worker.join()
                events.cancel_join_thread()
                events.close()


def snapshot(run_dir: Path | str, destination: Path | str) -> None:
    run_dir, destination = Path(run_dir).resolve(), Path(destination).resolve()
    with run_lock(run_dir):
        pair = checkpoint_pair(run_dir)
        if pair is None:
            raise ValueError("no complete checkpoint to snapshot")
        if not (run_dir / "run_manifest.json").exists():
            raise ValueError("snapshot requires a run manifest")
        destination.mkdir(parents=True, exist_ok=False)
        paths = [*pair, run_dir / "run_manifest.json"]
        if (run_dir / "progress.jsonl").exists():
            paths.append(run_dir / "progress.jsonl")
        paths.extend(sorted((run_dir / "attempts").glob("*.json")))
        checksums = {}
        for source in paths:
            relative = source.relative_to(run_dir)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            checksums[str(relative)] = digest(target)
            if checksums[str(relative)] != digest(source):
                raise ValueError("snapshot source changed while copying")
        write_json(destination / "checksums.json", checksums)
        (destination / "READY").write_text("complete\n")
        print(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    launch = commands.add_parser("run")
    launch.add_argument("config")
    launch.add_argument("--max-seconds", type=float, required=True)
    launch.add_argument("--grace-seconds", type=float, default=120)
    launch.add_argument("--wandb", choices=("disabled", "offline", "online"), default="disabled")
    launch.add_argument("--entity", default="weirenlan")
    launch.add_argument("--project", default="pocket-tts-taigi")
    launch.add_argument("--provenance", action="append", default=[])
    copy = commands.add_parser("snapshot")
    copy.add_argument("run_dir")
    copy.add_argument("destination")
    args = parser.parse_args()
    try:
        if args.action == "snapshot":
            snapshot(args.run_dir, args.destination)
            return 0
        return run(args)
    except (ValueError, OSError, KeyError) as error:
        print(f"Launcher rejected request: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
