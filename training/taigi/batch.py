"""Run a finite, resumable batch of bounded single-video weak-label jobs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from training.taigi.audio import sha256_file
from training.taigi.build_video import ROOT


class Cancelled(Exception):
    pass


class CleanupError(RuntimeError):
    pass


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def process_identity(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def descendants(pid: int) -> dict[int, str | None]:
    children = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            children.setdefault(int(fields[1]), []).append(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    found = {}
    pending = [pid]
    while pending:
        current = pending.pop()
        found[current] = process_identity(current)
        pending.extend(children.get(current, []))
    return found


def stop_wrapper(pid: int, identity: str) -> None:
    if process_identity(pid) != identity:
        return
    tracked = descendants(pid)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    grace = time.monotonic() + 15

    def alive() -> list[int]:
        return [
            p for p, token in tracked.items() if token is not None and process_identity(p) == token
        ]

    while alive() and time.monotonic() < grace:
        time.sleep(0.05)
    remaining = alive()
    for child in remaining:
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
    end = time.monotonic() + 5
    while alive() and time.monotonic() < end:
        time.sleep(0.05)
    if alive():
        raise CleanupError("wrapper descendants did not reach terminal state")


def execute(command: list[str], log: Path, deadline: float) -> None:
    """Give the wrapper time to clean up its independently grouped stage process."""
    if time.monotonic() >= deadline:
        raise TimeoutError("batch deadline expired")
    with log.open("wb") as output:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )
        identity = process_identity(process.pid)
        try:
            if identity is not None:
                atomic_json(log.with_suffix(".pid.json"), dict(pid=process.pid, identity=identity))
            code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            if code:
                raise RuntimeError(f"wrapper exited {code}; see {log}")
        finally:
            if process.poll() is None and identity is not None:
                stop_wrapper(process.pid, identity)
            process.wait(timeout=5)


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plan_jobs(path: Path) -> list[dict[str, Any]]:
    result = []
    intervals = {}
    for row in rows(path):
        if set(row) != {"url", "group_id", "start", "duration"}:
            raise ValueError("jobs require exactly url, group_id, start, duration")
        parsed = urlparse(row["url"])
        query = parse_qs(parsed.query)
        video = query.get("v", [""])[0] if parsed.path == "/watch" else parsed.path.lstrip("/")
        if (
            parsed.scheme != "https"
            or parsed.hostname
            not in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
            or "list" in query
            or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video)
        ):
            raise ValueError("jobs require individual HTTPS YouTube URLs")
        start, duration = float(row["start"]), float(row["duration"])
        if (
            not math.isfinite(start)
            or start < 0
            or not math.isfinite(duration)
            or not 0 < duration <= 20
            or not isinstance(row["group_id"], str)
            or not row["group_id"].strip()
        ):
            raise ValueError("invalid interval or group_id")
        interval = (video, start, duration)
        if interval in intervals:
            if intervals[interval] != row["group_id"]:
                raise ValueError("conflicting groups for duplicate video interval")
            continue
        intervals[interval] = row["group_id"]
        job = dict(
            url=f"https://www.youtube.com/watch?v={video}",
            group_id=row["group_id"],
            start=start,
            duration=duration,
        )
        job["key"] = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:24]
        result.append(job)
    if not result:
        raise ValueError("empty batch plan")
    return result


def verify_release(attempt: Path) -> dict[str, str]:
    report_path = attempt / "run-report.json"
    report = json.loads(report_path.read_text())
    release = Path(report["release"]).resolve()
    if report.get("state") != "weak_labels_ready" or not release.is_relative_to(attempt.resolve()):
        raise ValueError("release integrity: wrapper is not ready or release escaped attempt")
    metadata = rows(release / "metadata.jsonl")
    release_report = json.loads((release / "report.json").read_text())
    if not metadata or release_report["accepted_weak_label_count"] != len(metadata):
        raise ValueError("release integrity: accepted count mismatch")
    files = [
        report_path,
        *(
            release / name
            for name in (
                "report.json",
                "metadata.jsonl",
                "train.jsonl",
                "valid.jsonl",
                "quarantine.jsonl",
            )
        ),
    ]
    for split in ("train", "valid"):
        expected = [
            dict(
                path=r["published_audio_path"],
                start=r["start"],
                duration=r["duration"],
                transcript=r["input_text_zh"],
            )
            for r in metadata
            if r["split"] == split
        ]
        if rows(release / f"{split}.jsonl") != expected:
            raise ValueError("release integrity: compact manifest mismatch")
    if any(r["split"] not in {"train", "valid"} for r in metadata):
        raise ValueError("release integrity: unknown split")
    for row in metadata:
        audio = Path(row["published_audio_path"]).resolve()
        if not audio.is_relative_to(release) or sha256_file(audio) != row["audio_sha256"]:
            raise ValueError("release audio hash integrity failure")
        files.append(audio)
    return {str(path.resolve()): sha256_file(path) for path in files}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jobs", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--enhancer-python", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--max-seconds", type=float, default=3600)
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--min-free-gib", type=float, default=5)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--max-source-seconds", type=float, default=300)
    p.add_argument("--max-download-mib", type=float, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--js-runtime", choices=("node", "deno"), default="node")
    p.add_argument("--download-model", action="store_true")
    return p


def run_batch(
    args: argparse.Namespace, *, execute_fn: Callable[[list[str], Path, float], None] = execute
) -> dict[str, Any]:
    jobs = plan_jobs(args.jobs)
    for key, upper in [
        ("max_seconds", math.inf),
        ("timeout", 3600),
        ("max_source_seconds", 300),
        ("max_download_mib", 20),
    ]:
        if not math.isfinite(getattr(args, key)) or not 0 < getattr(args, key) <= upper:
            raise ValueError(f"invalid {key}")
    if args.max_attempts < 1 or not math.isfinite(args.min_free_gib) or args.min_free_gib < 0:
        raise ValueError("invalid attempt cap or disk reserve")
    config = {
        k: (str(v.expanduser().absolute()) if k == "enhancer_python" else str(v.resolve()))
        if isinstance(v, Path)
        else v
        for k, v in vars(args).items()
        if k not in {"jobs", "output"}
    }
    plan = dict(version=1, jobs=jobs, config=config)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path, state_path = out / "plan.json", out / "state.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError("batch plan/config changed; use a new output directory")
        else:
            if state_path.exists() or (out / "attempts").exists():
                raise ValueError("missing persisted batch plan")
            atomic_json(plan_path, plan)
        state: dict[str, Any] = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {
                "version": 1,
                "jobs": {job["key"]: {"state": "pending", "attempts": []} for job in jobs},
            }
        )
        if set(state["jobs"]) != {j["key"] for j in jobs}:
            raise ValueError("batch state does not match plan")
        for entry in state["jobs"].values():
            if entry["state"] == "complete":
                attempt = Path(entry["attempts"][-1]["path"])
                try:
                    if verify_release(attempt) != entry["checksums"]:
                        raise ValueError("saved release checksums changed")
                except (OSError, KeyError, ValueError) as error:
                    raise ValueError(f"completed artifact integrity failed: {error}") from error
            for attempt in entry["attempts"]:
                if attempt["state"] == "running":
                    previous_path = Path(attempt["path"])
                    pid_file = previous_path.parent / f"{previous_path.name}.pid.json"
                    if pid_file.exists():
                        process_record = json.loads(pid_file.read_text())
                        stop_wrapper(process_record["pid"], process_record["identity"])
                    attempt.update(state="interrupted", error="previous batch stopped")
                    entry["state"] = "interrupted"
        atomic_json(state_path, state)
        deadline = time.monotonic() + args.max_seconds

        def cancel(signum: int, frame: FrameType | None) -> None:
            # Ignore subsequent signals while the wrapper finishes cleaning its stage groups.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            raise Cancelled(f"received signal {signum}")

        previous = {s: signal.signal(s, cancel) for s in (signal.SIGTERM, signal.SIGINT)}
        reason = None
        try:
            for job_index, job in enumerate(jobs, start=1):
                entry = state["jobs"][job["key"]]
                while entry["state"] != "complete" and len(entry["attempts"]) < args.max_attempts:
                    if time.monotonic() >= deadline:
                        reason = "deadline"
                        break
                    if shutil.disk_usage(out).free < args.min_free_gib * 1024**3:
                        reason = "disk_reserve"
                        break
                    attempt_dir = out / "attempts" / job["key"] / str(len(entry["attempts"]) + 1)
                    attempt_dir.parent.mkdir(parents=True, exist_ok=True)
                    if attempt_dir.exists():
                        raise ValueError("unrecorded attempt directory; refusing overwrite")
                    attempt = dict(path=str(attempt_dir), state="running", started_at=time.time())
                    entry["attempts"].append(attempt)
                    entry["state"] = "running"
                    atomic_json(state_path, state)
                    command = [
                        sys.executable,
                        "-m",
                        "training.taigi.build_video",
                        job["url"],
                        str(attempt_dir),
                        "--enhancer-python",
                        config["enhancer_python"],
                        "--group-id",
                        job["group_id"],
                        "--train-only",
                        "--start",
                        str(job["start"]),
                        "--duration",
                        str(job["duration"]),
                    ]
                    for name in (
                        "timeout",
                        "max_source_seconds",
                        "max_download_mib",
                        "device",
                        "js_runtime",
                    ):
                        command += ["--" + name.replace("_", "-"), str(config[name])]
                    if args.cache_dir:
                        command += ["--cache-dir", config["cache_dir"]]
                    if args.download_model:
                        command += ["--download-model"]
                    try:
                        execute_fn(
                            command,
                            attempt_dir.parent / f"{attempt_dir.name}.log",
                            min(deadline, time.monotonic() + args.timeout),
                        )
                        entry["checksums"] = verify_release(attempt_dir)
                        attempt["state"] = entry["state"] = "complete"
                    except Cancelled as error:
                        attempt["state"] = entry["state"] = "interrupted"
                        attempt["error"] = str(error)
                        reason = "cancelled"
                    except CleanupError as error:
                        attempt["state"] = entry["state"] = "failed"
                        attempt["error"] = str(error)
                        reason = "cleanup_failed"
                    except Exception as error:
                        attempt["state"] = entry["state"] = "failed"
                        attempt["error"] = f"{type(error).__name__}: {error}"
                    finally:
                        attempt["ended_at"] = time.time()
                        atomic_json(state_path, state)
                        print(
                            f"Job {job_index}/{len(jobs)}: {attempt['state']}"
                            + (f" ({attempt['error']})" if "error" in attempt else ""),
                            flush=True,
                        )
                    if reason:
                        break
                if reason:
                    break
        except Cancelled:
            reason = "cancelled"
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)
            report = dict(
                state="complete"
                if all(e["state"] == "complete" for e in state["jobs"].values())
                else "incomplete",
                reason=reason,
                jobs=state["jobs"],
            )
            atomic_json(out / "batch-report.json", report)
        return report


def main() -> None:
    args = parser().parse_args()
    report = run_batch(args)
    completed = sum(entry["state"] == "complete" for entry in report["jobs"].values())
    print(
        f"Batch {report['state']}: {completed}/{len(report['jobs'])} complete; "
        f"report: {args.output.resolve() / 'batch-report.json'}",
        flush=True,
    )
    raise SystemExit(0 if report["state"] == "complete" else 1)


if __name__ == "__main__":
    main()
