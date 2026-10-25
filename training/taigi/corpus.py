"""Download and process every video in a frozen playlist inventory, with resumable stages."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import re
import shutil
import signal
import sys
import time
from collections import Counter, deque
from pathlib import Path
from types import FrameType
from typing import Any, Callable

import soundfile as sf

from training.taigi.audio import sha256_file
from training.taigi.batch import Cancelled, CleanupError, atomic_json, execute, stop_attempt
from training.taigi.corpus_worker import _context

Executor = Callable[[list[str], Path, float], None]
STAGES = ("acquire", "enhance", "transcribe")


def code_identity() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        name: sha256_file(directory / name)
        for name in (
            "corpus.py",
            "corpus_worker.py",
            "batch.py",
            "enhance.py",
            "pipeline.py",
            "transcriber.py",
            "audio.py",
            "schema.py",
        )
    }


def normalize_inventory(path: Path) -> list[dict[str, Any]]:
    inventory = json.loads(path.read_text())
    videos: dict[str, dict[str, Any]] = {}
    parents: dict[str, str] = {}
    playlists: list[deque[str]] = []

    def root(group: str) -> str:
        parents.setdefault(group, group)
        while parents[group] != group:
            group = parents[group]
        return group

    for source_index, source in enumerate(inventory["sources"]):
        playlist = str(source.get("playlist_id") or f"unknown-playlist-{source_index}")
        root(playlist)
        entries = source.get("entries") or []
        if not entries and source.get("exit_code", 0):
            entries = [{"id": None, "title": "playlist enumeration failed"}]
        playlist_keys: deque[str] = deque()
        playlists.append(playlist_keys)
        for index, entry in enumerate(entries):
            entry = entry if isinstance(entry, dict) else {}
            video_id = entry.get("id")
            valid_id = isinstance(video_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id)
            key = (
                video_id
                if valid_id
                else "unavailable-"
                + hashlib.sha256(f"{playlist}:{index}".encode()).hexdigest()[:20]
            )
            playlist_keys.append(key)
            if key not in videos:
                videos[key] = dict(
                    key=key,
                    video_id=video_id if valid_id else None,
                    url=f"https://www.youtube.com/watch?v={video_id}" if valid_id else None,
                    title=entry.get("title"),
                    duration=entry.get("duration"),
                    availability=entry.get("availability"),
                    playlists=[],
                )
            row = videos[key]
            if row["playlists"]:
                left, right = root(row["playlists"][0]), root(playlist)
                parents[max(left, right)] = min(left, right)
            if playlist not in row["playlists"]:
                row["playlists"].append(playlist)
    if not videos:
        raise ValueError("inventory has no entries")
    for row in videos.values():
        row["playlists"].sort()
        row["group_id"] = "playlist:" + root(row["playlists"][0])
    ordered: dict[str, dict[str, Any]] = {}
    while any(playlists):
        for playlist_keys in playlists:
            while playlist_keys:
                key = playlist_keys.popleft()
                if key not in ordered:
                    ordered[key] = videos[key]
                    break
    return list(ordered.values())


def verify_hashes(checksums: dict[str, str], directory: Path) -> None:
    if not checksums:
        raise ValueError("artifact integrity: empty checksum record")
    for name, digest in checksums.items():
        path = Path(name).resolve()
        try:
            if not path.is_relative_to(directory.resolve()) or sha256_file(path) != digest:
                raise ValueError(f"artifact integrity failed: {path}")
        except OSError as error:
            raise ValueError(f"artifact integrity failed: {path}") from error


def raw_artifacts(video_dir: Path) -> dict[str, str]:
    path = video_dir / "source-report.json"
    report = json.loads(path.read_text())
    if report.get("state") != "complete":
        raise ValueError("raw artifact integrity: incomplete source")
    verify_hashes(report["artifacts"], video_dir)
    return {**report["artifacts"], str(path): sha256_file(path)}


def worker_artifacts(video_dir: Path, stage: str) -> dict[str, str]:
    directory = video_dir / "stage"
    name = "enhancement-report.json" if stage == "enhance" else "result.json"
    path = directory / name
    report = json.loads(path.read_text())
    spec = json.loads((video_dir / "spec.json").read_text())
    _, _, identity = _context(spec)
    if any(report.get(key) != expected for key, expected in identity.items()):
        raise ValueError("worker artifact identity integrity: report does not match spec/code")
    if stage == "enhance":
        info = sf.info(video_dir / "source.wav")
        if (
            report.get("state") != "complete"
            or report.get("total_frames") != info.frames
            or report.get("covered_frames") != info.frames
            or report.get("sample_rate") != info.samplerate
        ):
            raise ValueError("worker artifact integrity: incomplete source coverage")
        candidates = directory / "candidates/candidates.jsonl"
        if str(candidates) not in report["artifacts"]:
            raise ValueError("worker artifact integrity: candidates are not verified")
    elif report.get("state") not in {"complete", "no_candidates"}:
        raise ValueError("worker artifact integrity: invalid result")
    elif report["state"] == "complete":
        release = Path(report["release_path"]).resolve()
        if (
            not release.is_relative_to(directory)
            or str(release / "metadata.jsonl") not in report["artifacts"]
        ):
            raise ValueError("worker artifact integrity: release is not verified")
    verify_hashes(report["artifacts"], video_dir)
    return {**report["artifacts"], str(path): sha256_file(path)}


def acquire(
    job: dict[str, Any],
    video_dir: Path,
    attempt_dir: Path,
    args: argparse.Namespace,
    deadline: float,
    execute_fn: Executor,
) -> dict[str, str]:
    runtime = shutil.which(args.js_runtime)
    if runtime is None:
        raise ValueError(f"JavaScript runtime unavailable: {args.js_runtime}")
    base = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--ignore-config",
        "--no-playlist",
        "--js-runtimes",
        f"{args.js_runtime}:{runtime}",
    ]
    template = str(attempt_dir / "source.%(ext)s")
    execute_fn(
        [*base, "--skip-download", "--write-info-json", "-o", template, job["url"]],
        attempt_dir / "metadata.log",
        deadline,
    )
    metadata_path = attempt_dir / "source.info.json"
    metadata = json.loads(metadata_path.read_text())
    duration = metadata.get("duration")
    if (
        metadata.get("id") != job["video_id"]
        or metadata.get("is_live")
        or metadata.get("_type") in {"playlist", "multi_video"}
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or not 0 < duration <= args.max_source_seconds
    ):
        raise ValueError(
            "source metadata is unavailable, live, or exceeds the finite duration limit"
        )
    execute_fn(
        [
            *base,
            "--max-filesize",
            str(int(args.max_download_mib * 1024**2)),
            "--match-filter",
            f"!is_live & duration <= {args.max_source_seconds}",
            "-f",
            "bestaudio",
            "-x",
            "--keep-video",
            "--audio-format",
            "wav",
            "-o",
            template,
            job["url"],
        ],
        attempt_dir / "download.log",
        deadline,
    )
    media = [
        p for p in attempt_dir.glob("source.*") if p.name not in {"source.wav", "source.info.json"}
    ]
    if len(media) != 1 or media[0].stat().st_size > args.max_download_mib * 1024**2:
        raise ValueError("download failed the single-media-file size limit")
    raw = attempt_dir / "source.wav"
    info = sf.info(raw)
    if not 0 < info.duration <= args.max_source_seconds:
        raise ValueError("decoded source exceeds duration limit or is empty")
    atomic_json(
        attempt_dir / "source-ready.json",
        dict(
            state="complete",
            frames=info.frames,
            sample_rate=info.samplerate,
            artifacts={
                name: sha256_file(attempt_dir / name) for name in ("source.wav", "source.info.json")
            },
        ),
    )
    return promote_source(attempt_dir, video_dir)


def promote_source(attempt_dir: Path, video_dir: Path) -> dict[str, str]:
    """Recover a verified download even if the supervisor stopped during promotion."""
    ready = json.loads((attempt_dir / "source-ready.json").read_text())
    if set(ready["artifacts"]) != {"source.wav", "source.info.json"}:
        raise ValueError("source promotion integrity: invalid artifact set")
    for name, digest in ready["artifacts"].items():
        final = video_dir / name
        source = final if final.exists() else attempt_dir / name
        if sha256_file(source) != digest:
            raise ValueError("source promotion integrity: changed download")
        if source != final:
            source.replace(final)
    atomic_json(
        video_dir / "source-report.json",
        {
            **ready,
            "artifacts": {
                str(video_dir / name): digest for name, digest in ready["artifacts"].items()
            },
        },
    )
    return raw_artifacts(video_dir)


def status(output: Path) -> dict[str, Any]:
    state = json.loads((output / "state.json").read_text())
    counts = Counter(entry["state"] for entry in state["videos"].values())
    return dict(
        total=len(state["videos"]),
        counts=dict(counts),
        reason=state.get("reason"),
        active=state.get("active"),
        state="complete"
        if state.get("reason") is None
        and all(e["state"] in {"complete", "no_candidates"} for e in state["videos"].values())
        else "incomplete",
    )


def run(args: argparse.Namespace, *, execute_fn: Executor = execute) -> dict[str, Any]:
    queue = normalize_inventory(args.inventory)
    for key in (
        "max_seconds",
        "max_source_seconds",
        "max_download_mib",
        "stage_timeout",
        "download_timeout",
    ):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"invalid {key}")
    if args.max_attempts < 1 or not math.isfinite(args.min_free_gib) or args.min_free_gib < 0:
        raise ValueError("invalid attempt limit or disk reserve")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value.expanduser().absolute()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"command", "inventory", "output", "max_seconds", "min_free_gib"}
    }
    plan = dict(
        version=1,
        inventory_sha256=sha256_file(args.inventory),
        queue=queue,
        config=config,
        code_sha256=code_identity(),
    )
    deadline = time.monotonic() + args.max_seconds
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path, state_path = out / "plan.json", out / "state.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError("frozen corpus plan/config changed")
        else:
            if state_path.exists() or (out / "videos").exists():
                raise ValueError("corpus plan is missing")
            atomic_json(plan_path, plan)
        state: dict[str, Any] = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {
                "version": 1,
                "videos": {
                    job["key"]: dict(
                        state="pending",
                        stages={stage: dict(state="pending", attempts=[]) for stage in STAGES},
                    )
                    for job in queue
                },
            }
        )
        if set(state["videos"]) != {job["key"] for job in queue}:
            raise ValueError("corpus state does not match frozen queue")
        state.update(reason=None, active=None)
        atomic_json(state_path, state)

        def cancel(signum: int, frame: FrameType | None) -> None:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            raise Cancelled(f"received signal {signum}")

        previous = {sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            # Reap owned unfinished work across the entire queue before admitting any GPU stage.
            for video_key, entry in state["videos"].items():
                for stage_name, stage in entry["stages"].items():
                    for attempt in stage["attempts"]:
                        if not attempt.get("cleanup_verified"):
                            for path in Path(attempt["directory"]).glob("*.pid.json"):
                                stop_attempt(json.loads(path.read_text()))
                            attempt["cleanup_verified"] = True
                        if attempt["state"] == "running":
                            attempt.update(state="interrupted", error="previous run stopped")
                            stage["state"] = "interrupted"
                    if stage["state"] == "complete":
                        verify_hashes(stage["checksums"], out)
                        if stage_name != "acquire":
                            worker_artifacts(out / "videos" / video_key, stage_name)
            atomic_json(state_path, state)
            for index, job in enumerate(queue, start=1):
                entry = state["videos"][job["key"]]
                if entry["state"] in {"complete", "no_candidates"}:
                    continue
                if job["video_id"] is None:
                    entry.update(
                        state="failed", error="inventory entry has no available video identifier"
                    )
                    atomic_json(state_path, state)
                    continue
                video_dir = out / "videos" / job["key"]
                video_dir.mkdir(parents=True, exist_ok=True)
                for stage_name in STAGES:
                    if code_identity() != plan["code_sha256"]:
                        raise ValueError("frozen corpus plan code changed")
                    stage = entry["stages"][stage_name]
                    if stage["state"] == "complete":
                        continue
                    # Recover a stage published just before the supervising process stopped.
                    artifact_path = video_dir / (
                        "source-report.json"
                        if stage_name == "acquire"
                        else "stage/enhancement-report.json"
                        if stage_name == "enhance"
                        else "stage/result.json"
                    )
                    if stage_name == "acquire" and not artifact_path.exists():
                        for previous_attempt in reversed(stage["attempts"]):
                            previous_dir = Path(previous_attempt["directory"])
                            if (previous_dir / "source-ready.json").exists():
                                promote_source(previous_dir, video_dir)
                                break
                    if artifact_path.exists():
                        stage["checksums"] = (
                            raw_artifacts(video_dir)
                            if stage_name == "acquire"
                            else worker_artifacts(video_dir, stage_name)
                        )
                        stage["state"] = "complete"
                        atomic_json(state_path, state)
                        continue
                    while (
                        stage["state"] != "complete" and len(stage["attempts"]) < args.max_attempts
                    ):
                        if code_identity() != plan["code_sha256"]:
                            raise ValueError("frozen corpus plan code changed")
                        if time.monotonic() >= deadline:
                            state["reason"] = "deadline"
                            break
                        if shutil.disk_usage(out).free < args.min_free_gib * 1024**3:
                            state["reason"] = "disk_reserve"
                            break
                        attempt_dir = (
                            video_dir / "attempts" / stage_name / str(len(stage["attempts"]) + 1)
                        )
                        attempt_dir.mkdir(parents=True, exist_ok=False)
                        attempt = dict(
                            directory=str(attempt_dir),
                            state="running",
                            started_at=time.time(),
                            cleanup_verified=False,
                        )
                        stage["attempts"].append(attempt)
                        stage["state"] = entry["state"] = "running"
                        state["active"] = dict(
                            video_id=job["video_id"], stage=stage_name, index=index
                        )
                        atomic_json(state_path, state)
                        print(
                            f"Video {index}/{len(queue)} {job['video_id']}: {stage_name}, attempt {len(stage['attempts'])}",
                            flush=True,
                        )
                        try:
                            stage_deadline = min(
                                deadline,
                                time.monotonic()
                                + (
                                    args.download_timeout
                                    if stage_name == "acquire"
                                    else args.stage_timeout
                                ),
                            )
                            if stage_name == "acquire":
                                checksums = acquire(
                                    job, video_dir, attempt_dir, args, stage_deadline, execute_fn
                                )
                            else:
                                spec_path = video_dir / "spec.json"
                                metadata = json.loads((video_dir / "source.info.json").read_text())
                                spec = dict(
                                    raw_audio=str(video_dir / "source.wav"),
                                    video_id=job["video_id"],
                                    group_id=job["group_id"],
                                    original_url=job["url"],
                                    license=metadata.get("license") or "unknown",
                                    output_dir=str(video_dir / "stage"),
                                    cache_dir=config["cache_dir"],
                                    device=args.device,
                                )
                                if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
                                    raise ValueError("persisted worker spec changed")
                                atomic_json(spec_path, spec)
                                python = (
                                    config["enhancer_python"]
                                    if stage_name == "enhance"
                                    else sys.executable
                                )
                                execute_fn(
                                    [
                                        python,
                                        "-m",
                                        "training.taigi.corpus_worker",
                                        stage_name,
                                        str(spec_path),
                                    ],
                                    attempt_dir / "worker.log",
                                    stage_deadline,
                                )
                                checksums = worker_artifacts(video_dir, stage_name)
                            stage.update(state="complete", checksums=checksums)
                            entry["state"] = "pending"
                            attempt.update(state="complete", cleanup_verified=True)
                        except Cancelled as error:
                            attempt.update(
                                state="interrupted", error=str(error), cleanup_verified=True
                            )
                            stage["state"] = entry["state"] = "interrupted"
                            state["reason"] = "cancelled"
                        except CleanupError as error:
                            attempt.update(state="failed", error=str(error))
                            stage["state"] = entry["state"] = "failed"
                            state["reason"] = "cleanup_failed"
                        except Exception as error:
                            attempt.update(
                                state="failed",
                                error=f"{type(error).__name__}: {error}",
                                cleanup_verified=True,
                            )
                            stage["state"] = entry["state"] = "failed"
                        finally:
                            attempt["ended_at"] = time.time()
                            state["active"] = None
                            atomic_json(state_path, state)
                        if state["reason"]:
                            break
                    if state["reason"] or stage["state"] != "complete":
                        break
                if state["reason"]:
                    break
                if all(stage["state"] == "complete" for stage in entry["stages"].values()):
                    entry["state"] = json.loads((video_dir / "stage/result.json").read_text())[
                        "state"
                    ]
                    atomic_json(state_path, state)
        except Cancelled:
            state["reason"] = "cancelled"
        except CleanupError:
            state["reason"] = "cleanup_failed"
            raise
        except (ValueError, OSError, KeyError):
            state["reason"] = "integrity_failure"
            raise
        finally:
            state["active"] = None
            atomic_json(state_path, state)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            report = status(out)
            atomic_json(out / "corpus-report.json", report)
        return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("inventory", type=Path)
    run_parser.add_argument("output", type=Path)
    run_parser.add_argument("--enhancer-python", type=Path, required=True)
    run_parser.add_argument("--cache-dir", type=Path)
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--max-seconds", type=float, default=86400)
    run_parser.add_argument("--max-attempts", type=int, default=2)
    run_parser.add_argument("--min-free-gib", type=float, default=10)
    run_parser.add_argument("--max-source-seconds", type=float, default=7200)
    run_parser.add_argument("--max-download-mib", type=float, default=512)
    run_parser.add_argument("--stage-timeout", type=float, default=7200)
    run_parser.add_argument("--download-timeout", type=float, default=1800)
    run_parser.add_argument("--js-runtime", choices=("node", "deno"), default="node")
    commands.add_parser("status").add_argument("output", type=Path)
    return p


def main() -> None:
    args = parser().parse_args()
    report = status(args.output) if args.command == "status" else run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.command == "run" and report["state"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
