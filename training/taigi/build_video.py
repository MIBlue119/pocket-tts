"""Build one bounded, enhanced Chinese weak-label release from a declared Taigi video."""

from __future__ import annotations

import argparse
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
from typing import Any
from urllib.parse import parse_qs, urlparse

import soundfile as sf

from training.taigi.audio import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def execute(command: list[str], log: Path, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("pipeline deadline expired")
    with log.open("wb") as output, log.with_suffix(".stderr.log").open("wb") as errors:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=output, stderr=errors, start_new_session=True
        )
        try:
            code = process.wait(timeout=remaining)
            if code:
                raise RuntimeError(f"stage exited {code}; inspect {log.name}")
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def build(args: argparse.Namespace) -> Path:
    parsed = urlparse(args.url)
    query = parse_qs(parsed.query)
    video_id = query.get("v", [""])[0] if parsed.path == "/watch" else parsed.path.lstrip("/")
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
        or "list" in query
        or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id)
    ):
        raise ValueError("provide one HTTPS YouTube watch URL without a playlist")
    for name, lower, upper in (
        ("duration", 0, 20),
        ("max_source_seconds", 0, 300),
        ("max_download_mib", 0, 20),
        ("timeout", 0, 3600),
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or not lower < value <= upper:
            raise ValueError(f"{name} must be in ({lower}, {upper}]")
    if (
        not math.isfinite(args.start)
        or args.start < 0
        or not args.group_id.strip()
        or not args.train_only
    ):
        raise ValueError("nonnegative start, group-id, and explicit --train-only are required")
    runtime = shutil.which(args.js_runtime)
    if runtime is None:
        raise ValueError(f"JavaScript runtime not found: {args.js_runtime}")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "state": "running",
        "stage": "metadata",
        "url": args.url,
        "started_at": time.time(),
        "timeout_seconds": args.timeout,
    }
    deadline = time.monotonic() + args.timeout

    def interrupted(signum: int, frame: FrameType | None) -> None:
        for received in (signal.SIGTERM, signal.SIGINT):
            signal.signal(received, signal.SIG_IGN)
        raise SystemExit(128 + signum)

    previous = {
        received: signal.signal(received, interrupted)
        for received in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        base = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--ignore-config",
            "--no-playlist",
            "--js-runtimes",
            f"{args.js_runtime}:{runtime}",
        ]
        execute(
            [*base, "--dump-single-json", "--skip-download", args.url],
            out / "metadata.log",
            deadline,
        )
        info = json.loads((out / "metadata.log").read_text())
        duration = info.get("duration")
        if (
            info.get("_type") in {"playlist", "multi_video"}
            or info.get("id") != video_id
            or info.get("is_live")
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or not 0 < duration <= args.max_source_seconds
        ):
            raise ValueError(
                "metadata must describe one finite video within the source duration cap"
            )
        report["stage"] = "download"
        execute(
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
                str(out / "source.%(ext)s"),
                args.url,
            ],
            out / "download.log",
            deadline,
        )
        raw = out / "source.wav"
        media = [p for p in out.glob("source.*") if p.suffix != ".wav"]
        if len(media) != 1 or media[0].stat().st_size > args.max_download_mib * 1024**2:
            raise ValueError("downloaded media does not satisfy the single-file byte cap")
        audio = sf.info(raw)
        first, count = round(args.start * audio.samplerate), round(args.duration * audio.samplerate)
        if audio.duration > args.max_source_seconds or first + count > audio.frames:
            raise ValueError(
                "decoded source exceeds duration cap or does not cover the requested interval"
            )
        report["stage"] = "enhance"
        enhanced = out / "enhanced.wav"
        command = [
            str(args.enhancer_python.expanduser().absolute()),
            "-m",
            "training.taigi.enhance",
            str(raw),
            str(enhanced),
            "--start",
            str(args.start),
            "--duration",
            str(args.duration),
            "--chunk-seconds",
            "5",
        ]
        if args.cache_dir:
            command += ["--cache-dir", str(args.cache_dir.resolve())]
        if args.download_model:
            command += ["--download-model"]
        execute(command, out / "enhance.log", deadline)
        sidecar = enhanced.with_suffix(".wav.json")
        lineage = json.loads(sidecar.read_text())
        enhanced_info = sf.info(enhanced)
        if (
            lineage["source_start_sample"] != first
            or lineage["num_samples"] != count
            or lineage["sample_rate"] != audio.samplerate
            or enhanced_info.frames != count
            or enhanced_info.samplerate != audio.samplerate
            or lineage["source_audio_sha256"] != sha256_file(raw)
            or lineage["audio_sha256"] != sha256_file(enhanced)
        ):
            raise ValueError(
                "enhancement output does not match the exact source interval and hashes"
            )
        sources = out / "sources.jsonl"
        source = {
            "source_id": video_id,
            "group_id": args.group_id,
            "language": "nan-tw",
            "language_provenance": "user_declared",
            "audio_variant": "enhanced",
            "audio_path": str(enhanced),
            "original_url": args.url,
            "source_audio_sha256": lineage["source_audio_sha256"],
            "source_interval": {
                "start_sample": first,
                "num_samples": count,
                "sample_rate": audio.samplerate,
            },
            "enhancement_sidecar": str(sidecar),
            "enhancement": lineage,
            "license": info.get("license") or "unknown",
        }
        sources.write_text(json.dumps(source, ensure_ascii=False) + "\n")
        report["stage"] = "prepare"
        candidates = out / "candidates"
        execute(
            [
                sys.executable,
                "-m",
                "training.taigi",
                "prepare",
                str(sources),
                str(candidates),
                "--train-only",
            ],
            out / "prepare.log",
            deadline,
        )
        report["stage"] = "transcribe"
        releases = out / "releases"
        execute(
            [
                sys.executable,
                "-m",
                "training.taigi",
                "transcribe",
                str(candidates / "candidates.jsonl"),
                str(releases),
                "--device",
                args.device,
            ],
            out / "transcribe.log",
            deadline,
        )
        published = list(releases.glob("[!.]*/report.json"))
        if len(published) != 1:
            raise ValueError("expected exactly one published release report")
        release_report = json.loads(published[0].read_text())
        if (
            not release_report.get("accepted_weak_label_count", 0)
            or not (published[0].parent / "train.jsonl").stat().st_size
        ):
            raise ValueError("release contains no accepted weak labels")
        report.update(
            state="weak_labels_ready",
            release=str(published[0].parent),
            release_report=release_report,
        )
        return published[0].parent
    except BaseException as error:
        report.update(state="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["ended_at"] = time.time()
        try:
            (out / "run-report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n"
            )
        finally:
            for received, handler in previous.items():
                signal.signal(received, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--enhancer-python", type=Path, required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--train-only", action="store_true", required=True)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--max-source-seconds", type=float, default=300)
    parser.add_argument("--max-download-mib", type=float, default=20)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--js-runtime", choices=("node", "deno"), default="node")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    print(build(parser.parse_args()))


if __name__ == "__main__":
    main()
