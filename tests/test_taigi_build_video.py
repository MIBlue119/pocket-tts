import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from training.taigi import build_video
from training.taigi.enhance import enhance_file
from training.taigi.pipeline import prepare_candidates, transcribe_release


def options(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        url="https://www.youtube.com/watch?v=abcdefghijk",
        output_dir=tmp_path / "out",
        enhancer_python=Path(sys.executable),
        group_id="speaker-one",
        start=1.25,
        duration=2,
        max_source_seconds=300,
        max_download_mib=20,
        timeout=300,
        js_runtime="node",
        cache_dir=None,
        download_model=False,
        device="cpu",
        train_only=True,
    )


@pytest.mark.parametrize("failure", [None, "metadata", "decoded", "lineage", "empty_asr"])
def test_complete_release_uses_exact_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    args = options(tmp_path)
    commands = []
    deadlines = []
    monkeypatch.setattr(build_video.shutil, "which", lambda name: "/usr/bin/node")

    def fake_execute(command: list[str], log: Path, deadline: float) -> None:
        commands.append(command)
        deadlines.append(deadline)
        log.touch()
        if log.name == "metadata.log":
            log.write_text(
                json.dumps({"id": "abcdefghijk", "duration": 301 if failure == "metadata" else 5})
            )
        elif log.name == "download.log":
            rate = 24000
            samples = (
                0.1
                * np.sin(
                    2 * np.pi * 440 * np.arange((1 if failure == "decoded" else 5) * rate) / rate
                )
            ).astype(np.float32)
            sf.write(args.output_dir / "source.wav", samples, rate, subtype="FLOAT")
            (args.output_dir / "source.webm").write_bytes(b"downloaded media")
        elif log.name == "enhance.log":
            enhance_file(
                Path(command[3]),
                Path(command[4]),
                lambda samples, rate: samples,
                start=args.start,
                duration=args.duration,
            )
            if failure == "lineage":
                sidecar = Path(command[4]).with_suffix(".wav.json")
                lineage = json.loads(sidecar.read_text())
                lineage["source_start_sample"] += 1
                sidecar.write_text(json.dumps(lineage))
        elif log.name == "prepare.log":
            prepare_candidates(Path(command[4]), Path(command[5]), train_only=True)
        elif log.name == "transcribe.log":
            transcribe_release(
                Path(command[4]),
                Path(command[5]),
                transcriber=lambda path, start, duration: ""
                if failure == "empty_asr"
                else "你吃飽了嗎？",
            )

    monkeypatch.setattr(build_video, "execute", fake_execute)
    if failure is not None:
        with pytest.raises(ValueError):
            build_video.build(args)
        report = json.loads((args.output_dir / "run-report.json").read_text())
        assert report["state"] == "failed" and "release" not in report
        return
    release = build_video.build(args)
    report = json.loads((args.output_dir / "run-report.json").read_text())
    source = json.loads((args.output_dir / "sources.jsonl").read_text())
    assert report["state"] == "weak_labels_ready"
    assert source["source_interval"] == {
        "start_sample": 30000,
        "num_samples": 48000,
        "sample_rate": 24000,
    }
    assert source["license"] == "unknown"
    assert source["original_url"] == args.url
    assert source["audio_variant"] == "enhanced"
    assert json.loads((release / "metadata.jsonl").read_text())["label_type"] == "asr_pseudo"
    assert len(set(deadlines)) == 1
    assert "--download-sections" not in sum(commands, [])
    assert "--ignore-config" in commands[0] and "--no-playlist" in commands[1]
    assert "--max-filesize" in commands[1]
    with pytest.raises(FileExistsError):
        build_video.build(args)


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/playlist?list=abc",
        "https://www.youtube.com/watch?v=abcdefghijk&list=abc",
        "https://other.example/abcdefghijk",
    ],
)
def test_reject_non_single_video(tmp_path: Path, url: str) -> None:
    args = options(tmp_path)
    args.url = url
    with pytest.raises(ValueError, match="watch URL"):
        build_video.build(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("duration", 21),
        ("max_source_seconds", 301),
        ("timeout", float("inf")),
        ("start", -1),
        ("train_only", False),
    ],
)
def test_reject_unbounded_options(tmp_path: Path, field: str, value: object) -> None:
    args = options(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        build_video.build(args)


def test_stage_failure_is_not_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = options(tmp_path)
    monkeypatch.setattr(build_video.shutil, "which", lambda name: "/usr/bin/node")

    def fail(command: list[str], log: Path, deadline: float) -> None:
        raise TimeoutError("deadline expired")

    monkeypatch.setattr(build_video, "execute", fail)
    with pytest.raises(TimeoutError):
        build_video.build(args)
    report = json.loads((args.output_dir / "run-report.json").read_text())
    assert report["state"] == "failed" and report["stage"] == "metadata"
    assert "release" not in report


def test_timeout_kills_external_descendants(tmp_path: Path) -> None:
    pidfile = tmp_path / "child.pid"
    script = (
        "import os,time; from pathlib import Path; pid=os.fork(); "
        f"Path({str(pidfile)!r}).write_text(str(os.getpid())) if pid == 0 else None; "
        "time.sleep(30)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        build_video.execute(
            [sys.executable, "-c", script], tmp_path / "stage.log", time.monotonic() + 0.3
        )
    pid = int(pidfile.read_text())
    status = Path(f"/proc/{pid}/stat")
    for _ in range(30):
        if not status.exists() or status.read_text().split()[2] == "Z":
            break
        time.sleep(0.01)
    assert not status.exists() or status.read_text().split()[2] == "Z"
