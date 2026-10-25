import json
from pathlib import Path
from typing import Any
from collections.abc import Callable

import numpy as np
import pytest
import soundfile as sf

from training.taigi import corpus_worker as worker
from training.taigi.enhance import ReusePredictor


def spec(tmp_path: Path, seconds: float = 2.125, silent: bool = False) -> dict[str, Any]:
    raw = tmp_path / "raw.wav"
    samples = np.zeros(round(seconds * 8000), dtype=np.float32)
    if not silent:
        samples = np.random.default_rng(0).normal(0, 0.05, len(samples)).astype("float32")
    sf.write(raw, samples, 8000, subtype="FLOAT")
    return dict(
        raw_audio=str(raw),
        video_id="video",
        group_id="group",
        original_url="url",
        license="unknown",
        output_dir=str(tmp_path / "out"),
        cache_dir=None,
        device="cpu",
    )


def predictor(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls = []
    monkeypatch.setattr(
        worker,
        "load_predictor",
        lambda *args: calls.append(1)
        or ReusePredictor(lambda samples, rate: samples, {"weights": "test"}),
    )
    return calls


def test_exact_coverage_resume_and_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = spec(tmp_path)
    calls = predictor(monkeypatch)
    monkeypatch.setattr(worker, "MAX_CHUNK_SECONDS", 1)
    report = worker.enhance(data)
    assert report["total_frames"] == report["covered_frames"] == 17000
    assert [(c["start_sample"], c["num_samples"]) for c in report["chunks"]] == [
        (0, 8000),
        (8000, 8000),
        (16000, 1000),
    ]
    assert len(calls) == 1
    assert worker.enhance(data) == report
    assert len(calls) == 1
    Path(report["chunks"][0]["audio_path"]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="integrity"):
        worker.enhance(data)


def test_silence_is_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = spec(tmp_path, silent=True)
    predictor(monkeypatch)
    worker.enhance(data)
    monkeypatch.setattr(worker, "breeze_transcriber", lambda device: pytest.fail())
    result = worker.transcribe(data)
    assert result["state"] == "no_candidates"
    assert result["quarantine_count"] == 1
    assert worker.transcribe(data) == result


def test_asr_cached_after_interruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = spec(tmp_path, seconds=4)
    predictor(monkeypatch)
    monkeypatch.setattr(worker, "MAX_CHUNK_SECONDS", 2)
    worker.enhance(data)
    calls = []
    loads = []

    def load(device: str) -> Callable[[Path, float, float], str]:
        loads.append(1)

        def transcriber(path: Path, start: float, duration: float) -> str:
            calls.append(path)
            if len(calls) == 2:
                raise RuntimeError("interrupted")
            return "中文"

        setattr(transcriber, "target_sample_rate", 16000)
        return transcriber

    monkeypatch.setattr(worker, "breeze_transcriber", load)
    with pytest.raises(RuntimeError, match="interrupted"):
        worker.transcribe(data)
    result = worker.transcribe(data)
    assert len(loads) == 2 and len(calls) == 3
    assert result["accepted_weak_label_count"] == 2
    assert result["state"] == "complete"
    assert worker.transcribe(data) == result
    metadata = Path(result["release_path"]) / "metadata.jsonl"
    assert all(
        json.loads(line)["label_adapter_version"] == "breeze-resample/v2"
        for line in metadata.read_text().splitlines()
    )


def test_failed_chunk_promotion_preserves_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = spec(tmp_path)
    predictor(monkeypatch)
    original = Path.rename

    def fail(path: Path, target: Path) -> Path:
        if path.name.startswith(".chunk-"):
            raise OSError("promotion failed")
        return original(path, target)

    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(OSError, match="promotion failed"):
        worker.enhance(data)
    output = Path(data["output_dir"])
    assert not (output / "enhancement-report.json").exists()
    assert list((output / "chunks").glob(".chunk-*/audio.wav"))
    monkeypatch.setattr(Path, "rename", original)
    assert worker.enhance(data)["state"] == "complete"


def test_enhancement_interruption_reuses_complete_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = spec(tmp_path, seconds=4.125)
    loads = predictor(monkeypatch)
    monkeypatch.setattr(worker, "MAX_CHUNK_SECONDS", 2)
    original = worker.enhance_file
    calls = []

    def interrupted(
        source: Path, output: Path, predict: ReusePredictor, *, start: float, duration: float
    ) -> Path:
        calls.append(start)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return original(source, output, predict, start=start, duration=duration)

    monkeypatch.setattr(worker, "enhance_file", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        worker.enhance(data)
    report = worker.enhance(data)
    assert calls == [0, 2, 2, 4]
    assert len(loads) == 2
    assert report["covered_frames"] == 33000


def test_empty_asr_and_failed_result_promotion_reuse_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = spec(tmp_path)
    predictor(monkeypatch)
    worker.enhance(data)
    calls = []

    class EmptyASR:
        target_sample_rate = 16000

        def __call__(self, *args: object) -> str:
            calls.append(1)
            return ""

    monkeypatch.setattr(worker, "breeze_transcriber", lambda device: EmptyASR())
    original = worker._write

    def fail(path: Path, value: dict[str, Any]) -> None:
        if path.name == "result.json":
            raise OSError("publication failed")
        return original(path, value)

    monkeypatch.setattr(worker, "_write", fail)
    with pytest.raises(OSError, match="publication failed"):
        worker.transcribe(data)
    assert not (Path(data["output_dir"]) / "result.json").exists()
    monkeypatch.setattr(worker, "_write", original)
    result = worker.transcribe(data)
    assert result["state"] == "no_candidates"
    assert result["quarantine_count"] == 1
    assert result["release_path"] is not None
    assert calls == [1]


def test_corrupt_transcript_cache_refuses_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = spec(tmp_path)
    predictor(monkeypatch)
    worker.enhance(data)

    class ASR:
        target_sample_rate = 16000

        def __call__(self, *args: object) -> str:
            return "文字"

    monkeypatch.setattr(worker, "breeze_transcriber", lambda device: ASR())
    original = worker._write

    def fail_result(path: Path, value: dict[str, Any]) -> None:
        if path.name == "result.json":
            raise OSError("interrupted")
        original(path, value)

    monkeypatch.setattr(worker, "_write", fail_result)
    with pytest.raises(OSError):
        worker.transcribe(data)
    cache_path = next((Path(data["output_dir"]) / "transcript-cache").glob("*.json"))
    cached = json.loads(cache_path.read_text())
    cached["text"] = "changed"
    cache_path.write_text(json.dumps(cached))
    monkeypatch.setattr(worker, "_write", original)
    with pytest.raises(ValueError, match="cache integrity"):
        worker.transcribe(data)


def test_candidates_preserve_canonical_original_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from training.taigi.merge import _original_interval
    from training.taigi.schema import read_jsonl

    data = spec(tmp_path, seconds=4)
    predictor(monkeypatch)
    monkeypatch.setattr(worker, "MAX_CHUNK_SECONDS", 2)
    report = worker.enhance(data)
    rows = read_jsonl(Path(data["output_dir"]) / "candidates" / "candidates.jsonl")
    assert len(rows) == 2
    for index, row in enumerate(rows):
        metadata = row["source_metadata"]
        assert metadata["original_video_id"] == "video"
        assert metadata["source_audio_sha256"] == report["raw_audio_sha256"]
        assert metadata["source_interval"] == dict(
            start_sample=index * 16000, num_samples=16000, sample_rate=8000
        )
        assert _original_interval(row) == (report["raw_audio_sha256"], index * 16000, 16000, 8000)


def test_changed_stage_code_refuses_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = spec(tmp_path, silent=True)
    predictor(monkeypatch)
    worker.enhance(data)
    worker.transcribe(data)
    original = worker.sha256_file

    def changed_hash(path: Path) -> str:
        if path.suffix == ".py":
            return "a" * 64
        return original(path)

    monkeypatch.setattr(worker, "sha256_file", changed_hash)
    with pytest.raises(ValueError, match="identity integrity"):
        worker.enhance(data)
    with pytest.raises(ValueError, match="identity integrity"):
        worker.transcribe(data)
