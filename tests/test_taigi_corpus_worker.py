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


def batch_rows(tmp_path: Path, count: int = 7) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        path = tmp_path / f"batch-{index}.wav"
        sf.write(path, np.full(800 + index * 80, 0.01 * index, dtype=np.float32), 8000)
        rows.append(
            {
                "audio_path": str(path),
                "audio_sha256": worker.sha256_file(path),
                "start": 0.0,
                "duration": 0.1 + index * 0.01,
            }
        )
    return rows


def test_batch_prefill_caches_order_duplicates_short_batch_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = batch_rows(tmp_path)
    loads = []
    batches = []
    singles = []

    class ASR:
        target_sample_rate = 16000

        def __call__(self, path: Path, start: float, duration: float) -> str:
            singles.append(path.name)
            return path.stem

        def transcribe_batch(self, intervals: list[tuple[Path, float, float]]) -> list[str]:
            batches.append([path.name for path, _, _ in intervals])
            return [path.stem for path, _, _ in intervals]

    monkeypatch.setattr(worker, "breeze_transcriber", lambda _: loads.append(1) or ASR())
    cached = worker.CachedTranscriber(tmp_path, "cpu", 4)
    assert cached(Path(rows[0]["audio_path"]), 0.0, rows[0]["duration"]) == "batch-0"
    cached.prefill(rows + rows)
    assert len(loads) == 1
    assert singles == ["batch-0.wav"]
    assert batches == [[f"batch-{i}.wav" for i in range(1, 5)], ["batch-5.wav", "batch-6.wav"]]
    resumed = worker.CachedTranscriber(tmp_path, "cpu", 4)
    resumed.prefill(rows)
    assert [resumed(Path(row["audio_path"]), 0.0, row["duration"]) for row in rows] == [
        f"batch-{i}" for i in range(7)
    ]
    assert len(loads) == 1
    cache = next((tmp_path / "transcript-cache").glob("*.json"))
    record = json.loads(cache.read_text())
    assert record["identity"]["asr_batch_size"] == 4
    record["text"] = "corrupt"
    cache.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="cache integrity"):
        resumed.prefill(rows)
    assert len(loads) == 1


def test_failed_batch_preserves_prior_batch_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = batch_rows(tmp_path)
    calls = []
    loads = []
    fail = True

    class ASR:
        target_sample_rate = 16000

        def transcribe_batch(self, intervals: list[tuple[Path, float, float]]) -> list[str]:
            calls.append([path.name for path, _, _ in intervals])
            if fail and len(calls) == 2:
                raise RuntimeError("second batch failed")
            return [path.stem for path, _, _ in intervals]

    monkeypatch.setattr(worker, "breeze_transcriber", lambda _: loads.append(1) or ASR())
    with pytest.raises(RuntimeError, match="second batch failed"):
        worker.CachedTranscriber(tmp_path, "cpu", 4).prefill(rows)
    assert len(list((tmp_path / "transcript-cache").glob("*.json"))) == 4
    fail = False
    resumed = worker.CachedTranscriber(tmp_path, "cpu", 4)
    resumed.prefill(rows)
    assert len(loads) == 2
    assert calls == [
        [f"batch-{i}.wav" for i in range(4)],
        [f"batch-{i}.wav" for i in range(4, 7)],
        [f"batch-{i}.wav" for i in range(4, 7)],
    ]
    assert len(list((tmp_path / "transcript-cache").glob("*.json"))) == 7
    resumed.prefill(rows)
    assert len(calls) == 3


def test_batch_prefill_rejects_candidate_audio_hash_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = batch_rows(tmp_path, 1)
    rows[0]["audio_sha256"] = "bad"
    monkeypatch.setattr(worker, "breeze_transcriber", lambda _: pytest.fail())
    with pytest.raises(ValueError, match="audio hash mismatch"):
        worker.CachedTranscriber(tmp_path, "cpu", 4).prefill(rows)


def test_worker_prefills_opt_in_batch_then_release_reads_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = spec(tmp_path, seconds=4)
    data["asr_batch_size"] = 4
    predictor(monkeypatch)
    monkeypatch.setattr(worker, "MAX_CHUNK_SECONDS", 1)
    worker.enhance(data)
    batches = []
    loads = []

    class ASR:
        target_sample_rate = 16000

        def __call__(self, path: Path, start: float, duration: float) -> str:
            raise AssertionError("release must read prefilled per-row caches")

        def transcribe_batch(self, intervals: list[tuple[Path, float, float]]) -> list[str]:
            batches.append(intervals)
            return ["中文弱標註"] * len(intervals)

    monkeypatch.setattr(worker, "breeze_transcriber", lambda _: loads.append(1) or ASR())
    result = worker.transcribe(data)
    assert result["asr_batch_size"] == 4
    assert result["accepted_weak_label_count"] == 4
    assert len(loads) == len(batches) == 1
    assert len(batches[0]) == 4
    assert worker.transcribe(data) == result
    assert len(loads) == 1
    data["asr_batch_size"] = 1
    with pytest.raises(ValueError, match="identity integrity"):
        worker.transcribe(data)


def test_batch_prefill_groups_stable_durations_without_changing_lookup_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = batch_rows(tmp_path)
    # Equal durations keep their supplied order; output lookup keeps the manifest order.
    for row, duration in zip(rows, [0.09, 0.03, 0.06, 0.03, 0.08, 0.04, 0.07], strict=True):
        row["duration"] = duration
    original = [dict(row) for row in rows]
    batches = []

    class ASR:
        target_sample_rate = 16000

        def transcribe_batch(self, intervals: list[tuple[Path, float, float]]) -> list[str]:
            batches.append([path.name for path, _, _ in intervals])
            return [path.stem for path, _, _ in intervals]

        def __call__(self, path: Path, start: float, duration: float) -> str:
            raise AssertionError("every original-order lookup must use its cache")

    monkeypatch.setattr(worker, "breeze_transcriber", lambda _: ASR())
    cached = worker.CachedTranscriber(tmp_path, "cpu", 4)
    cached.prefill(rows)
    assert batches == [
        ["batch-1.wav", "batch-3.wav", "batch-5.wav", "batch-2.wav"],
        ["batch-6.wav", "batch-4.wav", "batch-0.wav"],
    ]
    assert rows == original
    assert [cached(Path(row["audio_path"]), 0.0, row["duration"]) for row in rows] == [
        f"batch-{i}" for i in range(7)
    ]
    cached.prefill(rows)
    assert len(batches) == 2
