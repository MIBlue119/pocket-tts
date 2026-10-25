import json
import sys
import types
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from training.taigi.cli import main
from training.taigi.audio import sha256_file
from training.taigi.pipeline import prepare_candidates, transcribe_release
from training.taigi.transcriber import BREEZE_ADAPTER_VERSION, BREEZE_REVISION, breeze_transcriber


def _wav(
    path: Path, parts: list[tuple[float, float]], sample_rate: int = 16000, frequency: float = 220
) -> None:
    samples = []
    for duration, amplitude in parts:
        count = round(duration * sample_rate)
        time = np.arange(count) / sample_rate
        samples.append(amplitude * np.sin(2 * np.pi * frequency * time))
    audio = np.concatenate(samples)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((audio * 32767).astype("<i2").tobytes())


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_prepare_segments_silence_and_keeps_groups_in_one_split(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    _wav(first, [(1.0, 0.2), (0.5, 0.0), (1.0, 0.2)])
    _wav(second, [(1.0, 0.2)])
    source_manifest = tmp_path / "sources.jsonl"
    _jsonl(
        source_manifest,
        [
            {
                "audio_path": str(first),
                "source_id": "a",
                "language": "nan-tw",
                "group_id": "show-a",
                "license": "CC-BY-4.0",
                "repo": "example/taigi",
                "revision": "abc123",
            },
            {
                "audio_path": str(second),
                "source_id": "b",
                "language": "nan-tw",
                "group_id": "show-b",
            },
        ],
    )

    candidates = prepare_candidates(source_manifest, tmp_path / "work", min_duration=0.5)
    rows = _rows(candidates)

    assert len(rows) == 3
    assert {row["language_provenance"] for row in rows} == {"source_declared"}
    assert {row["audio_variant"] for row in rows} == {"raw"}
    assert len({row["split"] for row in rows if row["group_id"] == "show-a"}) == 1
    assert {row["split"] for row in rows} == {"train", "valid"}
    assert all(float(row["duration"]) <= 20 for row in rows)
    assert all(len(str(row["audio_sha256"])) == 64 for row in rows)
    first_metadata = next(row["source_metadata"] for row in rows if row["source_id"] == "a")
    assert first_metadata == {"license": "CC-BY-4.0", "repo": "example/taigi", "revision": "abc123"}


def test_prepare_keeps_duplicate_audio_in_one_split(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    duplicate = tmp_path / "duplicate.wav"
    distinct = tmp_path / "distinct.wav"
    _wav(first, [(1.0, 0.2)])
    duplicate.write_bytes(first.read_bytes())
    _wav(distinct, [(1.2, 0.15)])
    manifest = tmp_path / "sources.jsonl"
    _jsonl(
        manifest,
        [
            {"audio_path": str(first), "source_id": "a", "group_id": "a", "language": "nan-tw"},
            {"audio_path": str(duplicate), "source_id": "b", "group_id": "b", "language": "nan-tw"},
            {"audio_path": str(distinct), "source_id": "c", "group_id": "c", "language": "nan-tw"},
        ],
    )

    rows = _rows(prepare_candidates(manifest, tmp_path / "work"))
    splits = {row["source_id"]: row["split"] for row in rows}

    assert splits["a"] == splits["b"]
    assert splits["a"] != splits["c"]


def test_prepare_requires_two_groups_unless_train_only(tmp_path: Path) -> None:
    audio = tmp_path / "one.wav"
    _wav(audio, [(1.0, 0.2)])
    source_manifest = tmp_path / "sources.jsonl"
    _jsonl(source_manifest, [{"audio_path": str(audio), "source_id": "one", "language": "nan-tw"}])

    with pytest.raises(ValueError, match="two source groups"):
        prepare_candidates(source_manifest, tmp_path / "normal")

    candidates = prepare_candidates(source_manifest, tmp_path / "smoke", train_only=True)
    assert _rows(candidates)[0]["split"] == "train"


def test_manifest_rejects_unknown_language_instead_of_classifying_it(tmp_path: Path) -> None:
    audio = tmp_path / "one.wav"
    _wav(audio, [(1.0, 0.2)])
    source_manifest = tmp_path / "sources.jsonl"
    _jsonl(source_manifest, [{"audio_path": str(audio), "source_id": "one", "language": "unknown"}])

    with pytest.raises(ValueError, match="declare language='nan-tw'"):
        prepare_candidates(source_manifest, tmp_path / "work", train_only=True)


def test_prepare_quarantines_long_interval_without_inventing_a_cut(tmp_path: Path) -> None:
    audio = tmp_path / "long.wav"
    _wav(audio, [(2.0, 0.2)])
    source_manifest = tmp_path / "sources.jsonl"
    _jsonl(source_manifest, [{"audio_path": str(audio), "source_id": "long", "language": "nan-tw"}])

    candidates = prepare_candidates(
        source_manifest, tmp_path / "work", train_only=True, max_duration=1.0
    )

    assert _rows(candidates) == []
    quarantine = _rows(candidates.parent / "quarantine.jsonl")
    assert len(quarantine) == 1
    assert quarantine[0]["quarantine_reason"] == "too_long"
    assert float(quarantine[0]["duration"]) == pytest.approx(2.0, abs=0.02)


def test_enhanced_source_keeps_variant_and_lineage_in_release(tmp_path: Path) -> None:
    audio = tmp_path / "enhanced.wav"
    _wav(audio, [(1.0, 0.2)])
    source_manifest = tmp_path / "sources.jsonl"
    _jsonl(
        source_manifest,
        [
            {
                "audio_path": str(audio),
                "source_id": "reuse-one",
                "group_id": "original-one",
                "language": "nan-tw",
                "audio_variant": "enhanced",
                "raw_sha256": "1" * 64,
                "enhancer": "nvidia-reuse",
                "enhancer_revision": "pinned-revision",
            }
        ],
    )
    candidates = prepare_candidates(source_manifest, tmp_path / "work", train_only=True)
    candidate = _rows(candidates)[0]

    assert candidate["audio_variant"] == "enhanced"
    assert candidate["source_metadata"] == {
        "raw_sha256": "1" * 64,
        "enhancer": "nvidia-reuse",
        "enhancer_revision": "pinned-revision",
    }

    release = transcribe_release(
        candidates, tmp_path / "releases", transcriber=lambda path, start, duration: "中文弱標註"
    )
    metadata = _rows(release / "metadata.jsonl")[0]
    assert metadata["audio_variant"] == "enhanced"
    assert metadata["source_metadata"]["raw_sha256"] == "1" * 64


def test_transcribe_writes_immutable_compact_release_and_quarantine(tmp_path: Path) -> None:
    audio = tmp_path / "one.wav"
    _wav(audio, [(2.0, 0.2)])
    audio_hash = sha256_file(audio)
    candidates = tmp_path / "candidates.jsonl"
    base = {
        "source_id": "one",
        "group_id": "one",
        "audio_path": str(audio),
        "start": 0.0,
        "duration": 1.0,
        "audio_sha256": audio_hash,
        "language": "nan-tw",
        "language_provenance": "source_declared",
        "audio_variant": "raw",
        "split": "train",
        "status": "candidate",
        "quarantine_reason": None,
    }
    _jsonl(
        candidates,
        [{**base, "utterance_id": "one-0"}, {**base, "utterance_id": "one-1", "start": 1.0}],
    )

    def fake_transcriber(path: Path, start: float, duration: float) -> str:
        assert path == audio
        assert duration == 1.0
        return "食飽未？" if start == 0 else ""

    release = transcribe_release(candidates, tmp_path / "releases", transcriber=fake_transcriber)
    repeated = transcribe_release(candidates, tmp_path / "releases", transcriber=fake_transcriber)

    assert release == repeated
    train_row = _rows(release / "train.jsonl")[0]
    published_audio = Path(train_row["path"])
    assert train_row == {
        "duration": 1.0,
        "path": str(release / "audio" / f"{audio_hash}.wav"),
        "start": 0.0,
        "transcript": "食飽未？",
    }
    assert published_audio.read_bytes() == audio.read_bytes()
    assert sha256_file(published_audio) == audio_hash
    assert _rows(release / "valid.jsonl") == []
    metadata = _rows(release / "metadata.jsonl")[0]
    assert metadata["verification"] == "unverified_weak_label"
    assert metadata["label_model_revision"] == BREEZE_REVISION
    assert _rows(release / "quarantine.jsonl")[0]["quarantine_reason"] == "empty_transcript"


def test_transcribe_rejects_changed_audio_before_calling_asr(tmp_path: Path) -> None:
    audio = tmp_path / "changed.wav"
    _wav(audio, [(1.0, 0.2)])
    candidates = tmp_path / "candidates.jsonl"
    _jsonl(
        candidates,
        [
            {
                "utterance_id": "changed-0",
                "source_id": "changed",
                "group_id": "changed",
                "audio_path": str(audio),
                "start": 0.0,
                "duration": 1.0,
                "audio_sha256": "0" * 64,
                "split": "train",
            }
        ],
    )
    called = False

    def must_not_run(path: Path, start: float, duration: float) -> str:
        nonlocal called
        called = True
        return "wrong"

    with pytest.raises(ValueError, match="hash mismatch"):
        transcribe_release(candidates, tmp_path / "releases", transcriber=must_not_run)
    assert not called


def test_breeze_transcriber_resamples_exact_48khz_interval_before_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "48khz.wav"
    _wav(audio, [(1.0, 0.0), (20.0, 0.2), (1.0, 0.0)], sample_rate=48000, frequency=400)
    observed: dict[str, Any] = {}

    class FakePipe:
        feature_extractor = types.SimpleNamespace(sampling_rate=16000)

        def __call__(
            self: Any, inputs: dict[str, Any], generate_kwargs: dict[str, str]
        ) -> dict[str, str]:
            observed.update(inputs)
            observed["generate_kwargs"] = generate_kwargs
            return {"text": "測試"}

    fake_transformers = types.ModuleType("transformers")
    setattr(fake_transformers, "pipeline", lambda *args, **kwargs: FakePipe())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    transcribe = breeze_transcriber("cpu")
    text = transcribe(audio, 1.0, 20.0)

    samples = observed["array"]
    frequencies = np.fft.rfftfreq(len(samples), d=1 / 16000)
    dominant_frequency = frequencies[int(np.argmax(np.abs(np.fft.rfft(samples))))]
    assert text == "測試"
    assert transcribe.adapter_version == BREEZE_ADAPTER_VERSION
    assert observed["sampling_rate"] == 16000
    assert len(samples) == 320000
    assert dominant_frequency == pytest.approx(400, abs=1)


def test_cli_prepare_supports_train_only_smoke(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    audio = tmp_path / "one.wav"
    _wav(audio, [(1.0, 0.2)])
    manifest = tmp_path / "sources.jsonl"
    _jsonl(manifest, [{"audio_path": str(audio), "source_id": "one", "language": "nan-tw"}])

    assert main(["prepare", str(manifest), str(tmp_path / "work"), "--train-only"]) == 0
    assert capsys.readouterr().out.strip().endswith("candidates.jsonl")
