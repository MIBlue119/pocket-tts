import json
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from training.taigi.cli import main
from training.taigi.pipeline import prepare_candidates, transcribe_release
from training.taigi.transcriber import BREEZE_REVISION


def _wav(path: Path, parts: list[tuple[float, float]], sample_rate: int = 16000) -> None:
    samples = []
    for duration, amplitude in parts:
        count = round(duration * sample_rate)
        time = np.arange(count) / sample_rate
        samples.append(amplitude * np.sin(2 * np.pi * 220 * time))
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


def test_transcribe_writes_immutable_compact_release_and_quarantine(tmp_path: Path) -> None:
    audio = tmp_path / "one.wav"
    _wav(audio, [(2.0, 0.2)])
    candidates = tmp_path / "candidates.jsonl"
    base = {
        "source_id": "one",
        "group_id": "one",
        "audio_path": str(audio),
        "start": 0.0,
        "duration": 1.0,
        "audio_sha256": "a" * 64,
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
    assert _rows(release / "train.jsonl") == [
        {"duration": 1.0, "path": str(audio), "start": 0.0, "transcript": "食飽未？"}
    ]
    assert _rows(release / "valid.jsonl") == []
    metadata = _rows(release / "metadata.jsonl")[0]
    assert metadata["verification"] == "unverified_weak_label"
    assert metadata["label_model_revision"] == BREEZE_REVISION
    assert _rows(release / "quarantine.jsonl")[0]["quarantine_reason"] == "empty_transcript"


def test_cli_prepare_supports_train_only_smoke(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    audio = tmp_path / "one.wav"
    _wav(audio, [(1.0, 0.2)])
    manifest = tmp_path / "sources.jsonl"
    _jsonl(manifest, [{"audio_path": str(audio), "source_id": "one", "language": "nan-tw"}])

    assert main(["prepare", str(manifest), str(tmp_path / "work"), "--train-only"]) == 0
    assert capsys.readouterr().out.strip().endswith("candidates.jsonl")
