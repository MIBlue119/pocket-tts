from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from training.taigi.audio import sha256_file
from training.taigi.merge import merge_releases
from training.taigi.schema import read_jsonl
from training.taigi.transcriber import BREEZE_ADAPTER_VERSION, BREEZE_MODEL, BREEZE_REVISION


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _wav(path: Path, seed: bytes) -> None:
    samples = bytes(seed[index % len(seed)] for index in range(16000))
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(samples)


def _release(
    root: Path, name: str, rows: list[dict[str, object]], *, adapter: str = BREEZE_ADAPTER_VERSION
) -> Path:
    release = root / name
    audio_dir = release / "audio"
    audio_dir.mkdir(parents=True)
    metadata = []
    manifests: dict[str, list[dict[str, object]]] = {"train": [], "valid": []}
    for index, spec in enumerate(rows):
        supplied_audio = spec.get("audio_bytes", f"audio-{name}-{index}".encode())
        assert isinstance(supplied_audio, bytes)
        audio_bytes = supplied_audio
        audio = audio_dir / f"{index}.wav"
        _wav(audio, audio_bytes)
        digest = sha256_file(audio)
        split = str(spec.get("split", "train"))
        text = str(spec.get("text", "台語弱標註"))
        interval = spec.get(
            "source_interval", {"start_sample": 0, "num_samples": 24000, "sample_rate": 24000}
        )
        source_metadata = {
            "source_audio_sha256": spec.get("original_hash", f"original-{name}-{index}"),
            "source_interval": interval,
        }
        row = {
            "utterance_id": spec.get("utterance_id", f"{name}-{index}"),
            "source_id": spec.get("source_id", f"source-{name}-{index}"),
            "group_id": spec.get("group_id", f"group-{name}-{index}"),
            "published_audio_path": str(audio),
            "audio_sha256": digest,
            "start": spec.get("start", 0.0),
            "duration": spec.get("duration", 1.0),
            "input_text_zh": text,
            "split": split,
            "source_metadata": source_metadata,
            "label_type": "asr_pseudo",
            "label_provenance": "Breeze-ASR-26",
            "label_model": BREEZE_MODEL,
            "label_model_revision": BREEZE_REVISION,
            "label_adapter_version": adapter,
            "verification": "unverified_weak_label",
        }
        metadata.append(row)
        manifests[split].append(
            {
                "path": str(audio),
                "start": row["start"],
                "duration": row["duration"],
                "transcript": text,
            }
        )
    _jsonl(release / "metadata.jsonl", metadata)
    _jsonl(release / "train.jsonl", manifests["train"])
    _jsonl(release / "valid.jsonl", manifests["valid"])
    report = {
        "schema_version": "zh-taigi-semantic-pair/v1",
        "release_id": name,
        "accepted_weak_label_count": len(metadata),
        "model": BREEZE_MODEL,
        "revision": BREEZE_REVISION,
        "label_adapter_version": adapter,
        "verified_labels": 0,
    }
    (release / "report.json").write_text(json.dumps(report))
    return release


def test_merge_deduplicates_original_interval_and_publishes_actual_hashes(tmp_path: Path) -> None:
    common = {"original_hash": "a" * 64, "source_id": "video", "start": 0.25, "duration": 0.5}
    first = _release(
        tmp_path, "first", [{**common, "group_id": "one", "audio_bytes": b"enhanced-a"}]
    )
    second = _release(
        tmp_path, "second", [{**common, "group_id": "two", "audio_bytes": b"enhanced-b"}]
    )

    merged = merge_releases([first, second], tmp_path / "merged", train_only=True)

    metadata = read_jsonl(merged / "metadata.jsonl")
    report = json.loads((merged / "report.json").read_text())
    assert len(metadata) == 1
    assert report["deduplicated_count"] == 1
    assert report["verified_labels"] == 0
    assert "no quality claim" in report["label_provenance"]
    assert sha256_file(Path(metadata[0]["published_audio_path"])) == metadata[0]["audio_sha256"]
    for name, digest in report["manifest_sha256"].items():
        assert sha256_file(merged / name) == digest


def test_merge_rejects_bad_input_integrity_and_duplicate_conflicts(tmp_path: Path) -> None:
    base = {"original_hash": "b" * 64, "source_id": "video", "group_id": "same"}
    first = _release(tmp_path, "first", [{**base, "text": "甲"}])
    old = _release(tmp_path, "old", [{"group_id": "old"}], adapter="external_or_test")
    with pytest.raises(ValueError, match="label_adapter_version"):
        merge_releases([old], tmp_path / "merged-old", train_only=True)

    second = _release(tmp_path, "second", [{**base, "text": "乙"}])
    with pytest.raises(ValueError, match="conflicting text or provenance"):
        merge_releases([first, second], tmp_path / "merged-conflict", train_only=True)

    Path(read_jsonl(first / "metadata.jsonl")[0]["published_audio_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="audio hash mismatch"):
        merge_releases([first], tmp_path / "merged-corrupt", train_only=True)


@pytest.mark.parametrize(
    "spec,error",
    [
        ({"start": -0.1}, "invalid interval"),
        ({"duration": 0.0}, "invalid interval"),
        ({"start": float("nan")}, "invalid interval"),
        ({"start": 0.75, "duration": 0.5}, "audio bounds"),
    ],
)
def test_merge_rejects_invalid_or_out_of_bounds_intervals(
    tmp_path: Path, spec: dict[str, object], error: str
) -> None:
    release = _release(tmp_path, "invalid", [{"group_id": "only", **spec}])
    with pytest.raises(ValueError, match=error):
        merge_releases([release], tmp_path / "merged", train_only=True)


def test_related_groups_cannot_cross_split(tmp_path: Path) -> None:
    shared_original = "c" * 64
    release = _release(
        tmp_path,
        "input",
        [
            {"group_id": "related-a", "source_id": "same-video", "original_hash": shared_original},
            {
                "group_id": "related-b",
                "source_id": "same-video",
                "original_hash": shared_original,
                "start": 0.5,
                "duration": 0.5,
            },
            {"group_id": "independent", "source_id": "other-video", "original_hash": "d" * 64},
        ],
    )

    merged = merge_releases(
        [release], tmp_path / "merged", valid_groups={"related-b"}, train_only=False
    )
    rows = read_jsonl(merged / "metadata.jsonl")
    splits = {row["group_id"]: row["split"] for row in rows}
    assert splits == {"independent": "train", "related-a": "valid", "related-b": "valid"}


def test_duplicate_bridge_still_unites_groups_before_deduplication(tmp_path: Path) -> None:
    duplicate = {
        "source_id": "source-x",
        "original_hash": "e" * 64,
        "audio_bytes": b"same-enhancement",
    }
    release = _release(
        tmp_path,
        "bridge",
        [
            {**duplicate, "group_id": "group-a"},
            {**duplicate, "group_id": "group-b"},
            {
                "group_id": "group-b",
                "source_id": "source-y",
                "original_hash": "f" * 64,
                "audio_bytes": b"source-y",
            },
            {"group_id": "independent", "source_id": "source-z", "audio_bytes": b"source-z"},
        ],
    )

    merged = merge_releases([release], tmp_path / "merged", valid_groups={"group-a"})
    splits = {row["group_id"]: row["split"] for row in read_jsonl(merged / "metadata.jsonl")}
    assert splits == {"group-a": "valid", "group-b": "valid", "independent": "train"}


def test_same_parent_utterance_id_for_distinct_video_chunks_gets_unique_merged_ids(
    tmp_path: Path,
) -> None:
    base = {
        "group_id": "video",
        "source_id": "video-id",
        "utterance_id": "video-id-000000",
        "original_hash": "1" * 64,
    }
    first = _release(
        tmp_path,
        "chunk-one",
        [
            {
                **base,
                "source_interval": {"start_sample": 0, "num_samples": 24000, "sample_rate": 24000},
            }
        ],
    )
    second = _release(
        tmp_path,
        "chunk-two",
        [
            {
                **base,
                "source_interval": {
                    "start_sample": 480000,
                    "num_samples": 24000,
                    "sample_rate": 24000,
                },
            }
        ],
    )

    merged = merge_releases([first, second], tmp_path / "merged", train_only=True)
    rows = read_jsonl(merged / "metadata.jsonl")
    assert len(rows) == 2
    assert {row["parent_utterance_id"] for row in rows} == {"video-id-000000"}
    assert len({row["utterance_id"] for row in rows}) == 2
    assert all(row["utterance_id"].startswith("merged-") for row in rows)


def test_reuse_validates_manifests_and_audio(tmp_path: Path) -> None:
    source = _release(tmp_path, "input", [{"group_id": "only"}])
    merged = merge_releases([source], tmp_path / "merged", train_only=True)
    assert merge_releases([source], tmp_path / "merged", train_only=True) == merged

    (merged / "train.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="manifest failed integrity"):
        merge_releases([source], tmp_path / "merged", train_only=True)


def test_reuse_rejects_valid_json_report_tampering(tmp_path: Path) -> None:
    source = _release(tmp_path, "input", [{"group_id": "only"}])
    merged = merge_releases([source], tmp_path / "merged", train_only=True)
    report_path = merged / "report.json"
    report = json.loads(report_path.read_text())
    report["accepted_weak_label_count"] = 99
    report_path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="report failed integrity"):
        merge_releases([source], tmp_path / "merged", train_only=True)


def test_reuse_rejects_compact_manifest_tampering_even_when_report_hash_is_updated(
    tmp_path: Path,
) -> None:
    source = _release(tmp_path, "input", [{"group_id": "only", "text": "正確文字"}])
    merged = merge_releases([source], tmp_path / "merged", train_only=True)
    train_path = merged / "train.jsonl"
    train = read_jsonl(train_path)
    train[0]["transcript"] = "竄改文字"
    _jsonl(train_path, train)
    report_path = merged / "report.json"
    report = json.loads(report_path.read_text())
    report["manifest_sha256"]["train.jsonl"] = sha256_file(train_path)
    report_path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="train manifest is not the intended content"):
        merge_releases([source], tmp_path / "merged", train_only=True)
