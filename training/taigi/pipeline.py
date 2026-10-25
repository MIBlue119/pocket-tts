from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from training.taigi.acquire import acquire_url
from training.taigi.audio import load_audio, quality_reason, sha256_file, speech_intervals
from training.taigi.schema import SegmentRecord, SourceRecord, read_jsonl, write_jsonl
from training.taigi.transcriber import BREEZE_MODEL, BREEZE_REVISION, breeze_transcriber


def _parse_sources(path: Path, acquisition_dir: Path) -> list[SourceRecord]:
    sources: list[SourceRecord] = []
    for row in read_jsonl(path):
        language = row.get("language")
        if language != "nan-tw":
            raise ValueError(
                f"source {row.get('source_id', '<unknown>')} must declare language='nan-tw'"
            )
        source_id = str(row.get("source_id", "")).strip()
        if not source_id:
            raise ValueError("each source requires a non-empty source_id")
        audio_path = (
            Path(row["audio_path"]).expanduser().resolve() if row.get("audio_path") else None
        )
        url = str(row["url"]) if row.get("url") else None
        if (audio_path is None) == (url is None):
            raise ValueError(f"source {source_id} requires exactly one of audio_path or url")
        if url is not None:
            audio_path, video_id = acquire_url(url, acquisition_dir)
            if video_id not in source_id:
                source_id = f"{source_id}-{video_id}"
        assert audio_path is not None
        if not audio_path.is_file():
            raise ValueError(f"source audio does not exist: {audio_path}")
        suffix = audio_path.suffix.lower()
        if suffix not in {".wav", ".flac", ".mp3", ".m4a", ".ogg"}:
            raise ValueError(f"unsupported audio extension for {audio_path}")
        split = row.get("split")
        if split not in {None, "train", "valid"}:
            raise ValueError(f"source {source_id} has invalid split {split!r}")
        sources.append(
            SourceRecord(
                source_id=source_id,
                language="nan-tw",
                audio_path=audio_path,
                url=url,
                group_id=str(row.get("group_id") or source_id),
                split=split,
                audio_sha256=sha256_file(audio_path),
                source_metadata={
                    key: value
                    for key, value in row.items()
                    if key
                    not in {"audio_path", "url", "source_id", "language", "group_id", "split"}
                },
            )
        )
    if not sources:
        raise ValueError("source manifest is empty")
    return sources


def _assign_splits(
    sources: list[SourceRecord], train_only: bool, seed: int
) -> dict[str, Literal["train", "valid"]]:
    parent = {source.group: source.group for source in sources}

    def find(group: str) -> str:
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    digest_groups: dict[str, str] = {}
    for source in sources:
        previous_group = digest_groups.setdefault(source.audio_sha256, source.group)
        union(source.group, previous_group)

    explicit: dict[str, Literal["train", "valid"]] = {}
    groups = sorted({find(source.group) for source in sources})
    for source in sources:
        if source.split is not None:
            split_group = find(source.group)
            old = explicit.setdefault(split_group, source.split)
            if old != source.split:
                raise ValueError(
                    f"duplicate audio group {split_group} has conflicting supplied splits"
                )
    if train_only:
        if any(split == "valid" for split in explicit.values()):
            raise ValueError("--train-only conflicts with a supplied valid split")
        return {source.group: "train" for source in sources}
    if len(groups) < 2:
        raise ValueError(
            "at least two source groups are required; use --train-only for a smoke run"
        )
    unassigned = [group for group in groups if group not in explicit]
    random.Random(seed).shuffle(unassigned)
    assigned = dict(explicit)
    if "valid" not in assigned.values():
        if not unassigned:
            raise ValueError("supplied splits contain no valid group")
        assigned[unassigned.pop()] = "valid"
    for group in unassigned:
        assigned[group] = "train"
    if "train" not in assigned.values():
        candidate = next((group for group in groups if explicit.get(group) is None), None)
        if candidate is None:
            raise ValueError("supplied splits contain no train group")
        assigned[candidate] = "train"
    return {source.group: assigned[find(source.group)] for source in sources}


def prepare_candidates(
    source_manifest: Path,
    output_dir: Path,
    *,
    train_only: bool = False,
    seed: int = 0,
    silence_db: float = -42.0,
    min_silence: float = 0.35,
    min_duration: float = 0.8,
    max_duration: float = 20.0,
) -> Path:
    if min_duration <= 0 or max_duration < min_duration:
        raise ValueError("durations must satisfy 0 < min_duration <= max_duration")
    if min_silence <= 0:
        raise ValueError("min_silence must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = _parse_sources(source_manifest, output_dir / "downloads")
    splits = _assign_splits(sources, train_only, seed)
    candidates: list[SegmentRecord] = []
    quarantined: list[SegmentRecord] = []
    for source in sources:
        assert source.audio_path is not None
        audio = load_audio(source.audio_path)
        digest = source.audio_sha256
        intervals = speech_intervals(audio, silence_db, min_silence, min_duration)
        if not intervals:
            intervals = [(0.0, len(audio.samples) / audio.sample_rate)]
        for index, (start, duration) in enumerate(intervals):
            if duration < min_duration:
                reason = "too_short"
            elif duration > max_duration:
                reason = "too_long"
            else:
                reason = quality_reason(audio, start, duration)
            record = SegmentRecord(
                utterance_id=f"{source.source_id}-{index:06d}",
                source_id=source.source_id,
                group_id=source.group,
                audio_path=str(source.audio_path),
                start=round(start, 6),
                duration=round(duration, 6),
                audio_sha256=digest,
                split=splits[source.group],
                status="quarantined" if reason else "candidate",
                quarantine_reason=reason,
                source_metadata=source.source_metadata,
            )
            (quarantined if reason else candidates).append(record)
    write_jsonl(output_dir / "candidates.jsonl", candidates)
    write_jsonl(output_dir / "quarantine.jsonl", quarantined)
    report = {
        "schema_version": "taigi-candidates/v1",
        "candidate_count": len(candidates),
        "quarantine_count": len(quarantined),
        "quarantine_reasons": Counter(row.quarantine_reason for row in quarantined),
        "segmentation": {
            "method": "deterministic_rms_silence",
            "silence_db": silence_db,
            "min_silence": min_silence,
            "min_duration": min_duration,
            "max_duration": max_duration,
        },
    }
    (output_dir / "prepare-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir / "candidates.jsonl"


def transcribe_release(
    candidates_path: Path,
    releases_dir: Path,
    *,
    device: str = "cpu",
    transcriber: Callable[[Path, float, float], str] | None = None,
) -> Path:
    rows = read_jsonl(candidates_path)
    if not rows:
        raise ValueError("candidate manifest is empty")
    observed_hashes: dict[Path, str] = {}
    for row in rows:
        audio_path = Path(row["audio_path"]).resolve()
        expected_hash = str(row.get("audio_sha256", ""))
        if audio_path not in observed_hashes:
            observed_hashes[audio_path] = sha256_file(audio_path)
        actual_hash = observed_hashes[audio_path]
        if len(expected_hash) != 64 or actual_hash != expected_hash:
            raise ValueError(f"candidate audio hash mismatch: {audio_path}")
    transcribe = transcriber or breeze_transcriber(device)
    label_adapter_version = getattr(transcribe, "adapter_version", "external_or_test")
    label_target_sample_rate = getattr(transcribe, "target_sample_rate", None)
    accepted: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for row in rows:
        text = transcribe(
            Path(row["audio_path"]).resolve(), float(row["start"]), float(row["duration"])
        ).strip()
        enriched = {
            **row,
            "input_text_zh": text,
            "input_language": "zh-TW",
            "label_type": "asr_pseudo",
            "label_provenance": "Breeze-ASR-26",
            "label_model": BREEZE_MODEL,
            "label_model_revision": BREEZE_REVISION,
            "label_adapter_version": label_adapter_version,
            "label_target_sample_rate": label_target_sample_rate,
            "verification": "unverified_weak_label",
        }
        if not text:
            enriched["status"] = "quarantined"
            enriched["quarantine_reason"] = "empty_transcript"
            quarantine.append(enriched)
        else:
            enriched["status"] = "auto_accepted_candidate"
            accepted.append(enriched)
    release_input = json.dumps(accepted + quarantine, ensure_ascii=False, sort_keys=True).encode()
    release_id = hashlib.sha256(release_input).hexdigest()[:16]
    releases_dir = releases_dir.resolve()
    release = releases_dir / release_id
    if release.exists():
        for row in accepted:
            suffix = Path(row["audio_path"]).suffix.lower()
            published = release / "audio" / f"{row['audio_sha256']}{suffix}"
            if not published.is_file() or sha256_file(published) != row["audio_sha256"]:
                raise ValueError(f"existing release audio failed integrity check: {published}")
        return release
    temporary = releases_dir / f".{release_id}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        audio_dir = temporary / "audio"
        audio_dir.mkdir()
        published_paths: dict[tuple[str, str], str] = {}
        for row in accepted:
            source = Path(row["audio_path"]).resolve()
            suffix = source.suffix.lower()
            key = (row["audio_sha256"], suffix)
            final_audio = release / "audio" / f"{row['audio_sha256']}{suffix}"
            if key not in published_paths:
                temporary_audio = audio_dir / final_audio.name
                shutil.copyfile(source, temporary_audio)
                if sha256_file(temporary_audio) != row["audio_sha256"]:
                    raise ValueError(f"copied release audio failed integrity check: {source}")
                temporary_audio.chmod(0o444)
                published_paths[key] = str(final_audio)
            row["source_audio_path"] = row["audio_path"]
            row["published_audio_path"] = published_paths[key]
        for split in ("train", "valid"):
            compact = [
                {
                    "path": row["published_audio_path"],
                    "start": row["start"],
                    "duration": row["duration"],
                    "transcript": row["input_text_zh"],
                }
                for row in accepted
                if row["split"] == split
            ]
            write_jsonl(temporary / f"{split}.jsonl", compact)
        write_jsonl(temporary / "metadata.jsonl", accepted)
        write_jsonl(temporary / "quarantine.jsonl", quarantine)
        report = {
            "schema_version": "zh-taigi-semantic-pair/v1",
            "release_id": release_id,
            "accepted_weak_label_count": len(accepted),
            "quarantine_count": len(quarantine),
            "model": BREEZE_MODEL,
            "revision": BREEZE_REVISION,
            "label_adapter_version": label_adapter_version,
            "label_target_sample_rate": label_target_sample_rate,
            "verified_labels": 0,
        }
        (temporary / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        releases_dir.mkdir(parents=True, exist_ok=True)
        temporary.rename(release)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return release
