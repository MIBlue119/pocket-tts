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


def _parse_sources(path: Path, acquisition_dir: Path, allow_playlist: bool) -> list[SourceRecord]:
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
            audio_path, video_id = acquire_url(url, acquisition_dir, allow_playlist)
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
            )
        )
    if not sources:
        raise ValueError("source manifest is empty")
    return sources


def _assign_splits(
    sources: list[SourceRecord], train_only: bool, seed: int
) -> dict[str, Literal["train", "valid"]]:
    explicit: dict[str, Literal["train", "valid"]] = {}
    groups = sorted({source.group for source in sources})
    for source in sources:
        if source.split is not None:
            old = explicit.setdefault(source.group, source.split)
            if old != source.split:
                raise ValueError(f"group {source.group} has conflicting supplied splits")
    if train_only:
        if any(split == "valid" for split in explicit.values()):
            raise ValueError("--train-only conflicts with a supplied valid split")
        return {group: "train" for group in groups}
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
    return assigned


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
    allow_playlist: bool = False,
) -> Path:
    if min_duration <= 0 or max_duration < min_duration:
        raise ValueError("durations must satisfy 0 < min_duration <= max_duration")
    if min_silence <= 0:
        raise ValueError("min_silence must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = _parse_sources(source_manifest, output_dir / "downloads", allow_playlist)
    splits = _assign_splits(sources, train_only, seed)
    candidates: list[SegmentRecord] = []
    quarantined: list[SegmentRecord] = []
    for source in sources:
        assert source.audio_path is not None
        audio = load_audio(source.audio_path)
        digest = sha256_file(source.audio_path)
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
    transcribe = transcriber or breeze_transcriber(device)
    accepted: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for row in rows:
        text = transcribe(
            Path(row["audio_path"]), float(row["start"]), float(row["duration"])
        ).strip()
        enriched = {
            **row,
            "input_text_zh": text,
            "input_language": "zh-TW",
            "label_type": "asr_pseudo",
            "label_provenance": "Breeze-ASR-26",
            "label_model": BREEZE_MODEL,
            "label_model_revision": BREEZE_REVISION,
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
    release = releases_dir / release_id
    if release.exists():
        return release
    temporary = releases_dir / f".{release_id}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        for split in ("train", "valid"):
            compact = [
                {
                    "path": row["audio_path"],
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
