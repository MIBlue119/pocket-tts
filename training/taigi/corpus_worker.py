"""Resident, resumable model stages for one complete source video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import soundfile as sf

from training.taigi.audio import sha256_file
from training.taigi.enhance import ADAPTER_VERSION, MODEL, REVISION, enhance_file, load_predictor
from training.taigi.pipeline import prepare_candidates, transcribe_release
from training.taigi.schema import read_jsonl
from training.taigi.transcriber import (
    BREEZE_ADAPTER_VERSION,
    BREEZE_MODEL,
    BREEZE_REVISION,
    breeze_transcriber,
)

MAX_CHUNK_SECONDS = 600


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256_file(path) for path in paths}


def _verify(artifacts: dict[str, str]) -> None:
    for name, expected in artifacts.items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"artifact integrity check failed: {path}")


def _code_identity() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        name: sha256_file(directory / name)
        for name in ("corpus_worker.py", "enhance.py", "transcriber.py", "pipeline.py")
    }


def _context(spec: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    raw, output = Path(spec["raw_audio"]), Path(spec["output_dir"])
    if not raw.is_absolute() or not output.is_absolute():
        raise ValueError("raw_audio and output_dir must be absolute")
    identity = {key: spec[key] for key in ("video_id", "group_id", "original_url", "license")}
    batch_size = spec.get("asr_batch_size", 1)
    if type(batch_size) is not int or batch_size not in range(1, 5):
        raise ValueError("asr_batch_size must be 1 to 4")
    identity["asr_batch_size"] = batch_size
    identity["raw_audio_sha256"] = sha256_file(raw)
    identity["stage_code_sha256"] = _code_identity()
    return raw, output, identity


def _existing(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = _json(path)
    if any(value.get(key) != expected for key, expected in identity.items()):
        raise ValueError(f"artifact identity integrity check failed: {path}")
    _verify(value["artifacts"])
    return value


def enhance(spec: dict[str, Any]) -> dict[str, Any]:
    raw, output, identity = _context(spec)
    model = {
        "enhancement_model": MODEL,
        "enhancement_revision": REVISION,
        "adapter_version": ADAPTER_VERSION,
    }
    identity = {**identity, **model}
    report_path = output / "enhancement-report.json"
    existing = _existing(report_path, identity)
    if existing is not None:
        return existing
    with sf.SoundFile(raw) as audio:
        frames, rate = len(audio), audio.samplerate
    if frames <= 0:
        raise ValueError("source contains no decoded frames")
    chunk_root = output / "chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    predictor = None
    chunks: list[dict[str, Any]] = []
    sources = []
    artifacts: dict[str, str] = {}
    model_artifacts = None
    for start in range(0, frames, MAX_CHUNK_SECONDS * rate):
        count = min(MAX_CHUNK_SECONDS * rate, frames - start)
        directory = chunk_root / f"{start:012d}-{start + count:012d}"
        chunk_identity = {
            **identity,
            "start_sample": start,
            "num_samples": count,
            "sample_rate": rate,
        }
        completed: dict[str, Any] | None = _existing(directory / "complete.json", chunk_identity)
        if completed is None:
            if directory.exists():
                raise ValueError(f"incomplete final chunk integrity check failed: {directory}")
            if predictor is None:
                cache = Path(spec["cache_dir"]) if spec.get("cache_dir") else None
                predictor = load_predictor(cache, True, spec["device"])
            temporary = chunk_root / f".chunk-{uuid.uuid4().hex}"
            temporary.mkdir()
            audio_path = temporary / "audio.wav"
            metadata_path = enhance_file(
                raw, audio_path, predictor, start=start / rate, duration=count / rate
            )
            metadata = _json(metadata_path)
            if metadata["num_samples"] != count or metadata["source_start_sample"] != start:
                raise ValueError("enhancement changed exact source frame coverage")
            metadata["audio_path"] = str(directory / "audio.wav")
            _write(metadata_path, metadata)
            completed = {
                **chunk_identity,
                "audio_path": str(directory / "audio.wav"),
                "audio_sha256": sha256_file(audio_path),
                "metadata_path": str(directory / "audio.wav.json"),
                "metadata_sha256": sha256_file(metadata_path),
                "model_artifact_sha256": metadata["model_artifact_sha256"],
                "artifacts": {
                    str(directory / path.name): sha256_file(path)
                    for path in (audio_path, metadata_path)
                },
            }
            _write(temporary / "complete.json", completed)
            temporary.rename(directory)
        completed = _json(directory / "complete.json")
        current_model = completed["model_artifact_sha256"]
        if model_artifacts is not None and model_artifacts != current_model:
            raise ValueError("enhancement model artifact integrity mismatch across chunks")
        model_artifacts = current_model
        chunks.append(completed)
        artifacts.update(completed["artifacts"])
        artifacts.update(_hashes([directory / "complete.json"]))
        sources.append(
            {
                "source_id": f"{spec['video_id']}-{start:012d}",
                "language": "nan-tw",
                "audio_path": completed["audio_path"],
                "audio_variant": "enhanced",
                "group_id": spec["group_id"],
                "split": "train",
                **identity,
                "original_video_id": spec["video_id"],
                "source_audio_sha256": identity["raw_audio_sha256"],
                "source_interval": {
                    "start_sample": start,
                    "num_samples": count,
                    "sample_rate": rate,
                },
                "source_start_sample": start,
                "source_num_samples": count,
                "source_sample_rate": rate,
                "source_audio_path": str(raw),
                "enhancement_provenance": _json(Path(completed["metadata_path"])),
            }
        )
    source_path = output / "sources.jsonl"
    serialized = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in sources
    )
    if source_path.exists() and source_path.read_text() != serialized:
        raise ValueError("existing source manifest integrity mismatch")
    if not source_path.exists():
        temporary_source = output / f".sources-{uuid.uuid4().hex}.partial"
        temporary_source.write_text(serialized)
        temporary_source.rename(source_path)
    candidates = output / "candidates"
    if candidates.exists():
        marker = _existing(candidates / "complete.json", identity)
        if marker is None:
            raise ValueError("candidate directory missing completion integrity record")
    else:
        temporary_candidates = output / f".candidates-{uuid.uuid4().hex}"
        prepare_candidates(source_path, temporary_candidates, train_only=True)
        hashes = {
            str(candidates / path.name): sha256_file(path)
            for path in temporary_candidates.iterdir()
            if path.is_file()
        }
        _write(temporary_candidates / "complete.json", {**identity, "artifacts": hashes})
        temporary_candidates.rename(candidates)
    artifacts.update(_hashes([source_path, *sorted(candidates.iterdir())]))
    prepared = _json(candidates / "prepare-report.json")
    report = {
        **identity,
        "state": "complete",
        "total_frames": frames,
        "covered_frames": sum(chunk["num_samples"] for chunk in chunks),
        "sample_rate": rate,
        "chunks": chunks,
        "model_artifact_sha256": model_artifacts,
        "candidate_count": prepared["candidate_count"],
        "quarantine_count": prepared["quarantine_count"],
        "artifacts": artifacts,
    }
    _write(report_path, report)
    return report


class CachedTranscriber:
    adapter_version = BREEZE_ADAPTER_VERSION
    target_sample_rate = 16000

    def __init__(self, output: Path, device: str, batch_size: int = 1):
        if type(batch_size) is not int or batch_size not in range(1, 5):
            raise ValueError("ASR batch size must be 1 to 4")
        self.batch_size = batch_size
        self.cache = output / "transcript-cache"
        self.device = device
        self.transcriber = None
        self.hashes: dict[Path, str] = {}

    def _entry(self, path: Path, start: float, duration: float) -> tuple[Path, dict[str, Any]]:
        if path not in self.hashes:
            self.hashes[path] = sha256_file(path)
        identity = {
            "audio_sha256": self.hashes[path],
            "asr_batch_size": self.batch_size,
            "start": start,
            "duration": duration,
            "model": BREEZE_MODEL,
            "revision": BREEZE_REVISION,
            "adapter_version": self.adapter_version,
            "target_sample_rate": self.target_sample_rate,
            "stage_code_sha256": _code_identity(),
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return self.cache / f"{key}.json", identity

    def _cached(self, cache_path: Path, identity: dict[str, Any]) -> str | None:
        if cache_path.exists():
            cached = _json(cache_path)
            if cached.get("identity") != identity or not isinstance(cached.get("text"), str):
                raise ValueError(f"transcript cache integrity failure: {cache_path}")
            text = cached["text"]
            if cached.get("text_sha256") != hashlib.sha256(text.encode()).hexdigest():
                raise ValueError(f"transcript cache integrity failure: {cache_path}")
            return text
        return None

    def _load(self) -> None:
        if self.transcriber is None:
            self.transcriber = breeze_transcriber(self.device)
            if self.transcriber.target_sample_rate != self.target_sample_rate:
                raise ValueError("pinned Breeze target sample rate changed")

    def _store(self, cache_path: Path, identity: dict[str, Any], text: str) -> None:
        _write(
            cache_path,
            {
                "identity": identity,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            },
        )

    def __call__(self, path: Path, start: float, duration: float) -> str:
        cache_path, identity = self._entry(path, start, duration)
        cached = self._cached(cache_path, identity)
        if cached is not None:
            return cached
        self._load()
        assert self.transcriber is not None
        text = self.transcriber(path, start, duration).strip()
        self._store(cache_path, identity, text)
        return text

    def prefill(self, rows: list[dict[str, Any]]) -> None:
        pending: list[tuple[Path, dict[str, Any], tuple[Path, float, float]]] = []
        seen: set[Path] = set()

        def flush() -> None:
            if not pending:
                return
            self._load()
            assert self.transcriber is not None
            texts = self.transcriber.transcribe_batch([item[2] for item in pending])
            if len(texts) != len(pending) or any(not isinstance(text, str) for text in texts):
                raise ValueError("Breeze batch output does not match cache inputs")
            for (cache_path, identity, _), text in zip(pending, texts, strict=True):
                self._store(cache_path, identity, text.strip())
            pending.clear()

        for row in sorted(rows, key=lambda row: float(row["duration"])):
            interval = (
                Path(row["audio_path"]).resolve(),
                float(row["start"]),
                float(row["duration"]),
            )
            cache_path, identity = self._entry(*interval)
            if identity["audio_sha256"] != row.get("audio_sha256"):
                raise ValueError("candidate audio hash mismatch before ASR batch")
            cached = self._cached(cache_path, identity)
            if cached is not None or cache_path in seen:
                continue
            seen.add(cache_path)
            pending.append((cache_path, identity, interval))
            if len(pending) == self.batch_size:
                flush()
        flush()


def transcribe(spec: dict[str, Any]) -> dict[str, Any]:
    _, output, identity = _context(spec)
    identity.update(
        {
            "label_model": BREEZE_MODEL,
            "label_model_revision": BREEZE_REVISION,
            "label_adapter_version": BREEZE_ADAPTER_VERSION,
        }
    )
    existing = _existing(output / "result.json", identity)
    if existing is not None:
        return existing
    enhancement = _existing(
        output / "enhancement-report.json",
        {
            key: identity[key]
            for key in (
                "video_id",
                "group_id",
                "original_url",
                "license",
                "raw_audio_sha256",
                "stage_code_sha256",
            )
        },
    )
    if enhancement is None:
        raise ValueError("enhancement completion report missing")
    candidates = output / "candidates" / "candidates.jsonl"
    rows = read_jsonl(candidates)
    release = None
    accepted, rejected = 0, enhancement["quarantine_count"]
    artifacts = {**enhancement["artifacts"], **_hashes([output / "enhancement-report.json"])}
    if rows:
        transcriber = CachedTranscriber(output, spec["device"], identity["asr_batch_size"])
        if identity["asr_batch_size"] > 1:
            transcriber.prefill(rows)
        release = transcribe_release(
            candidates, output / "releases", device=spec["device"], transcriber=transcriber
        )
        report = _json(release / "report.json")
        accepted = report["accepted_weak_label_count"]
        rejected += report["quarantine_count"]
        artifacts.update(_hashes(sorted(path for path in release.rglob("*") if path.is_file())))
    result = {
        **identity,
        "state": "complete" if accepted else "no_candidates",
        "release_path": str(release) if release else None,
        "candidate_count": len(rows),
        "accepted_weak_label_count": accepted,
        "quarantine_count": rejected,
        "verified_labels": 0,
        "artifacts": artifacts,
    }
    _write(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("enhance", "transcribe"))
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    result = {"enhance": enhance, "transcribe": transcribe}[args.stage](_json(args.spec))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
