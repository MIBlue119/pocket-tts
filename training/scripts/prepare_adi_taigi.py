"""Prepare pinned ADI recordings on CPU; native transcripts remain reference metadata.

No enhancement, transcript translation, tokenizer change, or training occurs here.
The transcribe subcommand explicitly invokes the existing resident Breeze path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import tarfile
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from training.scripts.run_experiment import run_lock, write_json
from training.taigi.audio import sha256_file
from training.taigi.schema import read_jsonl, write_jsonl

REPO = "adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hokkien"
REVISION = "ffcc36b9afe31ec50f7742c0c4c614f1a5b40151"
VERSION = "adi-whole-clip/v1"


def text_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def path_key(path: str) -> str:
    return path.replace("\\", "/")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"client_id", "path", "sentence_id", "sentence", "locale"}.issubset(
            reader.fieldnames or []
        ):
            raise ValueError("ADI TSV requires literal tab-separated source columns")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("malformed ADI TSV columns")
    return rows


def archive_rows(path: Path, split: str) -> list[dict[str, Any]]:
    """Inspect safe regular members; never extract archive paths to the filesystem."""
    members: dict[str, tarfile.TarInfo] = {}
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not member.isfile()
                or "\\" in member.name
            ):
                raise ValueError("unsafe or nonregular ADI tar member")
            if member.name in members:
                raise ValueError("duplicate ADI tar member")
            members[member.name] = member
        rows = []
        stems = sorted({name.rsplit(".", 1)[0] for name in members})
        for stem in stems:
            names = {suffix: stem + suffix for suffix in (".json", ".txt", ".mp3")}
            if not all(name in members for name in names.values()):
                raise ValueError("ADI archive is missing an audio/text/metadata pair")
            for suffix in (".json", ".txt"):
                if members[names[suffix]].size > 1024 * 1024:
                    raise ValueError("ADI text member exceeds inspection bound")
            metadata_stream = archive.extractfile(members[names[".json"]])
            text_stream = archive.extractfile(members[names[".txt"]])
            assert metadata_stream is not None and text_stream is not None
            metadata_bytes, original_text = metadata_stream.read(), text_stream.read()
            metadata = json.loads(metadata_bytes)
            if not isinstance(metadata, dict) or any(
                not isinstance(metadata.get(key), str)
                for key in ("sentence", "locale", "path", "client_id", "sentence_id")
            ):
                raise ValueError("ADI metadata text/locale/identity fields are malformed")
            if original_text != metadata["sentence"].encode("utf-8"):
                raise ValueError("ADI TXT and JSON transcript bytes disagree")
            if metadata.get("split") != split or metadata.get("audio_ext") not in {"mp3", ".mp3"}:
                raise ValueError("ADI member split/audio metadata mismatch")
            rows.append(
                {
                    "archive": str(path.resolve()),
                    "audio_member": names[".mp3"],
                    "audio_offset": members[names[".mp3"]].offset_data,
                    "audio_size": members[names[".mp3"]].size,
                    "metadata_member": names[".json"],
                    "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                    "native_text_sha256": hashlib.sha256(original_text).hexdigest(),
                    "official_split": split,
                    "raw_metadata": metadata,
                }
            )
    return rows


def cohort(row: dict[str, Any]) -> str:
    metadata = row["raw_metadata"]
    locale, source_path = metadata.get("locale"), metadata.get("path")
    if not isinstance(locale, str) or not isinstance(source_path, str):
        return "invalid_locale_or_path"
    relative = PurePosixPath(path_key(source_path))
    if relative.is_absolute() or ".." in relative.parts:
        return "invalid_locale_or_path"
    tw_path = "common_voice_nan-tw" in source_path
    cn_path = "common_voice_zh-CN" in source_path
    if locale == "nan-tw" and tw_path and not cn_path:
        return "strict_nan_tw"
    if not locale.strip() and tw_path and not cn_path:
        return "inferred_locale_quarantine"
    if locale.casefold() == "zh-cn" and cn_path or not locale.strip() and cn_path:
        return "mandarin_source_excluded"
    return "locale_path_conflict"


def decode(row: dict[str, Any], check_duration: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    if row["audio_size"] > 50 * 1024 * 1024:
        raise ValueError("audio_member_too_large")
    # Offsets were validated from regular uncompressed tar members. Direct
    # bounded reads avoid reopening and rescanning thousands of tar headers per clip.
    with Path(row["archive"]).open("rb") as stream:
        stream.seek(row["audio_offset"])
        compressed = stream.read(row["audio_size"])
    if len(compressed) != row["audio_size"]:
        raise ValueError("truncated_audio_member")
    samples, rate = sf.read(io.BytesIO(compressed), dtype="float32", always_2d=True)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError("empty_or_nonfinite_audio")
    seconds = len(samples) / rate
    if check_duration and not 0.8 <= seconds <= 20:
        raise ValueError("duration_outside0.8_to20_seconds")
    channels = samples.shape[1]
    mono = samples.mean(axis=1)
    if rate != 24000:
        mono = resample_poly(mono, 24000, rate).astype(np.float32, copy=False)
    if not np.isfinite(mono).all():
        raise ValueError("nonfinite_resampled_audio")
    return mono, {
        "compressed_audio_sha256": hashlib.sha256(compressed).hexdigest(),
        "decoded_pcm_sha256": hashlib.sha256(mono.astype("<f4", copy=False).tobytes()).hexdigest(),
        "original_sample_rate": rate,
        "original_channels": channels,
        "original_frames": len(samples),
        "original_duration": seconds,
        "sample_rate": 24000,
        "frames": len(mono),
        "duration": len(mono) / 24000,
    }


def speakers_split(rows: list[dict[str, Any]], seed: int) -> dict[str, str]:
    durations: Counter[str] = Counter()
    for row in rows:
        durations[row["raw_metadata"]["client_id"]] += row["audio_identity"]["duration"]
    groups = sorted(
        durations, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    )
    if len(groups) < 2:
        raise ValueError("ADI release needs at least two independent speakers")
    target, seconds = sum(durations.values()) * 0.1, 0.0
    valid: set[str] = set()
    for group in groups:
        if len(valid) < len(groups) - 1 and abs(target - seconds - durations[group]) < abs(
            target - seconds
        ):
            valid.add(group)
            seconds += durations[group]
    if not valid:
        valid.add(min(groups, key=lambda group: durations[group]))
    return {group: "valid" if group in valid else "train" for group in groups}


def reference_parts(sentence: str) -> dict[str, Any]:
    suffix = re.search(r"[（(]([^（）()]*)[）)]\s*$", sentence)
    if (
        suffix is not None
        and re.search(r"[A-Za-z]", suffix.group(1))
        and not re.search(r"[\u3400-\u9fff]", suffix.group(1))
    ):
        return {
            "han_reference_display": sentence[: suffix.start()].rstrip(),
            "romanization_annotation": suffix.group(1),
            "reference_display_transform": "trailing_parenthetical_latin_reference_only/v1",
            "reference_is_Mandarin_translation": False,
        }
    return {
        "han_reference_display": sentence,
        "romanization_annotation": None,
        "reference_display_transform": "unchanged",
        "reference_is_Mandarin_translation": False,
    }


def prepare(raw: Path, output: Path, seed: int = 0, workers: int = 4) -> Path:
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("CPU preparation workers must be1 to4")
    inventory = json.loads((raw / "inventory.json").read_text())
    if inventory.get("repo_id") != REPO or inventory.get("revision") != REVISION:
        raise ValueError("ADI source repository/revision differs from pinned source")
    archives = [item for item in inventory["files"] if item["path"].endswith(".tar")]
    input_hashes = {name: sha256_file(raw / name) for name in ("train.tsv", "test.tsv", "LICENSE")}
    for item in archives:
        path = raw / item["path"]
        if path.stat().st_size != item["size"] or sha256_file(path) != item["lfs_sha256"]:
            raise ValueError("ADI pinned archive hash mismatch")
        input_hashes[item["path"]] = item["lfs_sha256"]
    identity = {
        "version": VERSION,
        "repo_id": REPO,
        "revision": REVISION,
        "inputs": input_hashes,
        "seed": seed,
        "script_sha256": sha256_file(Path(__file__)),
    }
    release_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    release = output.resolve() / release_id
    if release.exists():
        verify_release(release)
        return release
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / ("." + release_id + ".partial")
    temporary.mkdir(exist_ok=False)
    (temporary / "audio").mkdir()
    started = time.time()
    rows = {split: [] for split in ("train", "test")}
    for item in sorted(archives, key=lambda item: item["path"]):
        split = item["path"].split("/")[0]
        rows[split].extend(archive_rows(raw / item["path"], split))
    for split in ("train", "test"):
        original = read_tsv(raw / f"{split}.tsv")

        def key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
            return (
                path_key(row["path"]),
                row["sentence"],
                row["locale"].casefold(),
                row["client_id"],
                row["sentence_id"],
            )

        if len({row["audio_member"] for row in rows[split]}) != len(rows[split]):
            raise ValueError("duplicate ADI item across archives")
        if Counter(key(row) for row in original) != Counter(
            key(row["raw_metadata"]) for row in rows[split]
        ):
            raise ValueError("ADI archive rows disagree with literal TSV")
    test = rows["test"]
    test_clients = {
        row["raw_metadata"]["client_id"] for row in test if row["raw_metadata"].get("client_id")
    }
    test_paths = {path_key(row["raw_metadata"]["path"]) for row in test}
    test_compressed, test_pcm = set(), set()
    for row in test:
        _, audio_identity = decode(row, check_duration=False)
        test_compressed.add(audio_identity["compressed_audio_sha256"])
        test_pcm.add(audio_identity["decoded_pcm_sha256"])
    excluded, inferred, decoded, eligible_rows = [], [], [], []
    counts: Counter[str] = Counter()
    strict_count = sum(cohort(row) == "strict_nan_tw" for row in rows["train"])
    strict_test_speaker_overlap = sum(
        cohort(row) == "strict_nan_tw" and row["raw_metadata"].get("client_id") in test_clients
        for row in rows["train"]
    )
    for index, row in enumerate(rows["train"]):
        metadata = row["raw_metadata"]
        reason = cohort(row)
        if reason == "inferred_locale_quarantine":
            inferred.append({**row, "reason": reason})
        if reason == "strict_nan_tw":
            client, sentence = metadata.get("client_id"), metadata.get("sentence_id")
            if not client.strip() or not sentence.strip():
                reason = "missing_speaker_or_sentence_identity"
            elif not metadata["sentence"].strip():
                reason = "empty_native_reference"
            elif path_key(metadata["path"]) in test_paths:
                reason = "official_test_recording_path_overlap"
            else:
                reason = ""
        if reason:
            excluded.append({**row, "reason": reason})
            counts[reason] += 1
            continue
        eligible_rows.append(row)

    def convert(row: dict[str, Any]) -> tuple[dict[str, Any], str]:
        try:
            samples, audio_identity = decode(row)
        except (ValueError, RuntimeError) as error:
            return row, "audio_rejected:" + str(error)
        if (
            audio_identity["compressed_audio_sha256"] in test_compressed
            or audio_identity["decoded_pcm_sha256"] in test_pcm
        ):
            return row, "official_test_audio_overlap"
        source_id = (
            "adi-" + hashlib.sha256(f"{REVISION}:{row['audio_member']}".encode()).hexdigest()[:20]
        )
        wav = temporary / "audio" / f"{source_id}.wav"
        sf.write(wav, samples, 24000, subtype="FLOAT")
        row.update(
            audio_identity=audio_identity,
            source_id=source_id,
            audio_path=str(release / "audio" / wav.name),
            audio_sha256=sha256_file(wav),
        )
        return row, ""

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for completed, (row, reason) in enumerate(pool.map(convert, eligible_rows), start=1):
            if reason:
                excluded.append({**row, "reason": reason})
                counts[reason] += 1
            else:
                decoded.append(row)
            if completed % 100 == 0:
                write_json(
                    temporary / "progress.json",
                    {
                        "eligible_rows_completed": completed,
                        "eligible_total": len(eligible_rows),
                        "decoded": len(decoded),
                        "workers": workers,
                        "elapsed_seconds": time.time() - started,
                    },
                )
    split_map = speakers_split(decoded, seed)
    candidates = []
    dev_sentences, dev_texts, seen_compressed, seen_pcm = set(), set(), set(), set()
    for row in sorted(
        decoded,
        key=lambda row: (
            split_map[row["raw_metadata"]["client_id"]] != "valid",
            row["audio_member"],
        ),
    ):
        metadata, audio = row["raw_metadata"], row["audio_identity"]
        sentence, native_key = metadata["sentence_id"], text_key(metadata["sentence"])
        split = split_map[metadata["client_id"]]
        if split == "train" and (sentence in dev_sentences or native_key in dev_texts):
            reason = "train_sentence_overlap_with_development"
        elif (
            audio["compressed_audio_sha256"] in seen_compressed
            or audio["decoded_pcm_sha256"] in seen_pcm
        ):
            reason = "duplicate_audio_dev_precedence"
        else:
            reason = ""
        if reason:
            excluded.append({**row, "reason": reason})
            counts[reason] += 1
            (temporary / "audio" / Path(row["audio_path"]).name).unlink()
            continue
        if split == "valid":
            dev_sentences.add(sentence)
            dev_texts.add(native_key)
        seen_compressed.add(audio["compressed_audio_sha256"])
        seen_pcm.add(audio["decoded_pcm_sha256"])
        candidates.append(row)
    sources, segments = [], []
    for row in candidates:
        metadata, audio = row["raw_metadata"], row["audio_identity"]
        client = metadata["client_id"]
        group = f"adi:{REVISION}:speaker:" + hashlib.sha256(client.encode()).hexdigest()[:20]
        provenance = {
            "source_repo": REPO,
            "source_revision": REVISION,
            "source_license": "TRAIL-0.1",
            "license_sha256": input_hashes["LICENSE"],
            "official_split": "train",
            "locale_evidence": "explicit_nan_tw_and_matching_source_path",
            "archive": row["archive"],
            "audio_member": row["audio_member"],
            "metadata_sha256": row["metadata_sha256"],
            "native_text_sha256": row["native_text_sha256"],
            "native_transcript": metadata["sentence"],
            "original_metadata": metadata,
            **reference_parts(metadata["sentence"]),
            "audio_identity": audio,
            "parser_version": VERSION,
            "native_text_role": "reference_only_not_Mandarin_semantic_target",
        }
        source = {
            "source_id": row["source_id"],
            "group_id": group,
            "language": "nan-tw",
            "audio_variant": "raw",
            "audio_path": row["audio_path"],
            "audio_sha256": row["audio_sha256"],
            "split": split_map[client],
            **provenance,
        }
        sources.append(source)
        segments.append(
            {
                "utterance_id": row["source_id"] + "-whole",
                "source_id": row["source_id"],
                "group_id": group,
                "language": "nan-tw",
                "language_provenance": "source_declared",
                "audio_variant": "raw",
                "audio_path": row["audio_path"],
                "audio_sha256": row["audio_sha256"],
                "start": 0.0,
                "duration": audio["duration"],
                "split": split_map[client],
                "status": "candidate",
                "quarantine_reason": None,
                "source_metadata": provenance,
            }
        )
    write_jsonl(temporary / "sources.jsonl", sources)
    write_jsonl(temporary / "candidates.jsonl", segments)
    write_jsonl(temporary / "excluded.jsonl", excluded)
    write_jsonl(temporary / "inferred-locale-quarantine.jsonl", inferred)
    shutil.copyfile(raw / "LICENSE", temporary / "LICENSE")
    artifacts = {
        str(path.relative_to(temporary)): sha256_file(path)
        for path in temporary.rglob("*")
        if path.is_file()
    }
    training_rows = [
        row for row in candidates if split_map[row["raw_metadata"]["client_id"]] == "train"
    ]
    training_clients = {row["raw_metadata"]["client_id"] for row in training_rows}
    training_sentences = {row["raw_metadata"]["sentence_id"] for row in training_rows}
    training_texts = {text_key(row["raw_metadata"]["sentence"]) for row in training_rows}
    test_buckets = Counter(
        (
            "seen_speaker"
            if row["raw_metadata"]["client_id"] in training_clients
            else "unseen_speaker"
        )
        + "|"
        + (
            "seen_sentence"
            if row["raw_metadata"]["sentence_id"] in training_sentences
            or text_key(row["raw_metadata"]["sentence"]) in training_texts
            else "unseen_sentence"
        )
        for row in test
    )
    write_json(
        temporary / "report.json",
        {
            "state": "complete",
            "identity": identity,
            "train_input_rows": len(rows["train"]),
            "official_test_rows": len(test),
            "accepted": len(sources),
            "cpu_worker_limit": workers,
            "within_split_repeated_text_policy": "retain distinct recordings; drop only cross train/dev sentence overlap and global exact-audio duplicates",
            "unique_native_texts_by_split": {
                split: len(
                    {text_key(row["native_transcript"]) for row in sources if row["split"] == split}
                )
                for split in ("train", "valid")
            },
            "excluded": len(excluded),
            "exclusion_reasons": dict(counts),
            "inferred_locale_quarantined": len(inferred),
            "test_audio_check_failures": [],
            "strict_input_rows": strict_count,
            "strict_test_speaker_overlap_not_globally_excluded": strict_test_speaker_overlap,
            "split_policy": "speaker-disjoint development from official train; official test recordings/audio excluded, but official test speakers/sentences may be seen; dev sentence/audio takes precedence",
            "official_test_proxy_buckets": dict(test_buckets),
            "reference_profile": {
                "short_reference_le4_han_characters": sum(
                    sum(
                        "\u3400" <= char <= "\u9fff"
                        for char in reference_parts(row["raw_metadata"]["sentence"])[
                            "han_reference_display"
                        ]
                    )
                    <= 4
                    for row in candidates
                ),
                "reference_display_with_final_sentence_punctuation": sum(
                    reference_parts(row["raw_metadata"]["sentence"])["han_reference_display"]
                    .rstrip()
                    .endswith(("。", "？", "！", "?", "!"))
                    for row in candidates
                ),
                "meaning": "descriptive reference-length/punctuation counts, not word-versus-sentence linguistic validation",
            },
            "counts": dict(Counter(source["split"] for source in sources)),
            "speaker_counts": {
                split: len({row["group_id"] for row in segments if row["split"] == split})
                for split in ("train", "valid")
            },
            "duration_seconds": {
                split: sum(row["duration"] for row in segments if row["split"] == split)
                for split in ("train", "valid")
            },
            "artifacts": artifacts,
            "elapsed_seconds": time.time() - started,
            "limitations": [
                "Native labels are reference only; no semantic Breeze labels generated yet.",
                "Locale metadata is not independent spoken-language verification.",
                "Exact overlap against YouTube intervals has not been checked; do not claim it was.",
            ],
        },
    )
    temporary.rename(release)
    verify_release(release)
    return release


def verify_release(release: Path) -> dict[str, Any]:
    report = json.loads((release / "report.json").read_text())
    if report.get("state") != "complete":
        raise ValueError("ADI prepared release is incomplete")
    if (
        hashlib.sha256(json.dumps(report["identity"], sort_keys=True).encode()).hexdigest()[:20]
        != release.name
    ):
        raise ValueError("ADI prepared release identity mismatch")
    for name, expected in report["artifacts"].items():
        path = release / name
        if (
            not path.resolve().is_relative_to(release.resolve())
            or not path.is_file()
            or sha256_file(path) != expected
        ):
            raise ValueError("ADI prepared artifact integrity failure")
    return report


def transcribe(release: Path, output: Path, device: str, batch_size: int) -> Path:
    from training.taigi.corpus_worker import CachedTranscriber, _code_identity
    from training.taigi.merge import _validate_release
    from training.taigi.pipeline import transcribe_release
    from training.taigi.transcriber import BREEZE_ADAPTER_VERSION, BREEZE_MODEL, BREEZE_REVISION

    verify_release(release)
    candidates = release / "candidates.jsonl"
    plan = {
        "prepared_report_sha256": sha256_file(release / "report.json"),
        "candidates_sha256": sha256_file(candidates),
        "asr_batch_size": batch_size,
        "device": device,
        "model": BREEZE_MODEL,
        "revision": BREEZE_REVISION,
        "adapter": BREEZE_ADAPTER_VERSION,
        "target_sample_rate": 16000,
        "script_sha256": sha256_file(Path(__file__)),
        "stage_code_sha256": _code_identity(),
    }
    with run_lock(output):
        plan_path = output / "plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("ADI ASR stage plan changed; use a fresh output directory")
        result_path = output / "result.json"
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous.get("asr_identity") != plan:
                raise ValueError("ADI completed ASR identity changed")
            previous_release = Path(previous["release"])
            if not previous_release.resolve().is_relative_to((output / "releases").resolve()):
                raise ValueError("ADI completed ASR release escapes output")
            manifests = previous.get("release_manifest_hashes", {})
            required = {
                "metadata.jsonl",
                "train.jsonl",
                "valid.jsonl",
                "quarantine.jsonl",
                "report.json",
            }
            if set(manifests) != required or any(
                sha256_file(previous_release / name) != expected
                for name, expected in manifests.items()
            ):
                raise ValueError("ADI completed ASR release manifest integrity failure")
        write_json(plan_path, plan)
        for existing in (output / "releases").glob("*"):
            if existing.is_dir() and not existing.name.startswith("."):
                _validate_release(existing)
        cache = CachedTranscriber(output, device, batch_size)
        if batch_size > 1:
            cache.prefill(read_jsonl(candidates))
        result = transcribe_release(
            candidates, output / "releases", device=device, transcriber=cache
        )
        counts = json.loads((result / "report.json").read_text())
        write_json(
            output / "result.json",
            {
                "state": "complete" if counts["accepted_weak_label_count"] else "no_candidates",
                "release": str(result),
                "prepared_release": str(release),
                "asr_batch_size": batch_size,
                "asr_identity": plan,
                "accepted_weak_label_count": counts["accepted_weak_label_count"],
                "quarantine_count": counts["quarantine_count"],
                "quality_verified": False,
                "release_manifest_hashes": {
                    name: sha256_file(result / name)
                    for name in (
                        "metadata.jsonl",
                        "train.jsonl",
                        "valid.jsonl",
                        "quarantine.jsonl",
                        "report.json",
                    )
                },
                "native_transcript_role": "reference only; never relabelled as Mandarin target",
                "meaning": "Breeze semantic weak-label candidates; not independent translation truth",
            },
        )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("raw", type=Path)
    prepare_parser.add_argument("output", type=Path)
    prepare_parser.add_argument("--seed", type=int, default=0)
    prepare_parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    asr = commands.add_parser("transcribe")
    asr.add_argument("release", type=Path)
    asr.add_argument("output", type=Path)
    asr.add_argument("--device", default="cpu")
    asr.add_argument("--asr-batch-size", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    print(
        prepare(args.raw, args.output, args.seed, args.workers)
        if args.command == "prepare"
        else transcribe(args.release, args.output, args.device, args.asr_batch_size)
    )


if __name__ == "__main__":
    main()
