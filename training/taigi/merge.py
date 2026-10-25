from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from training.taigi.audio import sha256_file
from training.taigi.schema import read_jsonl, write_jsonl
from training.taigi.transcriber import BREEZE_ADAPTER_VERSION, BREEZE_MODEL, BREEZE_REVISION

Split = Literal["train", "valid"]


def _json_digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(release: Path, name: str) -> dict[str, str]:
    path = release / name
    if not path.is_file():
        raise ValueError(f"release is missing {name}: {release}")
    return {"sha256": sha256_file(path)}


def _validate_release(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    release = path.expanduser().resolve()
    report_path = release / "report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read release report: {report_path}") from error
    expected = {
        "schema_version": "zh-taigi-semantic-pair/v1",
        "model": BREEZE_MODEL,
        "revision": BREEZE_REVISION,
        "label_adapter_version": BREEZE_ADAPTER_VERSION,
        "verified_labels": 0,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"release {release} has unsupported {key}: {report.get(key)!r}")

    metadata = read_jsonl(release / "metadata.jsonl")
    manifests = {split: read_jsonl(release / f"{split}.jsonl") for split in ("train", "valid")}
    compact_counts: dict[str, int] = defaultdict(int)
    for split, rows in manifests.items():
        for row in rows:
            compact_counts[_json_digest({"split": split, **row})] += 1
    seen_audio: dict[Path, str] = {}
    audio_bounds: dict[Path, float] = {}
    for row in metadata:
        required = {
            "utterance_id",
            "source_id",
            "group_id",
            "published_audio_path",
            "audio_sha256",
            "start",
            "duration",
            "input_text_zh",
            "split",
            "source_metadata",
        }
        missing = required - row.keys()
        if missing:
            raise ValueError(f"release metadata is missing {sorted(missing)}: {release}")
        if not isinstance(row["source_metadata"], dict):
            raise ValueError(f"release metadata has invalid source_metadata: {release}")
        if not all(str(row[field]).strip() for field in ("utterance_id", "source_id", "group_id")):
            raise ValueError(f"release metadata has an empty identifier: {release}")
        if not str(row["input_text_zh"]).strip():
            raise ValueError(f"release metadata has an empty transcript: {release}")
        try:
            start, duration = float(row["start"]), float(row["duration"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"release metadata has invalid interval values: {release}") from error
        if not math.isfinite(start) or not math.isfinite(duration) or start < 0 or duration <= 0:
            raise ValueError(f"release metadata has invalid interval values: {release}")
        split = row["split"]
        if split not in manifests:
            raise ValueError(f"release metadata has invalid split {split!r}: {release}")
        if (
            row.get("label_type") != "asr_pseudo"
            or row.get("label_provenance") != "Breeze-ASR-26"
            or row.get("label_model") != BREEZE_MODEL
            or row.get("label_model_revision") != BREEZE_REVISION
            or row.get("label_adapter_version") != BREEZE_ADAPTER_VERSION
            or row.get("verification") != "unverified_weak_label"
        ):
            raise ValueError(f"release contains unsupported weak-label provenance: {release}")
        audio_path = Path(row["published_audio_path"])
        if not audio_path.is_absolute():
            audio_path = release / audio_path
        audio_path = audio_path.resolve()
        try:
            audio_path.relative_to((release / "audio").resolve())
        except ValueError as error:
            raise ValueError(f"published audio escapes release: {audio_path}") from error
        expected_hash = str(row["audio_sha256"])
        actual_hash = seen_audio.setdefault(audio_path, sha256_file(audio_path))
        if len(expected_hash) != 64 or actual_hash != expected_hash:
            raise ValueError(f"release audio hash mismatch: {audio_path}")
        if audio_path not in audio_bounds:
            try:
                import soundfile as sf

                info = sf.info(audio_path)
            except (ImportError, RuntimeError) as error:
                raise ValueError(f"cannot inspect release audio: {audio_path}") from error
            audio_bounds[audio_path] = info.frames / info.samplerate
        if start + duration > audio_bounds[audio_path] + 1e-9:
            raise ValueError(f"release interval exceeds audio bounds: {audio_path}")
        compact = {
            "path": row["published_audio_path"],
            "start": row["start"],
            "duration": row["duration"],
            "transcript": row["input_text_zh"],
        }
        key = _json_digest({"split": split, **compact})
        if not compact_counts[key]:
            raise ValueError(f"metadata does not match compact {split} manifest: {release}")
        compact_counts[key] -= 1
        row["_input_audio_path"] = str(audio_path)
        row["_parent_release"] = str(release)
    if any(compact_counts.values()) or sum(len(rows) for rows in manifests.values()) != len(
        metadata
    ):
        raise ValueError(f"compact manifests do not match metadata: {release}")
    if report.get("accepted_weak_label_count") != len(metadata):
        raise ValueError(f"release report count does not match metadata: {release}")
    identity = {
        "path": str(release),
        "release_id": report.get("release_id"),
        "files": {
            name: _file_identity(release, name)
            for name in ("report.json", "metadata.jsonl", "train.jsonl", "valid.jsonl")
        },
    }
    identity["identity_sha256"] = _json_digest(identity["files"])
    return metadata, identity


def _original_interval(row: dict[str, Any]) -> tuple[Any, ...] | None:
    source = row["source_metadata"]
    interval = source.get("source_interval")
    original_hash = source.get("source_audio_sha256")
    if not isinstance(interval, dict) or not original_hash:
        return None
    try:
        rate = int(interval["sample_rate"])
        first = int(interval["start_sample"]) + round(float(row["start"]) * rate)
        count = round(float(row["duration"]) * rate)
        available = int(interval["num_samples"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid source_metadata.source_interval") from error
    if rate <= 0 or first < int(interval["start_sample"]) or count <= 0:
        raise ValueError("invalid original source interval")
    if first + count > int(interval["start_sample"]) + available:
        raise ValueError("row interval exceeds original source interval")
    return str(original_hash), first, count, rate


def _deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    keys: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (item["_parent_release"], item["utterance_id"])):
        interval = _original_interval(row)
        row_keys = [("bytes", row["audio_sha256"], row["start"], row["duration"])]
        if interval is not None:
            row_keys.append(("original", *interval))
        matches = {id(keys[key]): keys[key] for key in row_keys if key in keys}
        if len(matches) > 1:
            raise ValueError("duplicate keys resolve to different metadata rows")
        if matches:
            previous = next(iter(matches.values()))
            comparable = (
                "input_text_zh",
                "label_type",
                "label_provenance",
                "label_model",
                "label_model_revision",
                "label_adapter_version",
                "verification",
            )
            if any(row.get(field) != previous.get(field) for field in comparable):
                raise ValueError("duplicate source interval has conflicting text or provenance")
            for key in row_keys:
                keys[key] = previous
            continue
        selected.append(row)
        for key in row_keys:
            keys[key] = row
    return selected


def _merged_utterance_id(row: dict[str, Any]) -> str:
    identity = {
        "source_id": row["source_id"],
        "original_interval": _original_interval(row),
        "audio_sha256": row["audio_sha256"],
        "start": row["start"],
        "duration": row["duration"],
    }
    return f"merged-{_json_digest(identity)[:24]}"


class _Groups:
    def __init__(self, groups: set[str]) -> None:
        self.parent = {group: group for group in groups}

    def find(self, group: str) -> str:
        while self.parent[group] != group:
            self.parent[group] = self.parent[self.parent[group]]
            group = self.parent[group]
        return group

    def union(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _assign_splits(
    rows: list[dict[str, Any]], valid_groups: set[str], train_only: bool
) -> dict[str, Split]:
    groups = _Groups({str(row["group_id"]) for row in rows})
    linked: dict[tuple[str, str], str] = {}
    for row in rows:
        group = str(row["group_id"])
        source = row["source_metadata"]
        dimensions = [
            ("source_id", str(row["source_id"])),
            ("enhanced_hash", str(row["audio_sha256"])),
        ]
        original_hash = source.get("source_audio_sha256")
        if original_hash:
            dimensions.append(("original_hash", str(original_hash)))
        for key in dimensions:
            previous = linked.setdefault(key, group)
            groups.union(group, previous)
    components: dict[str, set[str]] = defaultdict(set)
    for group in groups.parent:
        components[groups.find(group)].add(group)
    unknown = valid_groups - groups.parent.keys()
    if unknown:
        raise ValueError(f"--valid-group not found: {sorted(unknown)}")
    if train_only:
        if valid_groups:
            raise ValueError("--train-only cannot be combined with --valid-group")
        return {group: "train" for group in groups.parent}
    if len(components) < 2:
        raise ValueError("at least two unrelated source groups are required; use --train-only")
    heldout_roots = {groups.find(group) for group in valid_groups}
    if not heldout_roots:
        heldout_roots = {sorted(components)[-1]}
    if heldout_roots == set(components):
        raise ValueError("split would have no training group")
    return {
        group: "valid" if groups.find(group) in heldout_roots else "train"
        for group in groups.parent
    }


def _validate_published(
    release: Path, canonical: list[dict[str, Any]], parents: list[dict[str, Any]], input_count: int
) -> None:
    report = json.loads((release / "report.json").read_text(encoding="utf-8"))
    if report.get("schema_version") != "taigi-merged-weak-labels/v1":
        raise ValueError(f"existing merged release has the wrong schema: {release}")
    if report.get("release_id") != release.name or report.get("parents") != sorted(
        parents, key=lambda parent: parent["identity_sha256"]
    ):
        raise ValueError(f"existing merged release identity failed integrity check: {release}")
    counts = {
        split: sum(row["split"] == split for row in canonical) for split in ("train", "valid")
    }
    durations = {
        split: round(sum(float(row["duration"]) for row in canonical if row["split"] == split), 6)
        for split in ("train", "valid")
    }
    split_groups = {
        split: sorted({str(row["group_id"]) for row in canonical if row["split"] == split})
        for split in ("train", "valid")
    }
    expected_report = {
        "input_weak_label_count": input_count,
        "accepted_weak_label_count": len(canonical),
        "deduplicated_count": input_count - len(canonical),
        "counts": counts,
        "durations_seconds": durations,
        "split_groups": split_groups,
        "label_provenance": "Breeze-ASR-26 weak labels; unverified; no quality claim",
        "verified_labels": 0,
    }
    if any(report.get(key) != value for key, value in expected_report.items()):
        raise ValueError(f"existing merged release report failed integrity check: {release}")
    manifest_hashes = report.get("manifest_sha256")
    required_manifests = {"metadata.jsonl", "train.jsonl", "valid.jsonl"}
    if not isinstance(manifest_hashes, dict) or set(manifest_hashes) != required_manifests:
        raise ValueError(f"existing merged release has incomplete manifest hashes: {release}")
    for name, expected in manifest_hashes.items():
        path = release / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"existing merged release manifest failed integrity check: {path}")
    metadata = read_jsonl(release / "metadata.jsonl")
    suffixes: dict[str, str] = {}
    for row in canonical:
        digest, suffix = str(row["audio_sha256"]), str(row["_suffix"])
        suffixes[digest] = min(suffixes.get(digest, suffix), suffix)
    expected_metadata = [
        {
            **row,
            "published_audio_path": str(
                release / "audio" / f"{row['audio_sha256']}{suffixes[str(row['audio_sha256'])]}"
            ),
        }
        for row in canonical
    ]
    for row in expected_metadata:
        row.pop("_suffix")
    if metadata != expected_metadata:
        raise ValueError(f"existing merged release metadata is not the intended content: {release}")
    for row in metadata:
        audio = Path(row["published_audio_path"]).resolve()
        try:
            audio.relative_to((release / "audio").resolve())
        except ValueError as error:
            raise ValueError(f"existing merged release audio escapes release: {audio}") from error
        if not audio.is_file() or sha256_file(audio) != row["audio_sha256"]:
            raise ValueError(f"existing merged release audio failed integrity check: {audio}")
    if len(metadata) != report.get("accepted_weak_label_count"):
        raise ValueError("existing merged release metadata count failed integrity check")


def merge_releases(
    releases: list[Path],
    output_dir: Path,
    *,
    valid_groups: set[str] | None = None,
    train_only: bool = False,
) -> Path:
    if not releases:
        raise ValueError("at least one input release is required")
    all_rows: list[dict[str, Any]] = []
    parents: list[dict[str, Any]] = []
    for release in releases:
        rows, identity = _validate_release(release)
        all_rows.extend(rows)
        parents.append(identity)
    splits = _assign_splits(all_rows, valid_groups or set(), train_only)
    rows = _deduplicate(all_rows)
    if not rows:
        raise ValueError("input releases contain no accepted weak labels")
    canonical = []
    parent_by_path = {parent["path"]: parent["identity_sha256"] for parent in parents}
    for row in rows:
        canonical.append(
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "split",
                    "published_audio_path",
                    "source_audio_path",
                    "_input_audio_path",
                    "_parent_release",
                }
            }
            | {
                "parent_utterance_id": row["utterance_id"],
                "parent_release_identity_sha256": parent_by_path[row["_parent_release"]],
                "utterance_id": _merged_utterance_id(row),
                "split": splits[str(row["group_id"])],
                "_suffix": Path(row["_input_audio_path"]).suffix.lower() or ".wav",
            }
        )
    canonical.sort(key=lambda row: (row["split"], row["source_id"], row["utterance_id"]))
    if not train_only and {row["split"] for row in canonical} != {"train", "valid"}:
        raise ValueError("deduplication would leave an empty train or valid split")
    release_id = _json_digest(
        {"rows": canonical, "parents": sorted(p["identity_sha256"] for p in parents)}
    )[:16]
    output_dir = output_dir.expanduser().resolve()
    release = output_dir / release_id
    if release.exists():
        _validate_published(release, canonical, parents, len(all_rows))
        return release
    temporary = output_dir / f".{release_id}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        (temporary / "audio").mkdir()
        source_for_hash: dict[str, Path] = {}
        suffix_for_hash: dict[str, str] = {}
        for row in rows:
            digest = str(row["audio_sha256"])
            source = Path(row["_input_audio_path"])
            source_for_hash.setdefault(digest, source)
            suffix = source.suffix.lower() or ".wav"
            suffix_for_hash[digest] = min(suffix_for_hash.get(digest, suffix), suffix)
        for digest, source in source_for_hash.items():
            destination = temporary / "audio" / f"{digest}{suffix_for_hash[digest]}"
            shutil.copyfile(source, destination)
            if sha256_file(destination) != digest:
                raise ValueError(f"copied audio failed integrity check: {source}")
            destination.chmod(0o444)
        published_metadata: list[dict[str, Any]] = []
        for row in canonical:
            digest = str(row["audio_sha256"])
            published = release / "audio" / f"{digest}{suffix_for_hash[digest]}"
            published_row = {**row, "published_audio_path": str(published)}
            published_row.pop("_suffix")
            published_metadata.append(published_row)
        for split in ("train", "valid"):
            write_jsonl(
                temporary / f"{split}.jsonl",
                [
                    {
                        "path": row["published_audio_path"],
                        "start": row["start"],
                        "duration": row["duration"],
                        "transcript": row["input_text_zh"],
                    }
                    for row in published_metadata
                    if row["split"] == split
                ],
            )
        write_jsonl(temporary / "metadata.jsonl", published_metadata)
        counts = {
            split: sum(row["split"] == split for row in published_metadata)
            for split in ("train", "valid")
        }
        durations = {
            split: round(
                sum(float(row["duration"]) for row in published_metadata if row["split"] == split),
                6,
            )
            for split in ("train", "valid")
        }
        report = {
            "schema_version": "taigi-merged-weak-labels/v1",
            "release_id": release_id,
            "parents": sorted(parents, key=lambda parent: parent["identity_sha256"]),
            "input_weak_label_count": len(all_rows),
            "accepted_weak_label_count": len(published_metadata),
            "deduplicated_count": len(all_rows) - len(published_metadata),
            "counts": counts,
            "durations_seconds": durations,
            "split_groups": {
                split: sorted(
                    {str(row["group_id"]) for row in published_metadata if row["split"] == split}
                )
                for split in ("train", "valid")
            },
            "label_provenance": "Breeze-ASR-26 weak labels; unverified; no quality claim",
            "verified_labels": 0,
        }
        report["manifest_sha256"] = {
            name: sha256_file(temporary / name)
            for name in ("metadata.jsonl", "train.jsonl", "valid.jsonl")
        }
        (temporary / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary.rename(release)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return release


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge immutable corrected weak-label releases")
    parser.add_argument("releases", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--valid-group", action="append", default=[])
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args(argv)
    release = merge_releases(
        args.releases,
        args.output_dir,
        valid_groups=set(args.valid_group),
        train_only=args.train_only,
    )
    print(release)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
