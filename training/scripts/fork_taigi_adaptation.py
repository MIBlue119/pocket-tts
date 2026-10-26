"""Prepare a fresh data-adaptation run from a completed Taigi checkpoint.

This command only creates an immutable child configuration and copied restart
state. It does not start a trainer or make a quality-selection decision.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from training.args import TrainArgs, _from_dict
from training.scripts.build_adi_semantic_release import (
    DICTIONARY_LICENSE,
    LABEL_PROVENANCE as DICTIONARY_LABEL_PROVENANCE,
    LABEL_TYPE as DICTIONARY_LABEL_TYPE,
    SCHEMA as DICTIONARY_SCHEMA,
    VERIFICATION as DICTIONARY_VERIFICATION,
    _load_dictionary_rows,
    _validate_mapping,
    text_key,
)
from training.scripts import continue_taigi_training as continuation
from training.scripts.run_experiment import ROOT, digest, identity, run_lock, write_json
from training.taigi.schema import read_jsonl
from training.taigi.transcriber import BREEZE_ADAPTER_VERSION, BREEZE_MODEL, BREEZE_REVISION

MANDATORY_CONFIG_CHANGES = {"run_dir", "max_steps", "data.train_jsonl", "data.valid_jsonl"}
RECIPE_CONFIG_CHANGES = {
    "batch_size": "batch_size",
    "grad_accum_steps": "grad_accum_steps",
    "flow_batch_multiplier": "flow_batch_multiplier",
    "learning_rate": "optim.lr",
    "precompute": "data.precompute",
}


def _number(value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected a finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"expected a finite positive number, got {value!r}")
    return result


def _file_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": digest(path)}


def _manifest_row(
    release: Path, split: str, line: int, row: dict[str, Any]
) -> tuple[tuple[object, ...], Path]:
    required = {"path", "start", "duration", "transcript"}
    if set(row) != required:
        raise ValueError(
            f"{split} manifest row {line} must have exactly {sorted(required)}, got {sorted(row)}"
        )
    path_value, transcript = row["path"], row["transcript"]
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"{split} manifest row {line} has an invalid audio path")
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError(f"{split} manifest row {line} has an invalid transcript")
    start = _number(row["start"])
    duration = _number(row["duration"], positive=True)
    if start < 0:
        raise ValueError(f"{split} manifest row {line} has a negative start")
    audio = Path(path_value)
    if not audio.is_absolute():
        audio = release / audio
    audio = audio.resolve()
    try:
        audio.relative_to((release / "audio").resolve())
    except ValueError as error:
        raise ValueError(f"{split} manifest audio escapes release: {audio}") from error
    if not audio.is_file():
        raise ValueError(f"{split} manifest references missing audio: {audio}")
    try:
        import soundfile as sf

        info = sf.info(audio)
    except (ImportError, OSError, RuntimeError) as error:
        raise ValueError(f"cannot inspect release audio: {audio}") from error
    seconds = info.frames / info.samplerate if info.samplerate else 0.0
    if start + duration > seconds + 1e-9:
        raise ValueError(f"{split} manifest interval exceeds audio duration: {audio}")
    signature = (path_value, row["start"], row["duration"], transcript)
    return signature, audio


def _overlap(left: set[str], right: set[str], name: str) -> None:
    shared = left & right
    if shared:
        preview = sorted(shared)[:3]
        raise ValueError(f"train/valid {name} overlap: {preview}")


def verify_merged_release(path: Path) -> dict[str, Any]:
    """Verify the frozen merged release consumed by one adaptation fork."""

    release = path.expanduser().resolve()
    report_path = release / "report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read merged release report: {report_path}") from error
    schema = report.get("schema_version")
    if schema not in {"taigi-merged-weak-labels/v1", DICTIONARY_SCHEMA}:
        raise ValueError(f"unsupported merged release schema: {report.get('schema_version')!r}")
    if report.get("release_id") != release.name:
        raise ValueError("merged release identity does not match its directory")
    if report.get("verified_labels") != 0:
        raise ValueError("merged release must remain unverified weak-label data")
    if schema == "taigi-merged-weak-labels/v1":
        if report.get("label_provenance") != (
            "Breeze-ASR-26 weak labels; unverified; no quality claim"
        ):
            raise ValueError("merged release has unsupported label provenance")
        if not isinstance(report.get("parents"), list) or not report["parents"]:
            raise ValueError("merged release report has no parent identities")
    else:
        release_identity = report.get("identity")
        dictionary = report.get("dictionary_source")
        prepared = report.get("prepared_release")
        if (
            not isinstance(release_identity, dict)
            or hashlib.sha256(json.dumps(release_identity, sort_keys=True).encode()).hexdigest()[
                :20
            ]
            != release.name
            or release_identity.get("schema_version") != DICTIONARY_SCHEMA
            or report.get("selection") != "all-admitted-exact-parallel-mappings"
            or report.get("label_type") != DICTIONARY_LABEL_TYPE
            or report.get("label_provenance") != DICTIONARY_LABEL_PROVENANCE
            or report.get("verification") != DICTIONARY_VERIFICATION
            or report.get("quality_verified") is not False
            or not isinstance(dictionary, dict)
            or dictionary.get("license") != DICTIONARY_LICENSE
            or not isinstance(prepared, dict)
        ):
            raise ValueError("dictionary release has unsupported source/provenance contract")
        for record, field in (
            (dictionary, "sha256"),
            (dictionary, "mapping_sha256"),
            (prepared, "report_sha256"),
            (prepared, "candidates_sha256"),
            (prepared, "license_sha256"),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", str(record.get(field, ""))):
                raise ValueError("dictionary release has an invalid source identity")

    required_files = {"metadata.jsonl", "train.jsonl", "valid.jsonl"}
    hashes = report.get("manifest_sha256")
    if not isinstance(hashes, dict) or not required_files.issubset(hashes):
        raise ValueError("merged release report has incomplete manifest hashes")
    if schema == "taigi-merged-weak-labels/v1" and set(hashes) != required_files:
        raise ValueError("merged Breeze release has unexpected manifest hashes")
    if schema == DICTIONARY_SCHEMA:
        dictionary_files = {
            "ADI-LICENSE",
            "prepared-report.json",
            "dictionary-mappings.jsonl",
            "dictionary-source.ods",
            "excluded.jsonl",
        }
        if not dictionary_files.issubset(hashes):
            raise ValueError("dictionary release is missing frozen source evidence")
        if digest(release / "dictionary-source.ods") != report["dictionary_source"]["sha256"]:
            raise ValueError("dictionary source bytes do not match provenance")
        if (
            digest(release / "dictionary-mappings.jsonl")
            != report["dictionary_source"]["mapping_sha256"]
        ):
            raise ValueError("dictionary mapping bytes do not match provenance")
        if digest(release / "ADI-LICENSE") != report["prepared_release"]["license_sha256"]:
            raise ValueError("ADI license bytes do not match provenance")
        if digest(release / "prepared-report.json") != report["prepared_release"]["report_sha256"]:
            raise ValueError("prepared report bytes do not match provenance")
    for name in sorted(hashes):
        manifest = release / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("merged release report contains an unsafe manifest path")
        if not manifest.is_file() or digest(manifest) != hashes[name]:
            raise ValueError(f"merged release manifest failed integrity: {manifest}")
    if schema == DICTIONARY_SCHEMA:
        mapping_rows = read_jsonl(release / "dictionary-mappings.jsonl")
        dictionary_rows = _load_dictionary_rows(release / "dictionary-source.ods")
        mapping_clips = 0
        for line, mapping in enumerate(mapping_rows, 1):
            _validate_mapping(mapping, line, report["dictionary_source"]["sha256"], dictionary_rows)
            paths = mapping.get("source_paths")
            references = mapping.get("references")
            if (
                mapping.get("status") != "weak_admit_heuristic"
                or mapping.get("translation_status") != "dictionary_parallel_unverified"
                or mapping.get("human_verified") is not False
                or mapping.get("audio_verified") is not False
                or mapping.get("ambiguity_reasons") != []
                or mapping.get("license") != DICTIONARY_LICENSE
                or mapping.get("source_sha256") != report["dictionary_source"]["sha256"]
                or not isinstance(paths, list)
                or not paths
                or mapping.get("clip_count") != len(paths)
                or not isinstance(references, list)
                or not references
            ):
                raise ValueError(f"dictionary mapping row {line} violates the source contract")
            mapping_clips += len(paths)
        if (
            report.get("input_mapping_count") != len(mapping_rows)
            or report.get("input_weak_label_count") != mapping_clips
        ):
            raise ValueError("dictionary report input counts do not match frozen mappings")
        excluded = read_jsonl(release / "excluded.jsonl")
        if len(excluded) != report.get("deduplicated_count") or any(
            row.get("reason")
            not in {
                "mandarin_target_cross_split_valid_precedence",
                "mapping_source_absent_from_frozen_prepared",
                "diagnostic_mandarin_alias_holdout",
            }
            for row in excluded
        ):
            raise ValueError("dictionary target-overlap exclusions are inconsistent")
        absent = sum(
            row.get("reason") == "mapping_source_absent_from_frozen_prepared" for row in excluded
        )
        diagnostic_aliases = sum(
            row.get("reason") == "diagnostic_mandarin_alias_holdout" for row in excluded
        )
        if (
            report.get("prepared_source_absent_excluded") != absent
            or report.get("diagnostic_mandarin_alias_excluded") != diagnostic_aliases
            or report.get("cross_split_mandarin_target_excluded")
            != len(excluded) - absent - diagnostic_aliases
        ):
            raise ValueError("dictionary exclusion reason counts are inconsistent")

    manifests = {split: read_jsonl(release / f"{split}.jsonl") for split in ("train", "valid")}
    for split, rows in manifests.items():
        if not rows:
            raise ValueError(f"merged release {split} manifest is empty")

    compact: dict[str, Counter[tuple[object, ...]]] = {}
    for split, rows in manifests.items():
        parsed = [_manifest_row(release, split, line, row) for line, row in enumerate(rows, 1)]
        compact[split] = Counter(signature for signature, _ in parsed)

    metadata = read_jsonl(release / "metadata.jsonl")
    if len(metadata) != sum(len(rows) for rows in manifests.values()):
        raise ValueError("merged release metadata count does not match manifests")
    identities = {
        split: {"audio": set(), "group": set(), "source": set(), "original_audio": set()}
        for split in ("train", "valid")
    }
    audio_hashes: dict[Path, str] = {}
    durations = {"train": 0.0, "valid": 0.0}
    for line, row in enumerate(metadata, 1):
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
            "label_type",
            "label_provenance",
            "verification",
        }
        if schema == "taigi-merged-weak-labels/v1":
            required |= {"label_model", "label_model_revision", "label_adapter_version"}
        else:
            required |= {"human_verified", "audio_semantics_verified"}
        missing = required - row.keys()
        if missing:
            raise ValueError(f"merged metadata row {line} is missing {sorted(missing)}")
        split = row["split"]
        if split not in manifests:
            raise ValueError(f"merged metadata row {line} has invalid split {split!r}")
        if not isinstance(row["source_metadata"], dict):
            raise TypeError(f"merged metadata row {line} has invalid source_metadata")
        if schema == "taigi-merged-weak-labels/v1":
            valid_provenance = (
                row["label_type"] == "asr_pseudo"
                and row["label_provenance"] == "Breeze-ASR-26"
                and row["label_model"] == BREEZE_MODEL
                and row["label_model_revision"] == BREEZE_REVISION
                and row["label_adapter_version"] == BREEZE_ADAPTER_VERSION
                and row["verification"] == "unverified_weak_label"
            )
        else:
            source_metadata = row["source_metadata"]
            proof = source_metadata.get("dictionary_mapping")
            valid_provenance = (
                row["label_type"] == DICTIONARY_LABEL_TYPE
                and row["label_provenance"] == DICTIONARY_LABEL_PROVENANCE
                and row["verification"] == DICTIONARY_VERIFICATION
                and row["human_verified"] is False
                and row["audio_semantics_verified"] is False
                and source_metadata.get("dictionary_license") == DICTIONARY_LICENSE
                and source_metadata.get("dictionary_source_sha256")
                == report["dictionary_source"]["sha256"]
                and source_metadata.get("mapping_source_sha256")
                == report["dictionary_source"]["mapping_sha256"]
                and isinstance(proof, dict)
                and isinstance(proof.get("native_text"), str)
                and bool(proof["native_text"].strip())
                and proof.get("mandarin_text") == row["input_text_zh"]
                and isinstance(proof.get("references"), list)
                and bool(proof["references"])
                and all(
                    isinstance(reference, dict)
                    and reference.get("mandarin") == row["input_text_zh"]
                    and isinstance(reference.get("ods_row"), int)
                    and reference["ods_row"] >= 2
                    and reference.get("sheet") in {"詞彙比較", "例句"}
                    and isinstance(reference.get("row"), dict)
                    and text_key(str(reference["row"].get("漢字", "")))
                    == text_key(proof["native_text"])
                    and reference["row"].get(
                        "華語詞目" if reference["sheet"] == "詞彙比較" else "華語"
                    )
                    == row["input_text_zh"]
                    for reference in proof["references"]
                )
            )
        if not valid_provenance:
            raise ValueError(f"merged metadata row {line} has unsupported weak-label provenance")
        for field in ("utterance_id", "source_id", "group_id"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"merged metadata row {line} has invalid {field}")
        signature, audio = _manifest_row(
            release,
            str(split),
            line,
            {
                "path": row["published_audio_path"],
                "start": row["start"],
                "duration": row["duration"],
                "transcript": row["input_text_zh"],
            },
        )
        if compact[str(split)][signature] <= 0:
            raise ValueError(f"merged metadata row {line} does not match its compact manifest")
        compact[str(split)][signature] -= 1
        expected_hash = str(row["audio_sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError(f"merged metadata row {line} has an invalid audio hash")
        if audio not in audio_hashes:
            audio_hashes[audio] = digest(audio)
        actual_hash = audio_hashes[audio]
        if actual_hash != expected_hash:
            raise ValueError(f"merged release audio hash mismatch: {audio}")
        durations[str(split)] += float(row["duration"])
        values = identities[str(split)]
        values["audio"].add(expected_hash)
        values["group"].add(str(row["group_id"]))
        values["source"].add(str(row["source_id"]))
        original_hash = row["source_metadata"].get("source_audio_sha256")
        if original_hash:
            values["original_audio"].add(str(original_hash))
    if any(count for rows in compact.values() for count in rows.values()):
        raise ValueError("merged compact manifests contain rows absent from metadata")

    expected_counts = {split: len(rows) for split, rows in manifests.items()}
    expected_durations = {split: round(value, 6) for split, value in durations.items()}
    expected_groups = {split: sorted(identities[split]["group"]) for split in ("train", "valid")}
    if report.get("counts") != expected_counts:
        raise ValueError("merged release report counts do not match manifests")
    if report.get("accepted_weak_label_count") != sum(expected_counts.values()):
        raise ValueError("merged release accepted count does not match manifests")
    input_count = report.get("input_weak_label_count")
    if (
        type(input_count) is not int
        or input_count < sum(expected_counts.values())
        or report.get("deduplicated_count") != input_count - sum(expected_counts.values())
    ):
        raise ValueError("merged release input/deduplication counts are inconsistent")
    if report.get("durations_seconds") != expected_durations:
        raise ValueError("merged release report durations do not match metadata")
    if report.get("split_groups") != expected_groups:
        raise ValueError("merged release report split groups do not match metadata")
    for name in ("audio", "group", "source", "original_audio"):
        _overlap(identities["train"][name], identities["valid"][name], name)
    if schema == DICTIONARY_SCHEMA:
        targets = {
            split: {
                text_key(str(row["input_text_zh"])) for row in metadata if row["split"] == split
            }
            for split in ("train", "valid")
        }
        _overlap(targets["train"], targets["valid"], "normalized Mandarin target")

    files = {name: _file_record(release / name) for name in sorted(required_files)}
    audio_identity = json.dumps(sorted(audio_hashes.values()), separators=(",", ":")).encode()
    return {
        "path": str(release),
        "release_id": release.name,
        "schema_version": report["schema_version"],
        "report": _file_record(report_path),
        "files": files,
        "counts": expected_counts,
        "durations_seconds": expected_durations,
        "audio_file_count": len(audio_hashes),
        "audio_identity_sha256": hashlib.sha256(audio_identity).hexdigest(),
    }


def _leaves(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        if not value:
            return {prefix: {}}
        result: dict[str, object] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_leaves(item, path))
        return result
    return {prefix: value}


def config_changes(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, dict[str, Any]]:
    before, after = _leaves(parent), _leaves(child)
    changes = {}
    for path in sorted(before.keys() | after.keys()):
        if path not in before or path not in after or before[path] != after[path]:
            changes[path] = {
                "parent": {"present": path in before, "value": before.get(path)},
                "child": {"present": path in after, "value": after.get(path)},
            }
    return changes


def verify_config_changes(
    parent: dict[str, Any], child: dict[str, Any], allowed: set[str]
) -> dict[str, dict[str, Any]]:
    changes = config_changes(parent, child)
    unexpected = set(changes) - allowed
    if unexpected:
        raise ValueError(f"unapproved config changes: {sorted(unexpected)}")
    return changes


def _positive_int(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or type(value) is not int or value <= 0):
        raise ValueError(f"{name} must be a positive integer")


def fork_adaptation(
    parent_config: Path,
    release: Path,
    child: Path,
    endpoint: int,
    *,
    batch_size: int | None = None,
    grad_accum_steps: int | None = None,
    flow_batch_multiplier: int | None = None,
    learning_rate: float | None = None,
    precompute: bool | None = None,
) -> Path:
    """Mint one fresh child; omitted recipe values are inherited exactly."""

    _positive_int("endpoint", endpoint)
    for name, value in (
        ("batch_size", batch_size),
        ("grad_accum_steps", grad_accum_steps),
        ("flow_batch_multiplier", flow_batch_multiplier),
    ):
        _positive_int(name, value)
    if learning_rate is not None:
        _number(learning_rate, positive=True)
    if precompute is not None and type(precompute) is not bool:
        raise ValueError("precompute must be a boolean when supplied")

    parent_config = parent_config.expanduser().resolve()
    child = child.expanduser().resolve()
    parent, _ = continuation.verify_parent(parent_config)
    parent_run = (ROOT / parent["run_dir"]).resolve()
    release_identity = verify_merged_release(release)
    release = Path(release_identity["path"])
    if child == parent_run or child.is_relative_to(parent_run):
        raise ValueError("adaptation child must not be inside the parent run")

    created = False
    with run_lock(parent_run):
        try:
            parent, parent_identity = continuation.verify_parent(parent_config)
            if (ROOT / parent["run_dir"]).resolve() != parent_run:
                raise ValueError("parent run directory changed before fork")
            checkpoint, optimizer, parent_step = continuation.verify_pair(parent_run)
            if endpoint <= parent_step:
                raise ValueError("adaptation endpoint must exceed parent step")

            config = copy.deepcopy(parent)
            config["run_dir"] = str(child / "training")
            config["max_steps"] = endpoint
            config.setdefault("data", {})["train_jsonl"] = str(child / "data" / "train.jsonl")
            config["data"]["valid_jsonl"] = str(child / "data" / "valid.jsonl")
            requested = {
                "batch_size": batch_size,
                "grad_accum_steps": grad_accum_steps,
                "flow_batch_multiplier": flow_batch_multiplier,
                "learning_rate": learning_rate,
                "precompute": precompute,
            }
            if batch_size is not None:
                config["batch_size"] = batch_size
            if grad_accum_steps is not None:
                config["grad_accum_steps"] = grad_accum_steps
            if flow_batch_multiplier is not None:
                config["flow_batch_multiplier"] = flow_batch_multiplier
            if learning_rate is not None:
                config.setdefault("optim", {})["lr"] = learning_rate
            if precompute is not None:
                config["data"]["precompute"] = precompute
            # Exercise the same dataclass validation as the trainer before creating
            # the child. In particular, sentence sampling cannot consume latent caches.
            _from_dict(TrainArgs, config)
            allowed = MANDATORY_CONFIG_CHANGES | {
                RECIPE_CONFIG_CHANGES[name]
                for name, value in requested.items()
                if value is not None
            }
            changes = verify_config_changes(parent, config, allowed)

            tokenizer = (
                ROOT / config["model_overrides"]["flow_lm.lookup_table.tokenizer_path"]
            ).resolve()
            source_pair = {
                "checkpoint": _file_record(checkpoint),
                "optimizer": _file_record(optimizer),
            }
            child.mkdir(parents=True, exist_ok=False)
            created = True
            run_dir = child / "training"
            run_dir.mkdir()
            data_dir = child / "data"
            data_dir.mkdir()
            copied_manifests = {}
            for split in ("train", "valid"):
                source = release / f"{split}.jsonl"
                target = data_dir / f"{split}.jsonl"
                shutil.copyfile(source, target)
                target.chmod(0o444)
                copied_manifests[split] = _file_record(target)
                if (
                    copied_manifests[split]["sha256"]
                    != release_identity["files"][f"{split}.jsonl"]["sha256"]
                ):
                    raise ValueError(f"copied {split} manifest changed during child preparation")
            copied_pair = {}
            for name, source in (("checkpoint", checkpoint), ("optimizer", optimizer)):
                target = run_dir / source.name
                shutil.copyfile(source, target)
                copied_pair[name] = _file_record(target)
                if copied_pair[name]["sha256"] != source_pair[name]["sha256"]:
                    raise ValueError(f"copied {name} changed during child preparation")

            config_path = child / "training.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            provenance = {
                "parent_config": _file_record(parent_config),
                "parent_checkpoint": source_pair["checkpoint"],
                "parent_optimizer": source_pair["optimizer"],
                "release_report": release_identity["report"],
                "release_metadata": release_identity["files"]["metadata.jsonl"],
                "release_train": release_identity["files"]["train.jsonl"],
                "release_valid": release_identity["files"]["valid.jsonl"],
                "tokenizer": _file_record(tokenizer),
            }
            lineage = {
                "schema_version": "taigi-fresh-data-adaptation/v1",
                "parent_config": provenance["parent_config"],
                "parent_identity": parent_identity,
                "source_pair": source_pair,
                "copied_pair": copied_pair,
                "parent_step": parent_step,
                "endpoint": endpoint,
                "release": release_identity,
                "copied_manifests": copied_manifests,
                "tokenizer": provenance["tokenizer"],
                "tokenizer_continuity": {
                    "n_bins": config.get("model_overrides", {}).get("flow_lm.lookup_table.n_bins"),
                    "reset_text_embedding_setting_preserved": config.get("reset_text_embedding"),
                    "resume_uses_copied_checkpoint_model_state": True,
                },
                "allowed_config_changes": sorted(allowed),
                "config_changes": changes,
                "recipe": {
                    "requested_overrides": {
                        name: value for name, value in requested.items() if value is not None
                    },
                    "resolved": {
                        "batch_size": config.get("batch_size"),
                        "grad_accum_steps": config.get("grad_accum_steps"),
                        "flow_batch_multiplier": config.get("flow_batch_multiplier"),
                        "learning_rate": config.get("optim", {}).get("lr"),
                        "precompute": config.get("data", {}).get("precompute"),
                    },
                    "selection_claim": None,
                },
                "launcher_provenance": provenance,
                "resume_semantics": (
                    "model/optimizer/EMA restored at the cumulative parent step; tokenizer and "
                    "embedding IDs are unchanged; data-loader RNG restarts by step"
                ),
                "quality_verified": False,
                "quality_claim": "none; merged labels remain unverified weak labels",
            }
            lineage_path = child / "lineage.json"
            write_json(lineage_path, lineage)
            current = identity(config_path, config, [str(tokenizer), str(lineage_path)])
            write_json(run_dir / "run_manifest.json", {"schema_version": 1, "identity": current})
            continuation.verify_parent(config_path)
            return config_path
        except BaseException:
            if created:
                shutil.rmtree(child)
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a fresh Taigi data-adaptation fork without starting training"
    )
    parser.add_argument("--parent-config", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--endpoint", type=int, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--flow-batch-multiplier", type=int)
    parser.add_argument("--learning-rate", type=float)
    precompute = parser.add_mutually_exclusive_group()
    precompute.add_argument("--precompute", dest="precompute", action="store_true")
    precompute.add_argument("--no-precompute", dest="precompute", action="store_false")
    parser.set_defaults(precompute=None)
    args = parser.parse_args(argv)
    config = fork_adaptation(
        args.parent_config,
        args.release,
        args.child,
        args.endpoint,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        flow_batch_multiplier=args.flow_batch_multiplier,
        learning_rate=args.learning_rate,
        precompute=args.precompute,
    )
    print(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
