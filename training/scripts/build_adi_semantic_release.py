"""Build a frozen ADI release from exact MOE dictionary-parallel mappings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import unicodedata
import zipfile
from xml.etree import ElementTree
from pathlib import Path
from typing import Any

from training.scripts.prepare_adi_taigi import path_key, verify_release
from training.scripts.run_experiment import digest, write_json
from training.taigi.schema import read_jsonl, write_jsonl

SCHEMA = "taigi-dictionary-semantic-pairs/v1"
LABEL_TYPE = "dictionary_parallel_weak"
LABEL_PROVENANCE = "MOE Taiwanese dictionary exact parallel weak mapping"
VERIFICATION = "dictionary_matched_not_audio_semantics_verified"
DICTIONARY_LICENSE = "CC BY-ND 3.0 TW"
ADMITTED_STATUS = "weak_admit_heuristic"
TRANSLATION_STATUS = "dictionary_parallel_unverified"
CONTROL_USE = "reserved_semantic_diagnostic_control_not_training"
TEXT_ONLY_CONTROL_USE = "text_semantic_control_only"
TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
OFFICE_NS = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"


def text_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _string(row: dict[str, Any], field: str, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} has invalid {field}")
    return value


def _sha(value: object, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{context} has invalid sha256")
    return value


def _source_path(row: dict[str, Any], context: str) -> str:
    metadata = row.get("source_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{context} has invalid source metadata")
    original = metadata.get("original_metadata")
    if not isinstance(original, dict):
        raise ValueError(f"{context} has invalid original metadata")
    return path_key(_string(original, "path", context))


def _candidate_index(prepared: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    report = verify_release(prepared)
    candidates = read_jsonl(prepared / "candidates.jsonl")
    index: dict[str, dict[str, Any]] = {}
    for number, candidate in enumerate(candidates, 1):
        source_path = _source_path(candidate, f"candidate {number}")
        audio_path = _string(candidate, "audio_path", f"candidate {number}")
        candidate_audio = Path(audio_path)
        if candidate_audio.is_absolute():
            try:
                artifact_path = str(candidate_audio.resolve().relative_to(prepared.resolve()))
            except ValueError as error:
                raise ValueError(f"candidate {number} audio escapes prepared release") from error
        else:
            artifact_path = str(candidate_audio)
        if report.get("artifacts", {}).get(artifact_path) != candidate.get("audio_sha256"):
            raise ValueError(f"candidate {number} audio is absent from prepared artifact identity")
        if source_path in index:
            raise ValueError(f"prepared candidates repeat source path: {source_path}")
        index[source_path] = candidate
    return index, report


def _load_dictionary_rows(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    try:
        with zipfile.ZipFile(path) as archive:
            if "content.xml" not in archive.namelist() or archive.read("mimetype") != (
                b"application/vnd.oasis.opendocument.spreadsheet"
            ):
                raise ValueError("dictionary source is not an ODS spreadsheet")
            root = ElementTree.fromstring(archive.read("content.xml"))
    except (ElementTree.ParseError, KeyError, zipfile.BadZipFile) as error:
        raise ValueError("dictionary source is not an ODS spreadsheet") from error
    result: dict[tuple[str, int], dict[str, str]] = {}
    for table in root.iter(f"{{{TABLE_NS}}}table"):
        sheet = table.get(f"{{{TABLE_NS}}}name", "")
        expanded: list[list[str]] = []
        for xml_row in table.findall(f"{{{TABLE_NS}}}table-row"):
            values: list[str] = []
            for cell in list(xml_row):
                if cell.tag not in {
                    f"{{{TABLE_NS}}}table-cell",
                    f"{{{TABLE_NS}}}covered-table-cell",
                }:
                    continue
                value = "\n".join(
                    "".join(paragraph.itertext()) for paragraph in cell.findall(f"{{{TEXT_NS}}}p")
                )
                if not value:
                    value = cell.get(f"{{{OFFICE_NS}}}value", "")
                repeat = int(cell.get(f"{{{TABLE_NS}}}number-columns-repeated", "1"))
                values.extend([value] * repeat)
            repeat = int(xml_row.get(f"{{{TABLE_NS}}}number-rows-repeated", "1"))
            expanded.extend([values] * repeat)
        if not expanded:
            continue
        headers = expanded[0]
        for ods_row, values in enumerate(expanded[1:], 2):
            result[(sheet, ods_row)] = {
                header: values[index] if index < len(values) else ""
                for index, header in enumerate(headers)
                if header
            }
    return result


def _validate_mapping(
    row: dict[str, Any],
    number: int,
    dictionary_sha256: str,
    dictionary_rows: dict[tuple[str, int], dict[str, str]],
) -> tuple[str, str, list[str]]:
    context = f"mapping row {number}"
    native = _string(row, "native_text", context)
    mandarin = _string(row, "mandarin_text", context)
    if (
        row.get("status") != ADMITTED_STATUS
        or row.get("translation_status") != TRANSLATION_STATUS
        or row.get("human_verified") is not False
        or row.get("audio_verified") is not False
        or row.get("ambiguity_reasons") != []
    ):
        raise ValueError(f"{context} is not an unambiguous admitted weak mapping")
    candidates = row.get("mandarin_candidates")
    if candidates != [mandarin]:
        raise ValueError(f"{context} must have one exact Mandarin candidate")
    if row.get("license") != DICTIONARY_LICENSE:
        raise ValueError(f"{context} has unsupported dictionary license")
    if _sha(row.get("source_sha256"), context) != dictionary_sha256:
        raise ValueError(f"{context} dictionary source hash mismatch")
    references = row.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError(f"{context} lacks dictionary row proof")
    for reference in references:
        if not isinstance(reference, dict) or reference.get("mandarin") != mandarin:
            raise ValueError(f"{context} has inconsistent dictionary row proof")
        sheet, ods_row, source_row = (
            reference.get("sheet"),
            reference.get("ods_row"),
            reference.get("row"),
        )
        if (
            sheet not in {"詞彙比較", "例句"}
            or not isinstance(ods_row, int)
            or ods_row < 2
            or not isinstance(source_row, dict)
        ):
            raise ValueError(f"{context} dictionary proof lacks an ODS row")
        native_cell = source_row.get("漢字")
        mandarin_cell = source_row.get("華語詞目" if sheet == "詞彙比較" else "華語")
        actual_row = dictionary_rows.get((sheet, ods_row))
        if (
            text_key(str(native_cell or "")) != text_key(native)
            or mandarin_cell != mandarin
            or actual_row is None
            or any(actual_row.get(key) != value for key, value in source_row.items())
        ):
            raise ValueError(f"{context} dictionary proof cells disagree with the mapping")
    paths = row.get("source_paths")
    sentence_ids = row.get("source_sentence_ids")
    if not isinstance(paths, list) or not paths or not all(isinstance(v, str) for v in paths):
        raise ValueError(f"{context} has invalid source paths")
    if (
        not isinstance(sentence_ids, list)
        or not sentence_ids
        or not all(isinstance(v, str) and v for v in sentence_ids)
    ):
        raise ValueError(f"{context} has invalid source sentence IDs")
    normalized_paths = [path_key(value) for value in paths]
    if len(set(normalized_paths)) != len(normalized_paths):
        raise ValueError(f"{context} repeats a source path")
    if row.get("clip_count") != len(normalized_paths):
        raise ValueError(f"{context} clip count disagrees with source paths")
    return native, mandarin, normalized_paths


def _join(
    mapping: dict[str, Any], native: str, paths: list[str], index: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    sentence_ids = set(mapping["source_sentence_ids"])
    client_ids = set(mapping.get("source_client_ids", []))
    joined = []
    for source_path in paths:
        candidate = index.get(source_path)
        if candidate is None:
            raise ValueError(
                f"mapping source path is absent from prepared candidates: {source_path}"
            )
        metadata = candidate["source_metadata"]
        original = metadata["original_metadata"]
        if original.get("sentence_id") not in sentence_ids:
            raise ValueError(f"mapping sentence ID mismatch for {source_path}")
        if client_ids and original.get("client_id") not in client_ids:
            raise ValueError(f"mapping client ID mismatch for {source_path}")
        if text_key(str(metadata.get("han_reference_display", ""))) != text_key(native):
            raise ValueError(f"mapping native reference mismatch for {source_path}")
        joined.append(candidate)
    return joined


def _copy_audio(prepared: Path, temporary: Path, candidate: dict[str, Any]) -> str:
    source = (prepared / str(candidate["audio_path"])).resolve()
    try:
        source.relative_to((prepared / "audio").resolve())
    except ValueError as error:
        raise ValueError(f"prepared audio escapes release: {source}") from error
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"prepared audio is not a regular internal file: {source}")
    expected = _sha(candidate.get("audio_sha256"), "prepared candidate")
    if digest(source) != expected:
        raise ValueError(f"prepared audio hash mismatch: {source}")
    suffix = source.suffix.lower() or ".wav"
    relative = Path("audio") / f"{expected}{suffix}"
    destination = temporary / relative
    if not destination.exists():
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o444)
    if destination.is_symlink() or digest(destination) != expected:
        raise ValueError(f"copied audio hash mismatch: {destination}")
    return str(relative)


def _metadata_row(
    prepared: Path,
    temporary: Path,
    release: Path,
    candidate: dict[str, Any],
    mapping: dict[str, Any],
    mapping_sha256: str,
    dictionary_sha256: str,
) -> dict[str, Any]:
    published = str(release / _copy_audio(prepared, temporary, candidate))
    proof = {
        "native_text": mapping["native_text"],
        "mandarin_text": mapping["mandarin_text"],
        "match_method": mapping.get("match_method"),
        "references": mapping["references"],
        "source_url": mapping.get("source_url"),
        "license_url": mapping.get("license_url"),
        "limitations": mapping.get("limitations", []),
    }
    return {
        "utterance_id": f"adi-dict-{candidate['utterance_id']}",
        "source_id": candidate["source_id"],
        "group_id": candidate["group_id"],
        "published_audio_path": published,
        "audio_sha256": candidate["audio_sha256"],
        "start": candidate["start"],
        "duration": candidate["duration"],
        "input_text_zh": mapping["mandarin_text"],
        "split": candidate["split"],
        "source_metadata": {
            **candidate["source_metadata"],
            "prepared_utterance_id": candidate["utterance_id"],
            "dictionary_mapping": proof,
            "mapping_source_sha256": mapping_sha256,
            "dictionary_source_sha256": dictionary_sha256,
            "dictionary_license": DICTIONARY_LICENSE,
        },
        "label_type": LABEL_TYPE,
        "label_provenance": LABEL_PROVENANCE,
        "verification": VERIFICATION,
        "human_verified": False,
        "audio_semantics_verified": False,
    }


def _control_rows(
    diagnostics: Path | None,
    selection: str,
    index: dict[str, dict[str, Any]],
    prepared: Path,
    temporary: Path,
    release: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if selection == "none":
        return [], []
    if diagnostics is None:
        raise ValueError("diagnostic-64 control selection requires --diagnostics")
    rows = read_jsonl(diagnostics)
    if len(rows) != 64:
        raise ValueError("diagnostic-64 selection requires exactly 64 diagnostic cases")
    controls, quarantine = [], []
    seen_native: set[str] = set()
    for number, row in enumerate(rows, 1):
        native = _string(row, "native_text", f"diagnostic row {number}")
        if text_key(native) in seen_native:
            raise ValueError("diagnostic controls repeat a native key")
        seen_native.add(text_key(native))
        use = row.get("use")
        if use not in {CONTROL_USE, TEXT_ONLY_CONTROL_USE}:
            raise ValueError(f"diagnostic row {number} is not reserved from training")
        paths = row.get("source_paths")
        sentence_ids = row.get("source_sentence_ids")
        if not isinstance(paths, list) or not isinstance(sentence_ids, list):
            raise ValueError(f"diagnostic row {number} lacks exact source identities")
        mapping = {**row, "source_sentence_ids": sentence_ids}
        unavailable_paths = []
        for source_path in [path_key(str(v)) for v in paths]:
            if source_path not in index:
                unavailable_paths.append(source_path)
                continue
            candidate = _join(mapping, native, [source_path], index)[0]
            published_audio_path = str(release / _copy_audio(prepared, temporary, candidate))
            controls.append(
                {
                    "native_text": native,
                    "mandarin_candidate": row.get("mandarin_candidate"),
                    "utterance_id": candidate["utterance_id"],
                    "source_id": candidate["source_id"],
                    "group_id": candidate["group_id"],
                    "published_audio_path": published_audio_path,
                    "audio_sha256": candidate["audio_sha256"],
                    "start": candidate["start"],
                    "duration": candidate["duration"],
                    "split": candidate["split"],
                    "use": use,
                }
            )
        quarantine.append(
            {
                **row,
                "unavailable_source_paths": unavailable_paths,
                "quarantine_reason": (
                    "machine_only_no_exact_parallel_proof"
                    if not row.get("references")
                    else "diagnostic_control_not_training"
                ),
            }
        )
    return controls, quarantine


def _diagnostic_targets(path: Path) -> set[str]:
    rows = read_jsonl(path)
    if len(rows) != 64:
        raise ValueError("semantic release requires exactly 64 reserved diagnostic cases")
    targets = set()
    for number, row in enumerate(rows, 1):
        candidate = _string(row, "mandarin_candidate", f"diagnostic row {number}")
        if (
            row.get("translation_status") != "machine_translation_unverified"
            or row.get("human_verified") is not False
            or row.get("audio_verified") is not False
            or row.get("use") not in {CONTROL_USE, TEXT_ONLY_CONTROL_USE}
        ):
            raise ValueError(f"diagnostic row {number} violates the reserved-control contract")
        targets.add(text_key(candidate))
    return targets


def build_release(
    prepared: Path,
    mappings: Path,
    dictionary_source: Path,
    output_dir: Path,
    *,
    diagnostics: Path,
    control_selection: str = "none",
) -> Path:
    prepared, mappings = prepared.resolve(), mappings.resolve()
    dictionary_source, output_dir = dictionary_source.resolve(), output_dir.resolve()
    if control_selection not in {"none", "diagnostic-64"}:
        raise ValueError("unsupported control selection")
    index, prepared_report = _candidate_index(prepared)
    mapping_rows = read_jsonl(mappings)
    if not mapping_rows:
        raise ValueError("admitted mappings are empty")
    if not dictionary_source.is_file() or dictionary_source.is_symlink():
        raise ValueError("dictionary source must be a regular file")
    dictionary_rows = _load_dictionary_rows(dictionary_source)
    dictionary_sha256, mapping_sha256 = digest(dictionary_source), digest(mappings)
    diagnostics = diagnostics.resolve()
    diagnostic_targets = _diagnostic_targets(diagnostics)
    diagnostics_sha256 = digest(diagnostics)
    identity = {
        "schema_version": SCHEMA,
        "prepared_report_sha256": digest(prepared / "report.json"),
        "prepared_candidates_sha256": digest(prepared / "candidates.jsonl"),
        "mapping_sha256": mapping_sha256,
        "dictionary_source_sha256": dictionary_sha256,
        "diagnostics_sha256": diagnostics_sha256,
        "control_selection": control_selection,
    }
    release_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    release = output_dir / release_id
    if release.exists():
        from training.scripts.fork_taigi_adaptation import verify_merged_release

        verify_merged_release(release)
        return release
    temporary = output_dir / f".{release_id}.partial"
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    (temporary / "audio").mkdir()
    used_paths: set[str] = set()
    published, unavailable = [], []
    try:
        for number, mapping in enumerate(mapping_rows, 1):
            native, _, paths = _validate_mapping(
                mapping, number, dictionary_sha256, dictionary_rows
            )
            for source_path in paths:
                if source_path in used_paths:
                    raise ValueError(f"source path occurs in multiple mappings: {source_path}")
                used_paths.add(source_path)
            for source_path in paths:
                if source_path not in index:
                    unavailable.append(
                        {
                            "native_text": native,
                            "mandarin_text": mapping["mandarin_text"],
                            "source_path": source_path,
                            "reason": "mapping_source_absent_from_frozen_prepared",
                        }
                    )
                    continue
                candidate = _join(mapping, native, [source_path], index)[0]
                published.append(
                    _metadata_row(
                        prepared,
                        temporary,
                        release,
                        candidate,
                        mapping,
                        mapping_sha256,
                        dictionary_sha256,
                    )
                )
        input_clip_count = sum(int(row["clip_count"]) for row in mapping_rows)
        if len(published) + len(unavailable) != input_clip_count:
            raise ValueError("joined clip accounting does not match admitted mappings")
        diagnostic_excluded = [
            {**row, "reason": "diagnostic_mandarin_alias_holdout"}
            for row in published
            if text_key(row["input_text_zh"]) in diagnostic_targets
        ]
        diagnostic_ids = {row["utterance_id"] for row in diagnostic_excluded}
        published = [row for row in published if row["utterance_id"] not in diagnostic_ids]
        valid_targets = {
            text_key(row["input_text_zh"]) for row in published if row["split"] == "valid"
        }
        target_excluded = [
            {**row, "reason": "mandarin_target_cross_split_valid_precedence"}
            for row in published
            if row["split"] == "train" and text_key(row["input_text_zh"]) in valid_targets
        ]
        excluded = unavailable + diagnostic_excluded + target_excluded
        excluded_ids = {row["utterance_id"] for row in target_excluded}
        published = [row for row in published if row["utterance_id"] not in excluded_ids]
        by_split = {
            split: sorted(
                (row for row in published if row["split"] == split),
                key=lambda row: row["utterance_id"],
            )
            for split in ("train", "valid")
        }
        if any(not rows for rows in by_split.values()):
            raise ValueError("dictionary release requires nonempty train and valid splits")
        for field in ("audio_sha256", "group_id", "source_id"):
            left = {str(row[field]) for row in by_split["train"]}
            right = {str(row[field]) for row in by_split["valid"]}
            if left & right:
                raise ValueError(f"prepared split contract has cross-split {field} overlap")
        native = {
            split: {
                text_key(str(row["source_metadata"]["dictionary_mapping"]["native_text"]))
                for row in rows
            }
            for split, rows in by_split.items()
        }
        if native["train"] & native["valid"]:
            raise ValueError("prepared split contract has cross-split native reference overlap")
        controls, quarantine = _control_rows(
            diagnostics, control_selection, index, prepared, temporary, release
        )
        for field in ("audio_sha256", "source_id"):
            if {str(row[field]) for row in controls} & {str(row[field]) for row in published}:
                raise ValueError(f"diagnostic controls overlap training data by {field}")
        training_native = {
            text_key(str(row["source_metadata"]["dictionary_mapping"]["native_text"]))
            for row in published
        }
        if training_native & {text_key(str(row["native_text"])) for row in quarantine}:
            raise ValueError("diagnostic controls overlap training data by native reference")
        write_jsonl(temporary / "metadata.jsonl", by_split["train"] + by_split["valid"])
        for split, rows in by_split.items():
            write_jsonl(
                temporary / f"{split}.jsonl",
                [
                    {
                        "path": row["published_audio_path"],
                        "start": row["start"],
                        "duration": row["duration"],
                        "transcript": row["input_text_zh"],
                    }
                    for row in rows
                ],
            )
        write_jsonl(temporary / "excluded.jsonl", excluded)
        if control_selection == "diagnostic-64":
            write_jsonl(temporary / "control.jsonl", controls)
            write_jsonl(temporary / "quarantine.jsonl", quarantine)
        shutil.copyfile(prepared / "LICENSE", temporary / "ADI-LICENSE")
        shutil.copyfile(prepared / "report.json", temporary / "prepared-report.json")
        shutil.copyfile(mappings, temporary / "dictionary-mappings.jsonl")
        shutil.copyfile(dictionary_source, temporary / "dictionary-source.ods")
        shutil.copyfile(diagnostics, temporary / "diagnostic-translations.jsonl")
        evidence = [
            "ADI-LICENSE",
            "prepared-report.json",
            "dictionary-mappings.jsonl",
            "dictionary-source.ods",
        ]
        evidence.append("diagnostic-translations.jsonl")
        manifests = ["metadata.jsonl", "train.jsonl", "valid.jsonl", "excluded.jsonl", *evidence]
        if control_selection == "diagnostic-64":
            manifests += ["control.jsonl", "quarantine.jsonl"]
        counts = {split: len(rows) for split, rows in by_split.items()}
        durations = {
            split: round(sum(float(row["duration"]) for row in rows), 6)
            for split, rows in by_split.items()
        }
        report = {
            "schema_version": SCHEMA,
            "release_id": release_id,
            "identity": identity,
            "selection": "all-admitted-exact-parallel-mappings",
            "control_selection": control_selection,
            "prepared_release": {
                "release_id": prepared.name,
                "report_sha256": identity["prepared_report_sha256"],
                "candidates_sha256": identity["prepared_candidates_sha256"],
                "license_sha256": digest(prepared / "LICENSE"),
                "source_revision": prepared_report["identity"]["revision"],
            },
            "dictionary_source": {
                "sha256": dictionary_sha256,
                "license": DICTIONARY_LICENSE,
                "mapping_sha256": mapping_sha256,
            },
            "input_mapping_count": len(mapping_rows),
            "input_weak_label_count": input_clip_count,
            "accepted_weak_label_count": sum(counts.values()),
            "deduplicated_count": len(excluded),
            "counts": counts,
            "durations_seconds": durations,
            "split_groups": {
                split: sorted({str(row["group_id"]) for row in rows})
                for split, rows in by_split.items()
            },
            "cross_split_mandarin_target_excluded": len(target_excluded),
            "prepared_source_absent_excluded": len(unavailable),
            "diagnostic_mandarin_alias_excluded": len(diagnostic_excluded),
            "control_case_count": len(quarantine),
            "control_clip_count": len(controls),
            "label_type": LABEL_TYPE,
            "label_provenance": LABEL_PROVENANCE,
            "verification": VERIFICATION,
            "verified_labels": 0,
            "quality_verified": False,
            "manifest_sha256": {name: digest(temporary / name) for name in sorted(manifests)},
            "limitations": [
                "Dictionary matches are weak semantic labels, not audio-semantic verification.",
                "Client IDs are speaker proxies; same-speaker controls do not test unseen-speaker cloning.",
                "Diagnostic controls are quarantined and are not fabricated acoustic gold.",
            ],
        }
        write_json(temporary / "report.json", report)
        for artifact in temporary.rglob("*"):
            if artifact.is_file():
                os.chmod(artifact, 0o444)
        temporary.rename(release)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    from training.scripts.fork_taigi_adaptation import verify_merged_release

    try:
        verify_merged_release(release)
    except BaseException:
        shutil.rmtree(release, ignore_errors=True)
        raise
    return release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--dictionary-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--control-selection", choices=("none", "diagnostic-64"), default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release = build_release(
        args.prepared,
        args.mappings,
        args.dictionary_source,
        args.output_dir,
        diagnostics=args.diagnostics,
        control_selection=args.control_selection,
    )
    print(release)


if __name__ == "__main__":
    main()
