"""Dictionary semantic release construction is exact, frozen, and split-safe."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from training.scripts import build_adi_semantic_release as semantic
from training.scripts import fork_taigi_adaptation as adaptation
from training.scripts.run_experiment import digest


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    specs = [
        ("train-a", "train", "speaker-a", "台語甲", "target-shared", 0.1),
        ("train-b", "train", "speaker-b", "台語乙", "target-train", 0.2),
        ("train-c", "train", "speaker-d", "台語丁", "target-other", 0.3),
        ("valid-a", "valid", "speaker-c", "台語丙", "target-shared", -0.1),
    ]
    dictionary = tmp_path / "kautian.ods"
    with zipfile.ZipFile(dictionary, "w") as archive:
        archive.writestr("mimetype", "application/vnd.oasis.opendocument.spreadsheet")
        rows = [("漢字", "華語詞目")] + [(row[3], row[4]) for row in specs]
        xml_rows = "".join(
            "<table:table-row>"
            + "".join(
                f"<table:table-cell><text:p>{cell}</text:p></table:table-cell>" for cell in row
            )
            + "</table:table-row>"
            for row in rows
        )
        archive.writestr(
            "content.xml",
            '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
            'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
            'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
            f'<table:table table:name="詞彙比較">{xml_rows}</table:table>'
            "</office:document-content>",
        )
    dictionary_hash = digest(dictionary)
    identity = {"revision": "adi-fixture-revision", "input": "fixture"}
    prepared = (
        tmp_path / hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    )
    (prepared / "audio").mkdir(parents=True)
    (prepared / "LICENSE").write_bytes(b"TRAIL fixture license")
    candidates, mappings = [], []
    for number, (name, split, speaker, native, mandarin, sample) in enumerate(specs, 1):
        audio = prepared / "audio" / f"{name}.wav"
        sf.write(audio, np.full(24000, sample, dtype=np.float32), 24000, subtype="FLOAT")
        source_path, sentence_id = f"clips/{name}.mp3", f"sentence-{number}"
        candidates.append(
            {
                "utterance_id": f"utterance-{name}",
                "source_id": f"source-{name}",
                "group_id": speaker,
                "audio_path": str(audio.relative_to(prepared)),
                "audio_sha256": digest(audio),
                "start": 0.0,
                "duration": 1.0,
                "split": split,
                "source_metadata": {
                    "original_metadata": {
                        "path": source_path,
                        "sentence_id": sentence_id,
                        "client_id": speaker,
                    },
                    "han_reference_display": native,
                    "source_audio_sha256": digest(audio),
                },
            }
        )
        mappings.append(
            _mapping(native, mandarin, dictionary_hash, source_path, sentence_id, speaker, number)
        )
    _jsonl(prepared / "candidates.jsonl", candidates)
    artifacts = {
        str(path.relative_to(prepared)): digest(path)
        for path in prepared.rglob("*")
        if path.is_file()
    }
    (prepared / "report.json").write_text(
        json.dumps({"state": "complete", "identity": identity, "artifacts": artifacts}),
        encoding="utf-8",
    )
    mapping_path = tmp_path / "weak-admitted-mappings.jsonl"
    _jsonl(mapping_path, mappings)
    diagnostics = tmp_path / "diagnostic-translations.jsonl"
    _jsonl(
        diagnostics,
        [
            {
                "native_text": f"控制台語{number}",
                "mandarin_candidate": f"控制華語{number}",
                "translation_status": "machine_translation_unverified",
                "human_verified": False,
                "audio_verified": False,
                "use": semantic.CONTROL_USE,
            }
            for number in range(64)
        ],
    )
    return prepared, mapping_path, dictionary, diagnostics


def _mapping(
    native: str, mandarin: str, source_hash: str, path: str, sentence: str, client: str, row: int
) -> dict[str, Any]:
    return {
        "native_text": native,
        "mandarin_text": mandarin,
        "mandarin_candidates": [mandarin],
        "status": semantic.ADMITTED_STATUS,
        "translation_status": semantic.TRANSLATION_STATUS,
        "human_verified": False,
        "audio_verified": False,
        "match_method": "exact_dictionary_parallel",
        "ambiguity_reasons": [],
        "license": semantic.DICTIONARY_LICENSE,
        "license_url": "https://example.invalid/license",
        "source_url": "https://example.invalid/kautian.ods",
        "source_sha256": source_hash,
        "references": [
            {
                "sheet": "詞彙比較",
                "ods_row": row + 1,
                "mandarin": mandarin,
                "row": {"漢字": native, "華語詞目": mandarin},
            }
        ],
        "source_paths": [path],
        "source_sentence_ids": [sentence],
        "source_client_ids": [client],
        "clip_count": 1,
        "limitations": ["not audio-semantics verified"],
    }


def test_all_admitted_rows_and_valid_wins_mandarin_target_collision(tmp_path: Path) -> None:
    prepared, mappings, dictionary, diagnostics = _fixture(tmp_path)
    release = semantic.build_release(
        prepared, mappings, dictionary, tmp_path / "published", diagnostics=diagnostics
    )

    report = json.loads((release / "report.json").read_text(encoding="utf-8"))
    assert (report["input_mapping_count"], report["input_weak_label_count"]) == (4, 4)
    assert report["counts"] == {"train": 2, "valid": 1}
    assert report["cross_split_mandarin_target_excluded"] == 1
    excluded = [json.loads(line) for line in (release / "excluded.jsonl").read_text().splitlines()]
    assert (excluded[0]["input_text_zh"], excluded[0]["split"]) == ("target-shared", "train")
    metadata = [json.loads(line) for line in (release / "metadata.jsonl").read_text().splitlines()]
    assert {row["input_text_zh"] for row in metadata} == {
        "target-train",
        "target-other",
        "target-shared",
    }
    assert all(row["label_type"] == "dictionary_parallel_weak" for row in metadata)
    for row in metadata:
        audio = release / row["published_audio_path"]
        assert audio.is_file() and not audio.is_symlink()
        assert digest(audio) == row["audio_sha256"]
    assert adaptation.verify_merged_release(release)["counts"] == {"train": 2, "valid": 1}


@pytest.mark.parametrize(
    "mutation", ["source_hash", "license", "proof", "proof_sheet", "proof_row", "native"]
)
def test_rejects_tampered_mapping_provenance_and_exact_join(tmp_path: Path, mutation: str) -> None:
    prepared, mappings, dictionary, diagnostics = _fixture(tmp_path)
    rows = [json.loads(line) for line in mappings.read_text().splitlines()]
    if mutation == "source_hash":
        rows[0]["source_sha256"] = "0" * 64
    elif mutation == "license":
        rows[0]["license"] = "unknown"
    elif mutation == "proof":
        rows[0]["references"] = []
    elif mutation == "proof_sheet":
        rows[0]["references"][0]["sheet"] = "NOT_AN_ODS_SHEET"
        rows[0]["references"][0]["ods_row"] = -999
    elif mutation == "proof_row":
        rows[0]["references"][0]["ods_row"] = 4
    else:
        rows[0]["native_text"] = "錯誤台語"
    _jsonl(mappings, rows)
    with pytest.raises(ValueError):
        semantic.build_release(
            prepared, mappings, dictionary, tmp_path / "published", diagnostics=diagnostics
        )


def test_frozen_mapping_bytes_are_reverified(tmp_path: Path) -> None:
    prepared, mappings, dictionary, diagnostics = _fixture(tmp_path)
    release = semantic.build_release(
        prepared, mappings, dictionary, tmp_path / "published", diagnostics=diagnostics
    )
    frozen = release / "dictionary-mappings.jsonl"
    frozen.chmod(0o644)
    frozen.write_text(frozen.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity|mapping bytes"):
        adaptation.verify_merged_release(release)


def test_records_mapping_clip_absent_from_frozen_prepared(tmp_path: Path) -> None:
    prepared, mappings, dictionary, diagnostics = _fixture(tmp_path)
    rows = [json.loads(line) for line in mappings.read_text().splitlines()]
    rows[0]["source_paths"].append("clips/preparation-excluded.mp3")
    rows[0]["clip_count"] = 2
    _jsonl(mappings, rows)

    release = semantic.build_release(
        prepared, mappings, dictionary, tmp_path / "published", diagnostics=diagnostics
    )

    report = json.loads((release / "report.json").read_text())
    assert report["input_weak_label_count"] == 5
    assert report["prepared_source_absent_excluded"] == 1
    assert report["cross_split_mandarin_target_excluded"] == 1
    adaptation.verify_merged_release(release)


def test_reserved_diagnostic_mandarin_alias_is_held_out_from_both_splits(tmp_path: Path) -> None:
    prepared, mappings, dictionary, diagnostics = _fixture(tmp_path)
    rows = [json.loads(line) for line in diagnostics.read_text().splitlines()]
    rows[0]["mandarin_candidate"] = "target-train"
    _jsonl(diagnostics, rows)

    release = semantic.build_release(
        prepared, mappings, dictionary, tmp_path / "published", diagnostics=diagnostics
    )

    report = json.loads((release / "report.json").read_text())
    assert report["diagnostic_mandarin_alias_excluded"] == 1
    assert report["counts"] == {"train": 1, "valid": 1}
    manifests = (release / "train.jsonl").read_text() + (release / "valid.jsonl").read_text()
    assert "target-train" not in manifests


def test_rejects_frozen_prepared_speaker_split_overlap(tmp_path: Path) -> None:
    prepared, mappings, dictionary, diagnostics = _fixture(tmp_path)
    candidates = [
        json.loads(line) for line in (prepared / "candidates.jsonl").read_text().splitlines()
    ]
    candidates[-1]["group_id"] = candidates[1]["group_id"]
    _jsonl(prepared / "candidates.jsonl", candidates)
    report = json.loads((prepared / "report.json").read_text())
    report["artifacts"]["candidates.jsonl"] = digest(prepared / "candidates.jsonl")
    (prepared / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="group_id overlap"):
        semantic.build_release(
            prepared, mappings, dictionary, tmp_path / "published", diagnostics=diagnostics
        )


def test_cli_builds_verified_release(tmp_path: Path) -> None:
    prepared, mappings, dictionary, diagnostics = _fixture(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.scripts.build_adi_semantic_release",
            "--prepared",
            str(prepared),
            "--mappings",
            str(mappings),
            "--dictionary-source",
            str(dictionary),
            "--output-dir",
            str(tmp_path / "published"),
            "--diagnostics",
            str(diagnostics),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    release = Path(result.stdout.strip())
    assert release.is_dir()
    assert adaptation.verify_merged_release(release)["schema_version"] == semantic.SCHEMA


def test_diagnostic_64_is_copied_but_quarantined_from_training(tmp_path: Path) -> None:
    prepared, mappings, dictionary, _ = _fixture(tmp_path)
    candidates = [
        json.loads(line) for line in (prepared / "candidates.jsonl").read_text().splitlines()
    ]
    control_audio = prepared / "audio" / "control.wav"
    sf.write(control_audio, np.full(24000, 0.45, dtype=np.float32), 24000, subtype="FLOAT")
    audio = str(control_audio.relative_to(prepared))
    audio_hash = digest(control_audio)
    diagnostics = []
    for number in range(64):
        native = f"控制台語{number}"
        path, sentence = f"clips/control-{number}.mp3", f"control-{number}"
        candidates.append(
            {
                **candidates[0],
                "utterance_id": f"control-{number}",
                "source_id": f"control-source-{number}",
                "audio_path": audio,
                "audio_sha256": audio_hash,
                "source_metadata": {
                    **candidates[0]["source_metadata"],
                    "original_metadata": {
                        "path": path,
                        "sentence_id": sentence,
                        "client_id": "control-speaker",
                    },
                    "han_reference_display": native,
                },
            }
        )
        diagnostics.append(
            {
                "native_text": native,
                "mandarin_candidate": f"控制華語{number}",
                "translation_status": "machine_translation_unverified",
                "human_verified": False,
                "audio_verified": False,
                "source_paths": [path],
                "source_sentence_ids": [sentence],
                "use": (semantic.TEXT_ONLY_CONTROL_USE if number == 63 else semantic.CONTROL_USE),
                "references": [] if number == 63 else [{"ods_row": number + 2}],
            }
        )
    _jsonl(prepared / "candidates.jsonl", candidates)
    report = json.loads((prepared / "report.json").read_text())
    report["artifacts"]["candidates.jsonl"] = digest(prepared / "candidates.jsonl")
    report["artifacts"][audio] = audio_hash
    (prepared / "report.json").write_text(json.dumps(report), encoding="utf-8")
    diagnostics_path = tmp_path / "diagnostic-translations.jsonl"
    _jsonl(diagnostics_path, diagnostics)

    release = semantic.build_release(
        prepared,
        mappings,
        dictionary,
        tmp_path / "published",
        diagnostics=diagnostics_path,
        control_selection="diagnostic-64",
    )

    release_report = json.loads((release / "report.json").read_text())
    assert (release_report["control_case_count"], release_report["control_clip_count"]) == (64, 64)
    controls = [json.loads(line) for line in (release / "control.jsonl").read_text().splitlines()]
    assert all((release / row["published_audio_path"]).is_file() for row in controls)
    quarantine = [
        json.loads(line) for line in (release / "quarantine.jsonl").read_text().splitlines()
    ]
    assert (
        sum(
            row["quarantine_reason"] == "machine_only_no_exact_parallel_proof" for row in quarantine
        )
        == 1
    )
    training = (release / "train.jsonl").read_text() + (release / "valid.jsonl").read_text()
    assert "控制華語" not in training
