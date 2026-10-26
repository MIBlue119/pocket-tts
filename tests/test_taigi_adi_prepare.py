import csv
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from training.scripts import prepare_adi_taigi as adi
from training.taigi.schema import read_jsonl


def audio_bytes(frequency: float, seconds: float = 1.0) -> bytes:
    stream = io.BytesIO()
    samples = (
        0.1 * np.sin(2 * np.pi * frequency * np.arange(round(seconds * 8000)) / 8000)
    ).astype(np.float32)
    sf.write(stream, samples, 8000, format="WAV", subtype="FLOAT")
    return stream.getvalue()


def write_archive(path: Path, split: str, rows: list[dict[str, Any]]) -> None:
    with tarfile.open(path, "w") as archive:
        for index, row in enumerate(rows):
            metadata = {k: v for k, v in row.items() if k != "audio"}
            metadata.update(audio_ext="mp3", split=split)
            for suffix, data in {
                "mp3": row["audio"],
                "txt": row["sentence"].encode(),
                "json": json.dumps(metadata, ensure_ascii=False).encode(),
            }.items():
                member = tarfile.TarInfo(f"{split}-{index:09d}.{suffix}")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))


def source_fixture(tmp_path: Path, repeated_training_text: bool = False) -> Path:
    raw = tmp_path / "raw"
    (raw / "train").mkdir(parents=True)
    (raw / "test").mkdir()
    rows = [
        {
            "client_id": f"speaker-{i}",
            "sentence_id": f"sentence-{i}",
            "sentence": f"中文句子{i}（tai-gi）",
            "locale": "nan-tw",
            "path": f"clips/common_voice_nan-tw_{i}.mp3",
            "audio": audio_bytes(220 + i * 20),
        }
        for i in range(9)
    ]
    rows[0]["client_id"] = "dev-client"
    rows[0]["sentence"] = "食薰（tsia?h-hun）"
    rows[1]["sentence"] = rows[0]["sentence"]
    rows[2].update(client_id="official-test-client", sentence="官方測試也有這句")
    rows[3]["audio"] = audio_bytes(999)
    rows[4]["locale"] = ""
    rows[5].update(locale="zh-cn", path="clips\\common_voice_zh-CN_5.mp3")
    rows[6]["path"] = "clips/common_voice_zh-CN_6.mp3"
    rows[7]["audio"] = audio_bytes(333, 0.4)
    if repeated_training_text:
        rows[8]["sentence"] = rows[2]["sentence"]
        rows[8]["sentence_id"] = rows[2]["sentence_id"]
    test = [
        {
            "client_id": "official-test-client",
            "sentence_id": "official-test-sentence",
            "sentence": "官方測試也有這句",
            "locale": "nan-tw",
            "path": "clips/common_voice_nan-tw_test.mp3",
            "audio": audio_bytes(999),
        }
    ]
    files = []
    for split, records in [("train", rows), ("test", test)]:
        path = raw / split / f"{split}-000000.tar"
        write_archive(path, split, records)
        files.append(
            {
                "path": str(path.relative_to(raw)),
                "size": path.stat().st_size,
                "lfs_sha256": adi.sha256_file(path),
            }
        )
        with (raw / f"{split}.tsv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["client_id", "path", "sentence_id", "sentence", "locale"],
                delimiter="\t",
            )
            writer.writeheader()
            for row in records:
                writer.writerow(
                    {
                        key: "zh-CN" if key == "locale" and row[key] == "zh-cn" else row[key]
                        for key in writer.fieldnames
                    }
                )
    (raw / "inventory.json").write_text(
        json.dumps({"repo_id": adi.REPO, "revision": adi.REVISION, "files": files})
    )
    (raw / "LICENSE").write_text("TRAIL exact test license bytes")
    return raw


def test_all_eligible_rows_preserve_native_reference_and_development_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = source_fixture(tmp_path)
    monkeypatch.setattr(
        adi,
        "speakers_split",
        lambda rows, seed: {
            row["raw_metadata"]["client_id"]: "valid"
            if row["raw_metadata"]["client_id"] == "dev-client"
            else "train"
            for row in rows
        },
    )
    release = adi.prepare(raw, tmp_path / "prepared")
    report = adi.verify_release(release)
    sources = read_jsonl(release / "sources.jsonl")
    candidates = read_jsonl(release / "candidates.jsonl")
    assert report["accepted"] == 3
    assert report["counts"] == {"valid": 1, "train": 2}
    assert report["inferred_locale_quarantined"] == 1
    assert report["exclusion_reasons"]["official_test_audio_overlap"] == 1
    assert report["exclusion_reasons"]["train_sentence_overlap_with_development"] == 1
    assert report["official_test_proxy_buckets"] == {"seen_speaker|seen_sentence": 1}
    assert any(row["original_metadata"]["client_id"] == "official-test-client" for row in sources)
    native = next(row for row in sources if row["split"] == "valid")
    assert native["native_transcript"] == "食薰（tsia?h-hun）"
    assert native["han_reference_display"] == "食薰"
    assert native["reference_is_Mandarin_translation"] is False
    assert "transcript" not in native and "input_text_zh" not in native
    assert len({row["audio_identity"]["decoded_pcm_sha256"] for row in sources}) == 3
    assert all(row["start"] == 0 and row["duration"] == 1 for row in candidates)
    train = [row for row in sources if row["split"] == "train"]
    valid = [row for row in sources if row["split"] == "valid"]
    assert not {row["group_id"] for row in train} & {row["group_id"] for row in valid}
    assert not {adi.text_key(row["native_transcript"]) for row in train} & {
        adi.text_key(row["native_transcript"]) for row in valid
    }
    for row in sources:
        audio, rate = sf.read(row["audio_path"])
        assert rate == 24000 and np.isfinite(audio).all()
    assert (release / "LICENSE").read_bytes() == (raw / "LICENSE").read_bytes()
    assert adi.prepare(raw, tmp_path / "prepared") == release
    Path(sources[0]["audio_path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact integrity"):
        adi.verify_release(release)


def test_literal_tsv_backslashes_and_corrupted_romanization_are_not_repaired(
    tmp_path: Path,
) -> None:
    raw = source_fixture(tmp_path)
    rows = adi.read_tsv(raw / "train.tsv")
    assert rows[0]["sentence"] == "食薰（tsia?h-hun）"
    assert b"?" in rows[0]["sentence"].encode()
    assert rows[5]["path"] == "clips\\common_voice_zh-CN_5.mp3"
    assert adi.reference_parts(rows[0]["sentence"])["romanization_annotation"] == "tsia?h-hun"
    malformed = tmp_path / "literal-backslash.tsv"
    malformed.write_text("client_id\\tpath\\tsentence_id\\tsentence\\tlocale\n")
    with pytest.raises(ValueError, match="literal tab"):
        adi.read_tsv(malformed)


@pytest.mark.parametrize(
    "locale,path,expected",
    [
        (123, "clips/common_voice_nan-tw_x.mp3", "invalid_locale_or_path"),
        ("nan-tw", "clips/common_voice_zh-CN_x.mp3", "locale_path_conflict"),
        ("", "clips/common_voice_nan-tw_x.mp3", "inferred_locale_quarantine"),
        ("zh-CN", "clips/common_voice_zh-CN_x.mp3", "mandarin_source_excluded"),
    ],
)
def test_corrupt_or_conflicting_locale_never_guesses_from_text(
    locale: object, path: str, expected: str
) -> None:
    assert (
        adi.cohort({"raw_metadata": {"locale": locale, "path": path, "sentence": "中文看起來一樣"}})
        == expected
    )


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../bad.json", "file"),
        ("/bad.json", "file"),
        ("link", "symlink"),
        ("same.json", "duplicate"),
    ],
)
def test_archive_rejects_unsafe_or_duplicate_members(tmp_path: Path, name: str, kind: str) -> None:
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as stream:
        member = tarfile.TarInfo(name)
        if kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "/outside"
        stream.addfile(member, io.BytesIO())
        if kind == "duplicate":
            stream.addfile(member, io.BytesIO())
    with pytest.raises(ValueError, match="unsafe|duplicate"):
        adi.archive_rows(archive, "train")
    assert not (tmp_path.parent / "bad.json").exists()


def test_duration_weighted_speaker_split_is_deterministic_and_keeps_groups_together() -> None:
    rows = [
        {"raw_metadata": {"client_id": f"speaker-{i // 2}"}, "audio_identity": {"duration": 1.0}}
        for i in range(200)
    ]
    first = adi.speakers_split(rows, 0)
    assert first == adi.speakers_split(list(reversed(rows)), 0)
    assert set(first.values()) == {"train", "valid"}
    assert sum(value == "valid" for value in first.values()) == 10


def test_distinct_recordings_of_repeated_training_phrase_are_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = source_fixture(tmp_path, repeated_training_text=True)
    monkeypatch.setattr(
        adi,
        "speakers_split",
        lambda rows, seed: {
            row["raw_metadata"]["client_id"]: "valid"
            if row["raw_metadata"]["client_id"] == "dev-client"
            else "train"
            for row in rows
        },
    )
    release = adi.prepare(raw, tmp_path / "prepared")
    train = [row for row in read_jsonl(release / "sources.jsonl") if row["split"] == "train"]
    assert len(train) == 2
    assert train[0]["native_transcript"] == train[1]["native_transcript"]
    assert (
        train[0]["audio_identity"]["decoded_pcm_sha256"]
        != train[1]["audio_identity"]["decoded_pcm_sha256"]
    )
    assert adi.verify_release(release)["unique_native_texts_by_split"]["train"] == 1


def test_worker_count_preserves_selection_order_pcm_and_split(tmp_path: Path) -> None:
    raw = source_fixture(tmp_path)
    first = adi.prepare(raw, tmp_path / "one", workers=1)
    second = adi.prepare(raw, tmp_path / "four", workers=4)

    def identity(path: Path) -> list[tuple[str, str, str]]:
        return [
            (row["source_id"], row["split"], row["audio_identity"]["decoded_pcm_sha256"])
            for row in read_jsonl(path / "sources.jsonl")
        ]

    assert identity(first) == identity(second)
    assert first.name == second.name
    assert adi.verify_release(first)["cpu_worker_limit"] == 1
    assert adi.verify_release(second)["cpu_worker_limit"] == 4


def test_resident_asr_keeps_native_reference_separate_and_pins_completed_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from training.taigi import corpus_worker

    raw = source_fixture(tmp_path)
    release = adi.prepare(raw, tmp_path / "prepared")
    loads = []
    calls = []

    class ASR:
        target_sample_rate = 16000

        def transcribe_batch(self, intervals: list[tuple[Path, float, float]]) -> list[str]:
            calls.append(intervals)
            return ["這是中文語意候選"] * len(intervals)

        def __call__(self, path: Path, start: float, duration: float) -> str:
            raise AssertionError("prefilled cache must serve individual label reads")

    monkeypatch.setattr(corpus_worker, "breeze_transcriber", lambda _: loads.append(1) or ASR())
    output = tmp_path / "asr"
    weak = adi.transcribe(release, output, "cpu", 4)
    assert len(loads) == len(calls) == 1
    rows = read_jsonl(weak / "metadata.jsonl")
    assert all(
        row["input_text_zh"] == "這是中文語意候選" and row["label_type"] == "asr_pseudo"
        for row in rows
    )
    assert all(row["source_metadata"]["native_transcript"] != row["input_text_zh"] for row in rows)
    assert adi.transcribe(release, output, "cpu", 4) == weak
    assert len(loads) == 1
    (weak / "train.jsonl").write_text("corrupt")
    with pytest.raises(ValueError, match="manifest integrity"):
        adi.transcribe(release, output, "cpu", 4)
    assert len(loads) == 1
