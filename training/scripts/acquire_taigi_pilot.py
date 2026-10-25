"""Acquire a bounded ADI training sample for pipeline verification."""

import argparse
import io
import json
import tarfile
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download

REPO = "adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hokkien"
REVISION = "ffcc36b9afe31ec50f7742c0c4c614f1a5b40151"
SHARD = "train/train-000000.tar"


def acquire(output: Path, limit: int) -> Path:
    if not 1 <= limit <= 256:
        raise ValueError("pilot sample count must be between 1 and 256")
    output = output.resolve()
    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    archive = hf_hub_download(
        REPO, SHARD, repo_type="dataset", revision=REVISION, local_dir=output / "source"
    )
    records = []
    total_seconds = 0.0
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            metadata_file = tar.extractfile(member)
            if metadata_file is None:
                continue
            metadata = json.load(metadata_file)
            if metadata.get("locale") != "nan-tw":
                continue
            audio_file = tar.extractfile(member.name[:-5] + ".mp3")
            if audio_file is None:
                continue
            audio, sample_rate = sf.read(io.BytesIO(audio_file.read()), dtype="float32")
            duration = len(audio) / sample_rate
            if not 3 <= duration <= 12:
                continue
            path = audio_dir / (Path(member.name).stem + ".wav")
            sf.write(path, audio, sample_rate)
            total_seconds += duration
            records.append(
                {
                    "audio_path": str(path),
                    "source_id": "adi-" + Path(member.name).stem,
                    "group_id": metadata["client_id"],
                    "language": "nan-tw",
                    "source_repo": REPO,
                    "source_revision": REVISION,
                    "source_split": "train",
                    "source_license": "TRAIL-0.1",
                    "original_path": metadata["path"],
                    "duration": duration,
                }
            )
            if len(records) == limit:
                break
    if not records:
        raise ValueError("no qualifying Taiwanese training samples found")
    manifest = output / "sources.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"{len(records)} samples, {total_seconds:.3f} seconds")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/taigi-pilot"))
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    print(acquire(args.output, args.limit))


if __name__ == "__main__":
    main()
