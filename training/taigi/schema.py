from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    language: Literal["nan-tw"]
    audio_path: Path | None = None
    url: str | None = None
    group_id: str | None = None
    split: Literal["train", "valid"] | None = None

    @property
    def group(self: SourceRecord) -> str:
        return self.group_id or self.source_id


@dataclass(frozen=True)
class SegmentRecord:
    utterance_id: str
    source_id: str
    group_id: str
    audio_path: str
    start: float
    duration: float
    audio_sha256: str
    language: Literal["nan-tw"] = "nan-tw"
    language_provenance: Literal["source_declared"] = "source_declared"
    audio_variant: Literal["raw"] = "raw"
    split: Literal["train", "valid"] | None = None
    status: Literal["candidate", "quarantined"] = "candidate"
    quarantine_reason: str | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each line must be a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, records: list[dict[str, Any]] | list[SegmentRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            value = asdict(record) if isinstance(record, SegmentRecord) else record
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
