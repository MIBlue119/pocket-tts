#!/usr/bin/env python3
"""Stage the certified one-speaker ADI semantic control; never launch training."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from training.args import load_args
from training.scripts.fork_taigi_adaptation import verify_merged_release
from training.scripts.run_experiment import digest
from training.taigi.schema import read_jsonl

ROOT = Path("/mnt/HDD2/proj_tts/pocket-tts")
RELEASE = ROOT / "data/taigi-adi-20261026/semantic-releases/59fa3a25a6cbc4c0d891"
TOKENIZER = ROOT / "data/taigi-full-corpus/production/recovery-20261026-r2/job-200/tokenizer.json"
SELECTION = ROOT / "data/taigi-adi-20261026/diagnostics/h100-source-asr-v1/selection.jsonl"
GENERAL = ROOT / "data/taigi-full-corpus/production/recovery-20261026-r2/job-200/prompts.jsonl"
OUTPUT = ROOT / "data/taigi-adi-20261026/semantic-control-20261026"
GROUP = "adi:ffcc36b9afe31ec50f7742c0c4c614f1a5b40151:speaker:7148c1ddc1ad16beb838"
TOKENIZER_SHA256 = "1019fc7f25a85e29e95c88ef36d09b1dfedab59a32ebadb8c495595fe6fa9d07"
EXPECTED_RELEASE_COUNTS = {"train": 867, "valid": 79}
EXPECTED_CATEGORIES = {"seen": 8, "same_speaker_reserved": 16, "speaker_heldout": 7}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def artifact_hashes(directory: Path, names: list[str]) -> dict[str, str]:
    return {name: digest(directory / name) for name in sorted(names)}


def prompt_rows(selection: Path) -> dict[str, list[dict[str, Any]]]:
    rows = read_jsonl(selection)
    categories = Counter(str(row.get("category")) for row in rows)
    if len(rows) != 31 or dict(categories) != EXPECTED_CATEGORIES:
        raise ValueError(f"diagnostic selection differs from 8/16/7 contract: {categories}")
    result = {category: [] for category in EXPECTED_CATEGORIES}
    result["supplemental"] = []
    seen_targets: set[str] = set()
    for index, row in enumerate(rows):
        category = str(row["category"])
        target, native = row.get("mandarin_target"), row.get("native_reference")
        if not isinstance(target, str) or not target.strip() or not isinstance(native, str):
            raise ValueError(f"diagnostic selection row {index + 1} lacks target/reference")
        if row.get("human_verified") is not False or row.get("audio_verified") is not False:
            raise ValueError("diagnostic prompts must retain unverified status")
        result[category].append(
            {
                "id": f"{category}-{len(result[category]):02d}",
                "text": target,
                "category": category,
                "native_reference": native,
                "source_utterance_id": row.get("utterance_id"),
                "source_audio_sha256": row.get("audio_sha256"),
                "human_verified": False,
                "audio_verified": False,
            }
        )
        seen_targets.add(target)
    if "你好嗎？" not in seen_targets:
        result["supplemental"].append(
            {
                "id": "supplemental-ni-hao-ma",
                "text": "你好嗎？",
                "category": "supplemental_exact_text",
                "native_reference": None,
                "human_verified": False,
                "audio_verified": False,
            }
        )
    return result


def training_config(
    train: Path, valid: Path, tokenizer: Path, run_dir: Path, max_steps: int
) -> dict[str, Any]:
    return {
        "model_config": "pocket_tts/config/english_2026-04_24l.yaml",
        "model_overrides": {
            "flow_lm.lookup_table.tokenizer_path": str(tokenizer),
            "flow_lm.lookup_table.n_bins": 4096,
        },
        "start_from_pretrained": True,
        "reset_text_embedding": True,
        "freeze_head": False,
        "data": {
            "train_jsonl": str(train),
            "valid_jsonl": str(valid),
            "sampling_mode": "sentence",
            "precompute": False,
            "max_duration_sec": 20.0,
            "loader_procs": 1,
            "num_bucket_batches": 1,
            "max_voice_prompt_sec": 5.0,
            "prompt_trim_max_sec": 0.0,
            "final_punct_dropout": 0.0,
        },
        "flow": {"type": "lsd", "kwargs": {}},
        "flow_batch_multiplier": 4,
        "eos_loss_weight": 0.1,
        "text_dropout": 0.2,
        "voice_dropout": 0.0,
        "optim": {
            "type": "adamw",
            "lr": 0.0002,
            "schedule": "constant",
            "warmup_steps": 100,
            "max_norm": 1.0,
        },
        "run_dir": str(run_dir),
        "batch_size": 16,
        "grad_accum_steps": 4,
        "max_steps": max_steps,
        "compile": False,
        "stats_ema_steps": 0,
        "ema_decay": 0.999,
        "seed": 42,
        "log_freq": 10,
        "valid_freq": 200,
        "num_valid_batches": 4,
        "ckpt_freq": 100,
        "num_ckpt_keep": 3,
        "sample_sentences": [],
        "sample_temp": 0.3,
        "sample_cfg_coef": 2.0,
    }


def commands(
    config: Path, output: Path, run_dir: Path, provenance: Path, tokenizer: Path
) -> dict[str, str]:
    base = "python -m training.taigi.evaluate"
    return {
        "baseline0_seen_required": (
            f"{base} {config} {output / 'seen-prompts.jsonl'} "
            f"{output / 'baseline0-seen'} --device cuda:0 --max-frames 100 --seed 17 "
            "--cfg-coef 2.0 --asr"
        ),
        "baseline0_reserved_optional": (
            f"{base} {config} {output / 'reserved-prompts.jsonl'} "
            f"{output / 'baseline0-reserved'} --device cuda:0 --max-frames 100 --seed 17 --cfg-coef 2.0"
        ),
        "baseline0_supplemental_optional": (
            f"{base} {config} {output / 'supplemental-prompts.jsonl'} "
            f"{output / 'baseline0-supplemental'} --device cuda:0 --max-frames 100 --seed 17 "
            "--cfg-coef 2.0"
        ),
        "baseline0_heldout_optional": (
            f"{base} {config} {output / 'heldout-prompts.jsonl'} "
            f"{output / 'baseline0-heldout'} --device cuda:0 --max-frames 100 --seed 17 --cfg-coef 2.0"
        ),
        "baseline0_general_optional": (
            f"{base} {config} {output / 'general-prompts.jsonl'} "
            f"{output / 'baseline0-general'} --device cuda:0 --max-frames 125 --seed 17 --cfg-coef 2.0"
        ),
        "training200": (
            "python -m training.scripts.run_experiment run "
            f"{config} --max-seconds 5400 --grace-seconds 120 --wandb offline "
            f"--provenance {tokenizer} --provenance {provenance}"
        ),
        "checkpoint200": str(run_dir / "checkpoint_00000200.pt"),
    }


def stage(args: argparse.Namespace) -> Path:
    release, tokenizer = args.release.resolve(), args.tokenizer.resolve()
    selection, general = args.selection.resolve(), args.general_prompts.resolve()
    output, run_dir = args.output.resolve(), args.run_dir.resolve()
    if args.max_steps != 200:
        raise ValueError("this bounded staging script only prepares stage 200")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("stage-200 run directory must be fresh and empty")
    release_identity = verify_merged_release(release)
    if release.name != "59fa3a25a6cbc4c0d891" or release_identity["counts"] != (
        EXPECTED_RELEASE_COUNTS
    ):
        raise ValueError("semantic release identity/counts differ from the certified control input")
    if digest(tokenizer) != TOKENIZER_SHA256:
        raise ValueError("tokenizer differs from the certified existing 4096-token tokenizer")
    general_rows = read_jsonl(general)
    if len(general_rows) != 12:
        raise ValueError("general prompt panel must contain exactly 12 rows")
    prompts = prompt_rows(selection)

    metadata = read_jsonl(release / "metadata.jsonl")
    selected = [
        row for row in metadata if row.get("split") == "train" and row.get("group_id") == GROUP
    ]
    if (
        len(selected) != 284
        or len({row["input_text_zh"] for row in selected}) != 258
        or round(sum(float(row["duration"]) for row in selected), 6) != 603.504
    ):
        raise ValueError("top-speaker subset differs from certified 284/258/603.504 contract")
    train_rows = [
        {
            "path": row["published_audio_path"],
            "start": row["start"],
            "duration": row["duration"],
            "transcript": row["input_text_zh"],
        }
        for row in selected
    ]
    valid_rows = read_jsonl(release / "valid.jsonl")
    if len(valid_rows) != 79:
        raise ValueError("certified validation split must contain exactly 79 rows")

    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        for name, expected in provenance["artifact_sha256"].items():
            if digest(output / name) != expected:
                raise ValueError(f"existing staged artifact changed: {name}")
        return provenance_path
    owned = [
        "train.jsonl",
        "valid.jsonl",
        "subset-metadata.jsonl",
        "seen-prompts.jsonl",
        "supplemental-prompts.jsonl",
        "reserved-prompts.jsonl",
        "heldout-prompts.jsonl",
        "general-prompts.jsonl",
        "stage-200.yaml",
        "commands.json",
    ]
    if any((output / name).exists() for name in owned):
        raise ValueError("staging outputs already exist without verified provenance")
    temporary = output / ".setup.partial"
    temporary.mkdir(exist_ok=False)
    try:
        write_jsonl(temporary / "train.jsonl", train_rows)
        shutil.copyfile(release / "valid.jsonl", temporary / "valid.jsonl")
        write_jsonl(temporary / "subset-metadata.jsonl", selected)
        write_jsonl(temporary / "seen-prompts.jsonl", prompts["seen"])
        write_jsonl(temporary / "supplemental-prompts.jsonl", prompts["supplemental"])
        write_jsonl(temporary / "reserved-prompts.jsonl", prompts["same_speaker_reserved"])
        write_jsonl(temporary / "heldout-prompts.jsonl", prompts["speaker_heldout"])
        shutil.copyfile(general, temporary / "general-prompts.jsonl")
        config_path = output / "stage-200.yaml"
        config = training_config(
            output / "train.jsonl", output / "valid.jsonl", tokenizer, run_dir, args.max_steps
        )
        (temporary / "stage-200.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        command_values = commands(config_path, output, run_dir, provenance_path, tokenizer)
        (temporary / "commands.json").write_text(
            json.dumps(command_values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Validate the exact final-path config without loading a model or allocating a GPU.
        load_args(temporary / "stage-200.yaml")
        hashes = artifact_hashes(temporary, owned)
        provenance = {
            "schema_version": "taigi-h100-semantic-control/v1",
            "quality_verified": False,
            "training_started": False,
            "release": {
                "path": str(release),
                "release_id": release.name,
                "report_sha256": digest(release / "report.json"),
                "metadata_sha256": digest(release / "metadata.jsonl"),
                "train_sha256": digest(release / "train.jsonl"),
                "valid_sha256": digest(release / "valid.jsonl"),
            },
            "tokenizer": {"path": str(tokenizer), "sha256": digest(tokenizer), "n_bins": 4096},
            "prompt_sources": {
                "selection": {"path": str(selection), "sha256": digest(selection)},
                "general": {"path": str(general), "sha256": digest(general)},
            },
            "selection": {
                "group_id": GROUP,
                "train_rows": 284,
                "distinct_mandarin_targets": 258,
                "duration_seconds": 603.504,
                "valid_rows": 79,
                "prompt_counts": {
                    "selected_seen": 8,
                    "supplemental_exact_text": len(prompts["supplemental"]),
                    "same_speaker_reserved": 16,
                    "speaker_heldout": 7,
                    "general": 12,
                },
            },
            "artifact_sha256": hashes,
            "commands": command_values,
        }
        (temporary / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for name in [*owned, "provenance.json"]:
            os.replace(temporary / name, output / name)
            os.chmod(output / name, 0o444)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return provenance_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=RELEASE)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--general-prompts", type=Path, default=GENERAL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    result = stage(parse_args())
    print(result)
