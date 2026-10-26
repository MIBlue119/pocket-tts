#!/usr/bin/env python3
"""Paired oracle-history flow diagnostic for the eight fixed seen ADI rows."""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path
from typing import Any

import torch

from training.args import load_args
from training.checkpointing import load_checkpoint
from training.dataloader.encode import encode_batch
from training.dataloader.loader import DataLoader
from training.modules.builders import build_models
from training.scripts.fork_taigi_adaptation import verify_merged_release
from training.scripts.run_experiment import digest
from training.taigi.schema import read_jsonl

ROOT = Path("/mnt/HDD2/proj_tts/pocket-tts")
RELEASE = ROOT / "data/taigi-adi-20261026/semantic-releases/59fa3a25a6cbc4c0d891"
SELECTION = ROOT / "data/taigi-adi-20261026/diagnostics/h100-source-asr-v1/selection.jsonl"
TIMES = (0.1, 0.5, 0.9)
DECODE_STEPS = (1, 4)
SCOPES = {"first1": 1, "first4": 4, "all": None}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    )


def scoped_mse(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if prediction.shape != target.shape or prediction.ndim != 2 or not len(prediction):
        raise ValueError("paired diagnostic tensors must be nonempty [frames, channels]")
    frame_mse = (prediction.float() - target.float()).square().mean(dim=-1)
    result = {
        name: float(
            frame_mse[:count].mean().item() if count is not None else frame_mse.mean().item()
        )
        for name, count in SCOPES.items()
    }
    if any(not math.isfinite(value) for value in result.values()):
        raise FloatingPointError("oracle-history diagnostic produced a nonfinite MSE")
    return result


def fixed_seen(selection: Path) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(selection) if row.get("category") == "seen"]
    if len(rows) != 8 or len({row.get("utterance_id") for row in rows}) != 8:
        raise ValueError("selection must contain exactly eight unique seen utterance IDs")
    for row in rows:
        if row.get("human_verified") is not False or row.get("audio_verified") is not False:
            raise ValueError("seen rows must remain explicitly unverified")
    return rows


def exact_probe_rows(
    config_train: Path, release: Path, selection: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    verify_merged_release(release)
    metadata = read_jsonl(release / "metadata.jsonl")
    by_prepared_id: dict[str, dict[str, Any]] = {}
    for row in metadata:
        prepared_id = row.get("source_metadata", {}).get("prepared_utterance_id")
        if isinstance(prepared_id, str):
            if prepared_id in by_prepared_id:
                raise ValueError(f"release repeats prepared utterance identity: {prepared_id}")
            by_prepared_id[prepared_id] = row
    train_rows = read_jsonl(config_train)
    train_signatures = {
        (row.get("path"), row.get("start"), row.get("duration"), row.get("transcript"))
        for row in train_rows
    }
    probe, identities = [], []
    for selected in selection:
        prepared_id = selected["utterance_id"]
        row = by_prepared_id.get(prepared_id)
        if row is None or row.get("split") != "train":
            raise ValueError(
                f"seen selection is absent from certified released training: {prepared_id}"
            )
        if row["input_text_zh"] != selected["mandarin_target"]:
            raise ValueError(f"seen selection target changed in certified release: {prepared_id}")
        signature = (
            row["published_audio_path"],
            row["start"],
            row["duration"],
            row["input_text_zh"],
        )
        if signature not in train_signatures:
            raise ValueError(f"seen row is not in the actual configured manifest: {prepared_id}")
        probe.append(
            {
                "path": row["published_audio_path"],
                "start": row["start"],
                "duration": row["duration"],
                "transcript": row["input_text_zh"],
            }
        )
        identities.append(
            {
                "prepared_utterance_id": prepared_id,
                "released_utterance_id": row["utterance_id"],
                "published_audio_path": row["published_audio_path"],
                "audio_sha256": row["audio_sha256"],
                "start": row["start"],
                "duration": row["duration"],
                "mandarin_target": row["input_text_zh"],
                "native_reference": selected["native_reference"],
            }
        )
    return probe, identities


def choose_shuffled(tokens: list[torch.Tensor], index: int) -> tuple[int, torch.Tensor]:
    choices = [other for other in range(len(tokens)) if other != index]
    wrong = min(choices, key=lambda other: (abs(len(tokens[other]) - len(tokens[index])), other))
    return wrong, tokens[wrong]


@torch.no_grad()
def probe(args: argparse.Namespace) -> Path:
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    config_path, checkpoint = args.config.resolve(), args.checkpoint.resolve()
    release, selection_path = args.release.resolve(), args.selection.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = load_args(config_path)
    if config.data.sampling_mode != "sentence" or config.data.precompute:
        raise ValueError("probe requires sentence sampling with precompute false")
    selected = fixed_seen(selection_path)
    probe_rows, identities = exact_probe_rows(Path(config.data.train_jsonl), release, selected)
    manifest = output / "exact-seen-input.jsonl"
    write_jsonl(manifest, probe_rows)

    torch.manual_seed(config.seed)
    model, mimi, _ = build_models(config)
    step = load_checkpoint(checkpoint, model)
    device = torch.device(args.device)
    model.to(device).eval()
    mimi.to(device).eval()
    fl = model.flow_lm
    tokenizer = fl.conditioner.tokenizer
    tokenizer_path = Path(config.model_overrides["flow_lm.lookup_table.tokenizer_path"]).resolve()
    expected_tokens = [torch.tensor(tokenizer.encode(row["transcript"])) for row in probe_rows]
    loader = DataLoader(
        str(manifest),
        tokenizer.encode,
        batch_size=1,
        sample_rate=mimi.sample_rate,
        frame_rate=mimi.frame_rate,
        max_duration_sec=config.data.max_duration_sec,
        max_voice_prompt_sec=0.0,
        rank=0,
        world_size=1,
        seed=args.seed,
        shuffle=False,
        io_workers=1,
        num_bucket_batches=1,
        prompt_trim_max_sec=0.0,
        final_punct_dropout=0.0,
        sampling_mode="sentence",
    )
    batches = loader._batches()  # intentionally synchronous: no background prefetch thread
    row_reports = []
    try:
        for index in range(8):
            batch = next(batches)
            actual_tokens = batch.text_tokens[0]
            if not torch.equal(actual_tokens.cpu(), expected_tokens[index]):
                raise ValueError(f"loader token identity mismatch for row {index}")
            decoded = tokenizer.decode(actual_tokens.tolist())
            if decoded != probe_rows[index]["transcript"]:
                raise ValueError(f"loader decoded text mismatch for row {index}: {decoded!r}")
            latents, mask, voice, voice_counts = encode_batch(mimi, batch, device)
            if voice.shape[1] != 0 or voice_counts.count_nonzero().item() != 0:
                raise ValueError("probe unexpectedly received voice prompt frames")
            y = (latents - fl.emb_mean) / fl.emb_std
            valid = mask[0]
            y_valid = y[0, valid]
            correct_tokens = [actual_tokens.to(device)]
            wrong_index, wrong_tokens = choose_shuffled(expected_tokens, index)
            arms = {
                "correct": model.backbone_z(
                    fl, y, correct_tokens, voice, voice_counts, cfg_dropout=False
                )[0, valid],
                "shuffled": model.backbone_z(
                    fl, y, [wrong_tokens.to(device)], voice, voice_counts, cfg_dropout=False
                )[0, valid],
                "null": model.backbone_z(
                    fl, y, correct_tokens, voice, voice_counts, cfg_dropout=False, force_null=True
                )[0, valid],
            }
            generator = torch.Generator(device=device).manual_seed(args.seed + index)
            noise = torch.randn(
                y_valid.shape, generator=generator, device=device, dtype=y_valid.dtype
            )
            metrics: dict[str, Any] = {arm: {"flow_diag": {}, "endpoint": {}} for arm in arms}
            for arm, z in arms.items():
                for time in TIMES:
                    t = y_valid.new_full((len(y_valid), 1), time)
                    xt = time * y_valid + (1 - time) * noise
                    velocity = fl.flow_net(z, t, t, xt)
                    metrics[arm]["flow_diag"][str(time)] = scoped_mse(velocity, y_valid - noise)
                endpoint_noise = math.sqrt(0.3) * noise
                for steps in DECODE_STEPS:
                    endpoint = model.flow.decode(partial(fl.flow_net, z), endpoint_noise, steps)
                    metrics[arm]["endpoint"][str(steps)] = scoped_mse(endpoint, y_valid)
            row_reports.append(
                {
                    **identities[index],
                    "frames": int(valid.sum().item()),
                    "decoded_loader_text": decoded,
                    "wrong_row_index": wrong_index,
                    "wrong_mandarin_target": probe_rows[wrong_index]["transcript"],
                    "noise_seed": args.seed + index,
                    "metrics": metrics,
                }
            )
    finally:
        close_batches = getattr(batches, "close", None)
        if close_batches is not None:
            close_batches()

    result = {
        "schema_version": "taigi-oracle-history-probe1/v1",
        "state": "complete",
        "quality_verified": False,
        "meaning": "paired normalized-latent diagnostics under identical oracle history/noise",
        "limitations": [
            "No metric is a perceptual-quality or free-generation success claim.",
            "Oracle acoustic history can supply content even when text is ignored.",
            "Noise scale 0.3 endpoint MSE is descriptive, not a calibrated likelihood.",
        ],
        "checkpoint_step": step,
        "seed": args.seed,
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "times": list(TIMES),
        "decode_steps": list(DECODE_STEPS),
        "probe_max_voice_prompt_sec": 0.0,
        "effective_config": {
            "sampling_mode": config.data.sampling_mode,
            "precompute": config.data.precompute,
            "config_max_voice_prompt_sec": config.data.max_voice_prompt_sec,
            "final_punct_dropout": config.data.final_punct_dropout,
        },
        "inputs": {
            "probe_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": digest(Path(__file__).resolve()),
            },
            "config": {"path": str(config_path), "sha256": digest(config_path)},
            "checkpoint": {"path": str(checkpoint), "sha256": digest(checkpoint)},
            "tokenizer": {"path": str(tokenizer_path), "sha256": digest(tokenizer_path)},
            "release_report": {
                "path": str(release / "report.json"),
                "sha256": digest(release / "report.json"),
            },
            "release_metadata": {
                "path": str(release / "metadata.jsonl"),
                "sha256": digest(release / "metadata.jsonl"),
            },
            "configured_train": {
                "path": config.data.train_jsonl,
                "sha256": digest(config.data.train_jsonl),
            },
            "selection": {"path": str(selection_path), "sha256": digest(selection_path)},
            "exact_probe_manifest": {"path": str(manifest), "sha256": digest(manifest)},
        },
        "rows": row_reports,
    }
    path = output / "results.json"
    write_json(path, result)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--release", type=Path, default=RELEASE)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    print(probe(parse_args()))
