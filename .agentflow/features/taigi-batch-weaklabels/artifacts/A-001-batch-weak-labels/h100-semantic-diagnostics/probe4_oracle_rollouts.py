#!/usr/bin/env python3
"""Bounded oracle-history and reset-block waveform diagnostic for four seen rows."""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf
import torch

from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules.stateful_module import increment_steps, init_states
from probe1_oracle_history import (  # ty: ignore[unresolved-import]
    choose_shuffled,
    exact_probe_rows,
    fixed_seen,
)
from training.args import load_args
from training.checkpointing import load_checkpoint
from training.dataloader.encode import encode_batch
from training.dataloader.loader import DataLoader
from training.modules.builders import build_models
from training.modules.model import TrainableTTS
from training.scripts.run_experiment import digest
from training.taigi.schema import read_jsonl

ROOT = Path("/mnt/HDD2/proj_tts/pocket-tts")
CONTROL = ROOT / "data/taigi-adi-20261026/semantic-control-20261026"
RELEASE = ROOT / "data/taigi-adi-20261026/semantic-releases/59fa3a25a6cbc4c0d891"
SELECTION = ROOT / "data/taigi-adi-20261026/diagnostics/h100-source-asr-v1/selection.jsonl"
PROBE1_MANIFEST = CONTROL / "probe1-step800/exact-seen-input.jsonl"
ROW_COUNT = 4
TEMPERATURE = 0.3
DECODE_STEPS = 1
VARIANTS = (
    "source_mimi",
    "oracle_correct",
    "oracle_wrong",
    "block4",
    "block8",
    "full_self_fed_fixed_t",
)


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    )


def require_finite(tensor: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"nonfinite tensor in {label}")


def finite_float(value: torch.Tensor | float, label: str) -> float:
    result = float(value.item() if torch.is_tensor(value) else value)
    if not math.isfinite(result):
        raise FloatingPointError(f"nonfinite metric in {label}")
    return result


def latent_mse(candidate: torch.Tensor, source: torch.Tensor) -> dict[str, float]:
    if candidate.shape != source.shape or candidate.ndim != 2 or not len(candidate):
        raise ValueError("latent comparison requires matching nonempty [frames, channels]")
    require_finite(candidate, "predicted latents")
    require_finite(source, "source latents")
    per_frame = (candidate.float() - source.float()).square().mean(dim=-1)
    early = min(4, len(per_frame))
    return {
        "early_first4": finite_float(per_frame[:early].mean(), "early latent MSE"),
        "all": finite_float(per_frame.mean(), "all-frame latent MSE"),
    }


def condition_prefix(
    model: TrainableTTS, tokens: torch.Tensor, device: torch.device
) -> torch.Tensor:
    fl = model.flow_lm
    text = fl.conditioner(tokens[None].to(device))[0]
    return torch.cat([fl.bos_before_voice[0], text], dim=0)[None]


def oracle_decode(
    model: TrainableTTS, normalized_source: torch.Tensor, tokens: torch.Tensor, noise: torch.Tensor
) -> torch.Tensor:
    """Decode all frames from the full teacher-forced backbone in one batch."""
    fl = model.flow_lm
    empty_voice = normalized_source.new_empty(1, 0, model.ldim)
    voice_counts = torch.zeros(1, dtype=torch.long, device=normalized_source.device)
    z = model.backbone_z(
        fl,
        normalized_source,
        [tokens.to(normalized_source.device)],
        empty_voice,
        voice_counts,
        cfg_dropout=False,
    ).float()[0]
    result = model.flow.decode(partial(fl.flow_net, z), noise, DECODE_STEPS)
    require_finite(result, "oracle decode")
    return result


def reset_block_rollout(
    model: TrainableTTS,
    normalized_source: torch.Tensor,
    tokens: torch.Tensor,
    noise: torch.Tensor,
    block_size: int | None,
) -> torch.Tensor:
    """Fixed-T CFG=1 rollout; resets rebuild KV from the entire oracle history.

    ``block_size=None`` is fully self-fed. ``block_size=1`` is the oracle-equivalence
    sanity path: every frame rebuilds through all preceding source frames.
    """
    if normalized_source.shape[0] != 1 or normalized_source.ndim != 3:
        raise ValueError("rollout requires one [1, frames, channels] source row")
    if block_size is not None and block_size < 1:
        raise ValueError("block size must be positive or None")
    fl = model.flow_lm
    device = normalized_source.device
    prefix = condition_prefix(model, tokens, device)
    frame_count = normalized_source.shape[1]
    predictions: list[torch.Tensor] = []
    state = None
    for frame in range(frame_count):
        at_boundary = frame == 0 or (block_size is not None and frame % block_size == 0)
        if at_boundary:
            # Rebuild from scratch using BOS and *every* ground-truth predecessor. This is
            # intentionally not a one-frame correction on top of contaminated self-fed KV.
            oracle_inputs = torch.cat(
                [fl.bos_emb.view(1, 1, -1), normalized_source[:, :frame]], dim=1
            )
            inputs = torch.cat([prefix, fl.input_linear(oracle_inputs)], dim=1)
            state = init_states(fl, 1, prefix.shape[1] + frame_count + 2)
        else:
            inputs = fl.input_linear(predictions[-1].view(1, 1, -1))
        assert state is not None
        out = fl.out_norm(fl.transformer(inputs, state))
        if fl.z_proj is not None:
            out = fl.z_proj(out)
        increment_steps(fl, state, increment=inputs.shape[1])
        z = out[:, -1].float()
        predicted = model.flow.decode(
            partial(fl.flow_net, z), noise[frame : frame + 1], DECODE_STEPS
        )[0]
        require_finite(predicted, f"rollout frame {frame}")
        predictions.append(predicted)
    return torch.stack(predictions)


def decode_wave(
    mimi: MimiModel, latents: torch.Tensor, device: torch.device
) -> npt.NDArray[np.float32]:
    require_finite(latents, "Mimi input latents")
    if len(latents) < 8:
        raise ValueError("Mimi waveform diagnostic requires at least eight latent frames")
    ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
    state = init_states(mimi, 1, (len(latents) + 8) * ratio)
    audio = mimi.decode_from_latent(latents[None].to(device), state)[0, 0].float().cpu().numpy()
    if not np.isfinite(audio).all():
        raise FloatingPointError("Mimi decoder produced nonfinite audio")
    return audio


def save_wave(path: Path, audio: npt.NDArray[np.float32], sample_rate: int) -> dict[str, Any]:
    sf.write(path, audio, sample_rate, subtype="FLOAT")
    rms = finite_float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))), "audio RMS")
    peak = finite_float(np.abs(audio).max(), "audio peak")
    return {
        "audio_path": str(path),
        "audio_sha256": digest(path),
        "samples": int(audio.size),
        "seconds": finite_float(audio.size / sample_rate, "audio duration"),
        "rms": rms,
        "peak": peak,
    }


def certified_rows(
    config_train: Path, release: Path, selection_path: Path, probe1_manifest: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = fixed_seen(selection_path)
    expected_rows, identities = exact_probe_rows(config_train, release, selected)
    actual_probe1_rows = read_jsonl(probe1_manifest)
    if actual_probe1_rows != expected_rows:
        raise ValueError(
            "actual probe1 exact manifest no longer matches certified release/config join"
        )
    return actual_probe1_rows, identities


@torch.no_grad()
def probe(args: argparse.Namespace) -> Path:
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    config_path = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    release = args.release.resolve()
    selection = args.selection.resolve()
    probe1_manifest = args.probe1_manifest.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    config = load_args(config_path)
    if config.data.sampling_mode != "sentence" or config.data.precompute:
        raise ValueError("probe requires sentence sampling with precompute false")
    all_rows, all_identities = certified_rows(
        Path(config.data.train_jsonl), release, selection, probe1_manifest
    )
    probe_rows, identities = all_rows[:ROW_COUNT], all_identities[:ROW_COUNT]
    exact_manifest = output / "exact-first4-input.jsonl"
    write_jsonl(exact_manifest, probe_rows)

    torch.manual_seed(config.seed)
    model, mimi, _ = build_models(config)
    step = load_checkpoint(checkpoint, model)
    device = torch.device(args.device)
    model.float().to(device).eval()
    mimi.to(device).eval()
    fl = model.flow_lm
    tokenizer = fl.conditioner.tokenizer
    tokenizer_path = Path(config.model_overrides["flow_lm.lookup_table.tokenizer_path"]).resolve()
    expected_tokens = [torch.tensor(tokenizer.encode(row["transcript"])) for row in all_rows]
    loader = DataLoader(
        str(exact_manifest),
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
    report: dict[str, Any] = {
        "schema_version": "taigi-oracle-rollout-probe4/v1",
        "state": "running",
        "quality_verified": False,
        "meaning": "fixed-length oracle-history and self-fed waveform diagnostics",
        "limitations": [
            "EOS is ignored and every model variant is forced to the source frame count.",
            "Full self-fed fixed-T audio is diagnostic, not a public generation-quality test.",
            "Breeze ASR is intentionally deferred; this run produces waveforms and latent MSE only.",
            "Mimi source reconstruction is a codec reference, not model generation.",
        ],
        "checkpoint_step": step,
        "seed": args.seed,
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "cfg_coef": 1.0,
        "decode_steps": DECODE_STEPS,
        "temperature": TEMPERATURE,
        "rows": [],
    }
    report_path = output / "results.json"
    write_json(report_path, report)
    batches = loader._batches()  # synchronous by design; no background prefetch shutdown
    try:
        for index in range(ROW_COUNT):
            batch = next(batches)
            tokens = batch.text_tokens[0]
            if not torch.equal(tokens.cpu(), expected_tokens[index]):
                raise ValueError(f"loader token identity mismatch for row {index}")
            decoded_text = tokenizer.decode(tokens.tolist())
            if decoded_text != probe_rows[index]["transcript"]:
                raise ValueError(f"loader decoded text mismatch for row {index}: {decoded_text!r}")
            source_latents, mask, voice, voice_counts = encode_batch(mimi, batch, device)
            if voice.shape[1] != 0 or voice_counts.count_nonzero().item() != 0:
                raise ValueError("probe unexpectedly received voice prompt frames")
            frame_count = int(mask[0].sum().item())
            expected_mask = torch.arange(mask.shape[1], device=mask.device) < frame_count
            if not torch.equal(mask[0], expected_mask):
                raise ValueError("probe requires a contiguous valid-frame mask")
            source_raw = source_latents[0, :frame_count].float()
            normalized = ((source_raw - fl.emb_mean) / fl.emb_std)[None]
            require_finite(normalized, "normalized source")
            wrong_index, wrong_tokens = choose_shuffled(expected_tokens, index)
            generator = torch.Generator(device=device).manual_seed(args.seed + index)
            noise = math.sqrt(TEMPERATURE) * torch.randn(
                (frame_count, model.ldim), generator=generator, device=device, dtype=torch.float32
            )
            require_finite(noise, "fixed rollout noise")

            normalized_variants = {
                "source_mimi": normalized[0],
                "oracle_correct": oracle_decode(model, normalized, tokens, noise),
                "oracle_wrong": oracle_decode(model, normalized, wrong_tokens, noise),
                "block4": reset_block_rollout(model, normalized, tokens, noise, 4),
                "block8": reset_block_rollout(model, normalized, tokens, noise, 8),
                "full_self_fed_fixed_t": reset_block_rollout(
                    model, normalized, tokens, noise, None
                ),
            }
            if tuple(normalized_variants) != VARIANTS:
                raise AssertionError("probe variant set changed")
            row_report = {
                **identities[index],
                "frames": frame_count,
                "decoded_loader_text": decoded_text,
                "wrong_row_index": wrong_index,
                "wrong_mandarin_target": all_rows[wrong_index]["transcript"],
                "noise_seed": args.seed + index,
                "variants": {},
            }
            for variant, normalized_output in normalized_variants.items():
                raw_output = normalized_output * fl.emb_std + fl.emb_mean
                audio = decode_wave(mimi, raw_output, device)
                path = output / f"row-{index:02d}-{variant}.wav"
                row_report["variants"][variant] = {
                    "normalized_latent_mse": latent_mse(normalized_output, normalized[0]),
                    **save_wave(path, audio, mimi.sample_rate),
                }
            report["rows"].append(row_report)
            write_json(report_path, report)
        if len(report["rows"]) != ROW_COUNT:
            raise AssertionError("probe did not produce exactly four rows")
        wave_count = sum(len(row["variants"]) for row in report["rows"])
        if wave_count != ROW_COUNT * len(VARIANTS):
            raise AssertionError("probe did not produce exactly 24 waveforms")
        report["waveform_count"] = wave_count
        report["inputs"] = {
            "probe_script": {"path": str(Path(__file__).resolve()), "sha256": digest(__file__)},
            "probe1_script": {
                "path": str((CONTROL / "probe1_oracle_history.py").resolve()),
                "sha256": digest(CONTROL / "probe1_oracle_history.py"),
            },
            "probe1_exact_manifest": {
                "path": str(probe1_manifest),
                "sha256": digest(probe1_manifest),
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
            "selection": {"path": str(selection), "sha256": digest(selection)},
            "exact_first4_manifest": {
                "path": str(exact_manifest),
                "sha256": digest(exact_manifest),
            },
        }
        report["state"] = "complete"
    except BaseException as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        close_batches = getattr(batches, "close", None)
        if close_batches is not None:
            close_batches()
        write_json(report_path, report)
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--release", type=Path, default=RELEASE)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--probe1-manifest", type=Path, default=PROBE1_MANIFEST)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    print(probe(parse_args()))
