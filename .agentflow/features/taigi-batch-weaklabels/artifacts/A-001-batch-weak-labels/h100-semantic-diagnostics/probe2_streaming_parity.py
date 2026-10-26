#!/usr/bin/env python3
"""Compare full and streaming backbones under identical oracle latent history."""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path
from typing import Any

import torch

from pocket_tts.modules.stateful_module import increment_steps, init_states
from probe1_oracle_history import exact_probe_rows, fixed_seen  # ty: ignore[unresolved-import]
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
NOISE_VARIANCE = 0.3
DECODE_STEPS = 1


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    )


def finite_float(value: torch.Tensor | float, label: str) -> float:
    result = float(value.item() if torch.is_tensor(value) else value)
    if not math.isfinite(result):
        raise FloatingPointError(f"nonfinite parity metric: {label}")
    return result


def frame_errors(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Raw per-frame differences; reference is the full-sequence computation."""
    if candidate.shape != reference.shape or candidate.ndim < 1 or not len(candidate):
        raise ValueError("parity tensors must have the same nonempty frame-first shape")
    delta = (candidate.float() - reference.float()).flatten(start_dim=1)
    baseline = reference.float().flatten(start_dim=1)
    max_abs = delta.abs().amax(dim=1)
    reference_l2 = torch.linalg.vector_norm(baseline, dim=1)
    relative_l2 = torch.linalg.vector_norm(delta, dim=1) / reference_l2.clamp_min(
        torch.finfo(torch.float32).eps
    )
    frames = [
        {
            "frame": index,
            "max_abs": finite_float(max_abs[index], f"frame {index} max_abs"),
            "relative_l2": finite_float(relative_l2[index], f"frame {index} relative_l2"),
            "reference_l2": finite_float(reference_l2[index], f"frame {index} reference_l2"),
        }
        for index in range(len(candidate))
    ]
    return {
        "max_abs": finite_float(max_abs.max(), "summary max_abs"),
        "max_relative_l2": finite_float(relative_l2.max(), "summary max_relative_l2"),
        "mean_relative_l2": finite_float(relative_l2.mean(), "summary mean_relative_l2"),
        "per_frame": frames,
    }


def streaming_backbone(
    model: TrainableTTS, normalized_latents: torch.Tensor, tokens: torch.Tensor
) -> torch.Tensor:
    """The CFG=1 generation backbone with oracle history substituted after each frame."""
    if normalized_latents.shape[0] != 1 or normalized_latents.ndim != 3:
        raise ValueError("streaming parity requires one [1, frames, channels] row")
    fl = model.flow_lm
    text = fl.conditioner(tokens[None].to(normalized_latents.device))[0]
    prefix = torch.cat([fl.bos_before_voice[0], text], dim=0)[None]
    state = init_states(fl, 1, prefix.shape[1] + normalized_latents.shape[1] + 2)
    outputs = []
    for frame in range(normalized_latents.shape[1]):
        latent_input = (
            fl.bos_emb.view(1, 1, -1) if frame == 0 else normalized_latents[:, frame - 1 : frame]
        )
        transformed = fl.input_linear(latent_input)
        if frame == 0:
            transformed = torch.cat([prefix, transformed], dim=1)
        out = fl.out_norm(fl.transformer(transformed, state))
        if fl.z_proj is not None:
            out = fl.z_proj(out)
        increment_steps(fl, state, increment=transformed.shape[1])
        outputs.append(out[:, -1].float())
    return torch.stack(outputs, dim=1)


def certified_rows(
    config_train: Path, release: Path, selection_path: Path, probe1_manifest: Path, count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = fixed_seen(selection_path)
    expected_rows, identities = exact_probe_rows(config_train, release, selected)
    actual_probe1_rows = read_jsonl(probe1_manifest)
    if actual_probe1_rows != expected_rows:
        raise ValueError(
            "actual probe1 exact manifest no longer matches certified release/config join"
        )
    return actual_probe1_rows[:count], identities[:count]


@torch.no_grad()
def probe(args: argparse.Namespace) -> Path:
    if not 1 <= args.rows <= 8:
        raise ValueError("rows must be between 1 and 8")
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
    probe_rows, identities = certified_rows(
        Path(config.data.train_jsonl), release, selection, probe1_manifest, args.rows
    )
    exact_manifest = output / "exact-parity-input.jsonl"
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
    expected_tokens = [torch.tensor(tokenizer.encode(row["transcript"])) for row in probe_rows]
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
    batches = loader._batches()  # synchronous by design; avoids prefetch shutdown hazards
    row_reports = []
    try:
        for index in range(args.rows):
            batch = next(batches)
            tokens = batch.text_tokens[0]
            if not torch.equal(tokens.cpu(), expected_tokens[index]):
                raise ValueError(f"loader token identity mismatch for row {index}")
            decoded_text = tokenizer.decode(tokens.tolist())
            if decoded_text != probe_rows[index]["transcript"]:
                raise ValueError(f"loader decoded text mismatch for row {index}: {decoded_text!r}")
            latents, mask, voice, voice_counts = encode_batch(mimi, batch, device)
            if voice.shape[1] != 0 or voice_counts.count_nonzero().item() != 0:
                raise ValueError("probe unexpectedly received voice prompt frames")
            frame_count = int(mask[0].sum().item())
            expected_mask = torch.arange(mask.shape[1], device=mask.device) < frame_count
            if not torch.equal(mask[0], expected_mask):
                raise ValueError("probe requires a contiguous valid-frame mask")
            normalized = ((latents - fl.emb_mean) / fl.emb_std)[:, :frame_count].float()

            z_full = model.backbone_z(
                fl, normalized, [tokens.to(device)], voice, voice_counts, cfg_dropout=False
            ).float()
            z_stream = streaming_backbone(model, normalized, tokens)
            eos_full = fl.out_eos(z_full).squeeze(0)
            eos_stream = fl.out_eos(z_stream).squeeze(0)
            generator = torch.Generator(device=device).manual_seed(args.seed + index)
            noise = math.sqrt(NOISE_VARIANCE) * torch.randn(
                (frame_count, model.ldim), generator=generator, device=device, dtype=torch.float32
            )
            decoded_full = model.flow.decode(
                partial(fl.flow_net, z_full.squeeze(0)), noise, DECODE_STEPS
            )
            decoded_stream = model.flow.decode(
                partial(fl.flow_net, z_stream.squeeze(0)), noise, DECODE_STEPS
            )
            decoded_full = decoded_full * fl.emb_std + fl.emb_mean
            decoded_stream = decoded_stream * fl.emb_std + fl.emb_mean
            row_reports.append(
                {
                    **identities[index],
                    "frames": frame_count,
                    "decoded_loader_text": decoded_text,
                    "noise_seed": args.seed + index,
                    "metrics": {
                        "z": frame_errors(z_stream.squeeze(0), z_full.squeeze(0)),
                        "eos_logits": frame_errors(eos_stream, eos_full),
                        "decoded_latents": frame_errors(decoded_stream, decoded_full),
                    },
                }
            )
    finally:
        close_batches = getattr(batches, "close", None)
        if close_batches is not None:
            close_batches()

    result = {
        "schema_version": "taigi-same-history-streaming-parity-probe2/v1",
        "state": "complete",
        "quality_verified": False,
        "meaning": "full-sequence versus streaming backbone parity under identical oracle history",
        "limitations": [
            "This is a wiring parity diagnostic, not a generated-speech quality claim.",
            "The comparison uses oracle target-latent history rather than autoregressive history.",
            "Relative errors are descriptive and have no pass threshold.",
        ],
        "checkpoint_step": step,
        "seed": args.seed,
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "cfg_coef": 1.0,
        "decode_steps": DECODE_STEPS,
        "noise_variance": NOISE_VARIANCE,
        "rows_requested": args.rows,
        "probe_max_voice_prompt_sec": 0.0,
        "inputs": {
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
            "exact_parity_manifest": {
                "path": str(exact_manifest),
                "sha256": digest(exact_manifest),
            },
        },
        "rows": row_reports,
    }
    results = output / "results.json"
    write_json(results, result)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--release", type=Path, default=RELEASE)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--probe1-manifest", type=Path, default=PROBE1_MANIFEST)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    print(probe(parse_args()))
