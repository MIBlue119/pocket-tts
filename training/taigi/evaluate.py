"""Generate bounded fixed-seed Chinese prompts and record unverified ASR diagnostics."""

from __future__ import annotations

import argparse
import gc
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from pocket_tts.modules.stateful_module import init_states
from training.args import load_args
from training.checkpointing import load_checkpoint
from training.modules.builders import build_models
from training.taigi.audio import sha256_file
from training.taigi.schema import read_jsonl
from training.taigi.transcriber import BREEZE_ADAPTER_VERSION, BREEZE_MODEL, BREEZE_REVISION


def read_prompts(path: Path) -> list[dict[str, Any]]:
    prompts = read_jsonl(path)
    if not 1 <= len(prompts) <= 20:
        raise ValueError("provide 1 to 20 fixed prompts")
    seen = set()
    for prompt in prompts:
        identity, text = prompt.get("id"), prompt.get("text")
        if not isinstance(identity, str) or not identity.strip() or identity in seen:
            raise ValueError("prompt IDs must be nonempty and unique")
        if not isinstance(text, str) or not text.strip() or len(text) > 500:
            raise ValueError("prompt text must contain 1 to 500 characters")
        seen.add(identity)
    return prompts


def audio_metrics(samples: np.ndarray, sample_rate: int) -> dict[str, Any]:
    result: dict[str, Any] = {"samples": int(samples.size), "seconds": samples.size / sample_rate}
    if not samples.size:
        return {**result, "status": "empty"}
    if not np.isfinite(samples).all():
        return {**result, "status": "nonfinite"}
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    return {
        **result,
        "status": "silent" if rms < 1e-4 else "audio",
        "rms": rms,
        "peak": float(np.abs(samples).max()),
        "clipping_fraction": float(np.mean(np.abs(samples) >= 1.0)),
    }


def evaluate(args: argparse.Namespace) -> Path:
    prompts = read_prompts(args.prompts)
    if not 8 <= args.max_frames <= 375:
        raise ValueError("max-frames must be between 8 and 375")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = load_args(str(args.config))
    # Match the trainer's seeded build; checkpoint evaluation then replaces all raw model state.
    torch.manual_seed(config.seed)
    model, mimi, _ = build_models(config)
    step = load_checkpoint(args.checkpoint, model) if args.checkpoint else 0
    device = torch.device(args.device)
    model.to(device).eval()
    mimi.to(device).eval()
    report: dict[str, Any] = {
        "state": "running",
        "config_sha256": sha256_file(args.config),
        "tokenizer_sha256": sha256_file(
            Path(config.model_overrides["flow_lm.lookup_table.tokenizer_path"])
        ),
        "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint else None,
        "weights": "raw_checkpoint"
        if args.checkpoint
        else "seeded_pretrained_with_reset_text_embeddings",
        "step": step,
        "seed": args.seed,
        "max_frames": args.max_frames,
        "temperature": 0.3,
        "cfg_coef": 1.0,
        "voice_prompt_frames": 0,
        "quality_verified": False,
        "prompts": [],
    }
    report_path = output / "report.json"

    def save() -> None:
        temporary = output / "report.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(report_path)

    save()
    try:
        with torch.no_grad():
            for index, prompt in enumerate(prompts):
                torch.manual_seed(args.seed)
                tokens = torch.tensor(model.flow_lm.conditioner.tokenizer.encode(prompt["text"]))
                latents = model.generate(
                    [tokens],
                    [torch.empty(0, model.ldim, device=device)],
                    max_frames=args.max_frames,
                    temp=0.3,
                    cfg_coef=1.0,
                )[0]
                row = {
                    **prompt,
                    "generated_frames": len(latents),
                    "hit_frame_limit": len(latents) == args.max_frames,
                }
                if len(latents) < 8:
                    row.update(
                        status="too_few_frames_for_decoder", seconds=len(latents) / mimi.frame_rate
                    )
                else:
                    ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
                    state = init_states(mimi, 1, (len(latents) + 8) * ratio)
                    audio = (
                        mimi.decode_from_latent(latents[None], state)[0, 0].float().cpu().numpy()
                    )
                    row.update(audio_metrics(audio, mimi.sample_rate))
                    if row["status"] != "nonfinite":
                        path = output / f"prompt-{index:02d}.wav"
                        sf.write(path, audio, mimi.sample_rate, subtype="FLOAT")
                        row.update(audio_path=str(path), audio_sha256=sha256_file(path))
                    del state, audio
                report["prompts"].append(row)
                save()
                print(f"Prompt {index + 1}/{len(prompts)}: {row['status']}", flush=True)
                del latents, tokens
        del model, mimi
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if args.asr:
            from training.taigi.transcriber import breeze_transcriber

            transcribe = breeze_transcriber(args.device)
            report["asr"] = {
                "model": BREEZE_MODEL,
                "revision": BREEZE_REVISION,
                "adapter": BREEZE_ADAPTER_VERSION,
                "meaning": "same-model consistency only; not independent Taiwanese quality",
            }
            for row in report["prompts"]:
                if row.get("audio_path") and row["status"] == "audio":
                    text = transcribe(Path(row["audio_path"]), 0, row["seconds"])
                    row["asr_text"] = text
                    reference = "".join(c for c in row["text"] if c.isalnum())
                    recovered = "".join(c for c in text if c.isalnum())
                    row["character_sequence_similarity"] = SequenceMatcher(
                        None, reference, recovered
                    ).ratio()
                    save()
        report["state"] = "complete"
    except BaseException as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("prompts", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=125)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--asr", action="store_true")
    print(evaluate(parser.parse_args()))


if __name__ == "__main__":
    main()
