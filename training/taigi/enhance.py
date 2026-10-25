"""Opt-in, pinned RE-USE candidate enhancement; never publishes a training release."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from training.taigi.audio import sha256_file

MODEL = "nvidia/NVIDIA-NemotronLabs-RE-USE"
REVISION = "31f82f2efad1095034492f236e8dac9239ba713b"
ADAPTER_VERSION = "reuse-cpu-overlap/v1"


@dataclass(frozen=True)
class ReusePredictor:
    forward: Callable[[np.ndarray, int], np.ndarray]
    artifacts: dict[str, str]

    def __call__(self: ReusePredictor, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        return self.forward(samples, sample_rate)


def overlap_enhance(
    samples: np.ndarray,
    sample_rate: int,
    predict: Callable[[np.ndarray, int], np.ndarray],
    chunk_seconds: float = 5.0,
) -> np.ndarray:
    """Keep assembly on CPU and transfer at most one bounded window to the model."""
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("expected nonempty, finite mono audio")
    if not 0.5 <= chunk_seconds <= 15:
        raise ValueError("chunk_seconds must be between 0.5 and 15")
    if sample_rate not in {8000, 16000, 22050, 24000, 32000, 44100, 48000}:
        raise ValueError("unsupported RE-USE sample rate")
    size = round(sample_rate * chunk_seconds)
    hop = max(1, size // 2)
    starts = list(range(0, max(1, len(samples) - size + 1), hop))
    final_start = max(0, len(samples) - size)
    if starts[-1] != final_start:
        starts.append(final_start)
    output = np.zeros(len(samples), dtype=np.float32)
    weights = np.zeros(len(samples), dtype=np.float32)
    window = np.maximum(np.hanning(size).astype(np.float32), 1e-7)
    for start in starts:
        clip = samples[start : start + size]
        # STFT reflect padding needs more samples than half its analysis window.
        padded = np.pad(clip, (0, max(0, round(sample_rate * 0.05) - len(clip))))
        enhanced = np.asarray(predict(padded, sample_rate), dtype=np.float32)
        if enhanced.shape != padded.shape or not np.isfinite(enhanced).all():
            raise ValueError("enhancer returned invalid samples or changed sample count")
        output[start : start + len(clip)] += enhanced[: len(clip)] * window[: len(clip)]
        weights[start : start + len(clip)] += window[: len(clip)]
    return output / weights


def load_predictor(cache_dir: Path | None, download: bool, device: str) -> ReusePredictor:
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("RE-USE requires an available CUDA device")
    snapshot = Path(
        snapshot_download(
            MODEL,
            revision=REVISION,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=not download,
            allow_patterns=["config.json", "model.safetensors", "models/*.py", "LICENSE"],
        )
    )
    # Import only model modules in a private namespace. Upstream inference modules
    # perform unpinned network requests at import time and must never be imported.
    namespace = "_taigi_reuse_" + REVISION
    package = types.ModuleType(namespace)
    package.__path__ = [str(snapshot / "models")]
    sys.modules[namespace] = package
    generator = importlib.import_module(namespace + ".generator_SEMamba_time_d4")
    stfts = importlib.import_module(namespace + ".stfts")
    cfg = json.loads((snapshot / "config.json").read_text())
    model = generator.SEMamba(cfg)
    model.load_state_dict(load_file(str(snapshot / "model.safetensors")), strict=True)
    model.to(device).eval()

    def predict(samples: np.ndarray, rate: int) -> np.ndarray:
        stft = cfg["stft_cfg"]

        def scaled(key: str) -> int:
            value = int(round(stft[key] * rate // stft["sampling_rate"]))
            return value if value % 2 == 0 else value + 1

        fft, hop, win = (scaled(key) for key in ("n_fft", "hop_size", "win_size"))
        compression = cfg["model_cfg"]["compress_factor"]
        with torch.inference_mode():
            audio = torch.from_numpy(samples.copy()).unsqueeze(0).to(device)
            magnitude, phase, _ = stfts.mag_phase_stft(
                audio,
                n_fft=fft,
                hop_size=hop,
                win_size=win,
                compress_factor=compression,
                center=True,
                addeps=False,
            )
            amplitude, phase, _ = model(magnitude, phase)
            mag = torch.expm1(torch.relu(amplitude))
            amplitude[:, :, ((mag == 0).sum(1) / mag.shape[1] > 0.5)[0]] = 0
            result = stfts.mag_phase_istft(amplitude, phase, fft, hop, win, compression)
            result = result[0].cpu().numpy()
        if len(result) < len(samples):
            result = np.pad(result, (0, len(samples) - len(result)), constant_values=1e-8)
        return result[: len(samples)]

    artifacts = {
        name: sha256_file(snapshot / name) for name in ("config.json", "model.safetensors")
    }
    return ReusePredictor(predict, artifacts)


def enhance_file(
    source: Path,
    output: Path,
    predict: Callable[[np.ndarray, int], np.ndarray],
    *,
    chunk_seconds: float = 5.0,
    start: float = 0.0,
    duration: float = 30.0,
) -> Path:
    import soundfile as sf

    if start < 0 or not 0 < duration <= 600:
        raise ValueError("start must be nonnegative and duration in (0, 600] seconds")
    metadata_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite existing audio or provenance")
    if output.suffix.lower() != ".wav":
        raise ValueError("output must be .wav (float32; no silent PCM clipping)")
    raw_hash = sha256_file(source)
    with sf.SoundFile(source) as handle:
        rate = handle.samplerate
        first = round(start * rate)
        if first >= len(handle):
            raise ValueError("start is beyond source audio")
        handle.seek(first)
        samples = handle.read(round(duration * rate), dtype="float32", always_2d=True).mean(1)
    enhanced = overlap_enhance(samples, rate, predict, chunk_seconds)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, enhanced, rate, subtype="FLOAT")
    metadata = {
        "schema_version": "taigi-enhanced-candidate/v1",
        "status": "candidate",
        "verification": "unverified_enhancement",
        "audio_variant": "enhanced",
        "source_audio_path": str(source.resolve()),
        "source_audio_sha256": raw_hash,
        "source_start_sample": first,
        "num_samples": len(samples),
        "sample_rate": rate,
        "source_clip_pcm_f32le_sha256": hashlib.sha256(samples.astype("<f4").tobytes()).hexdigest(),
        "model_artifact_sha256": predict.artifacts if isinstance(predict, ReusePredictor) else {},
        "audio_path": str(output.resolve()),
        "audio_sha256": sha256_file(output),
        "enhancement_model": MODEL,
        "enhancement_revision": REVISION,
        "adapter_version": ADAPTER_VERSION,
        "adapter_sha256": sha256_file(Path(__file__)),
        "chunk_seconds": chunk_seconds,
        "hop_fraction": 0.5,
        "bandwidth_extension": False,
        "peak_abs": float(np.max(np.abs(enhanced))),
        "clipping_fraction": float(np.mean(np.abs(enhanced) >= 0.999)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--chunk-seconds", type=float, default=5)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    predict = load_predictor(args.cache_dir, args.download_model, args.device)
    print(
        enhance_file(
            args.source,
            args.output,
            predict,
            chunk_seconds=args.chunk_seconds,
            start=args.start,
            duration=args.duration,
        )
    )


if __name__ == "__main__":
    main()
