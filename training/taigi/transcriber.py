from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

BREEZE_MODEL = "MediaTek-Research/Breeze-ASR-26"
BREEZE_REVISION = "7b992682e7f5ceedd0a41ebec240f01ba469d19e"
BREEZE_ADAPTER_VERSION = "breeze-resample/v2"


def _read_interval(path: Path, start: float, duration: float) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError("Breeze transcription requires `soundfile`") from error
    with sf.SoundFile(path) as audio:
        sample_rate = audio.samplerate
        audio.seek(round(start * sample_rate))
        samples = audio.read(round(duration * sample_rate), dtype="float32", always_2d=True)
    return samples.mean(axis=1), sample_rate


@dataclass
class BreezeTranscriber:
    pipe: Any
    target_sample_rate: int
    adapter_version: str = BREEZE_ADAPTER_VERSION

    def __call__(self: BreezeTranscriber, path: Path, start: float, duration: float) -> str:
        samples, source_sample_rate = _read_interval(path, start, duration)
        if source_sample_rate != self.target_sample_rate:
            samples = resample_poly(samples, self.target_sample_rate, source_sample_rate).astype(
                np.float32, copy=False
            )
        result: Any = self.pipe(
            {"array": samples, "sampling_rate": self.target_sample_rate},
            generate_kwargs={"language": "zh", "task": "transcribe"},
        )
        return str(result["text"]).strip()


def breeze_transcriber(device: str) -> BreezeTranscriber:
    try:
        from transformers import pipeline
    except ImportError as error:
        raise RuntimeError("Breeze transcription requires `transformers`") from error
    pipe = pipeline(
        "automatic-speech-recognition", model=BREEZE_MODEL, revision=BREEZE_REVISION, device=device
    )
    feature_extractor = pipe.feature_extractor
    if feature_extractor is None:
        raise RuntimeError("Breeze pipeline did not load a feature extractor")
    target_sample_rate = int(feature_extractor.sampling_rate)
    return BreezeTranscriber(pipe=pipe, target_sample_rate=target_sample_rate)
