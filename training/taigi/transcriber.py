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

    def _input(
        self: BreezeTranscriber, path: Path, start: float, duration: float
    ) -> dict[str, Any]:
        samples, source_sample_rate = _read_interval(path, start, duration)
        if source_sample_rate != self.target_sample_rate:
            samples = resample_poly(samples, self.target_sample_rate, source_sample_rate).astype(
                np.float32, copy=False
            )
        return {"array": samples, "sampling_rate": self.target_sample_rate}

    def __call__(self: BreezeTranscriber, path: Path, start: float, duration: float) -> str:
        result: Any = self.pipe(
            self._input(path, start, duration),
            generate_kwargs={"language": "zh", "task": "transcribe"},
        )
        return str(result["text"]).strip()

    def transcribe_batch(
        self: BreezeTranscriber, intervals: list[tuple[Path, float, float]]
    ) -> list[str]:
        if not 1 <= len(intervals) <= 4:
            raise ValueError("Breeze batch must contain 1 to 4 intervals")
        # The HF pipeline removes dictionary keys while preparing each example.
        # Build independent 16 kHz inputs for every invocation and every row.
        inputs = [self._input(path, start, duration) for path, start, duration in intervals]
        results = self.pipe(
            inputs, batch_size=len(inputs), generate_kwargs={"language": "zh", "task": "transcribe"}
        )
        if not isinstance(results, list) or len(results) != len(intervals):
            raise ValueError("Breeze batch output count does not match input")
        return [str(result["text"]).strip() for result in results]


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
