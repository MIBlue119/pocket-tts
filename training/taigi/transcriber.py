from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

BREEZE_MODEL = "MediaTek-Research/Breeze-ASR-26"
BREEZE_REVISION = "7b992682e7f5ceedd0a41ebec240f01ba469d19e"


def breeze_transcriber(device: str) -> Callable[[Path, float, float], str]:
    try:
        from transformers import pipeline
    except ImportError as error:
        raise RuntimeError("Breeze transcription requires `transformers`") from error
    pipe = pipeline(
        "automatic-speech-recognition", model=BREEZE_MODEL, revision=BREEZE_REVISION, device=device
    )

    def transcribe(path: Path, start: float, duration: float) -> str:
        from training.taigi.audio import load_audio

        audio = load_audio(path)
        first = round(start * audio.sample_rate)
        last = round((start + duration) * audio.sample_rate)
        result: Any = pipe(
            {"array": audio.samples[first:last], "sampling_rate": audio.sample_rate},
            generate_kwargs={"language": "zh", "task": "transcribe"},
        )
        return str(result["text"]).strip()

    return transcribe
