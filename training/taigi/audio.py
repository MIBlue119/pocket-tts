from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int


def load_audio(path: Path) -> AudioData:
    try:
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError("audio preparation requires `soundfile`") from error
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return AudioData(samples=samples.mean(axis=1), sample_rate=sample_rate)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def speech_intervals(
    audio: AudioData, silence_db: float, min_silence: float, min_duration: float
) -> list[tuple[float, float]]:
    frame_samples = max(1, round(audio.sample_rate * 0.02))
    usable = len(audio.samples) // frame_samples * frame_samples
    if usable == 0:
        return []
    frames = audio.samples[:usable].reshape(-1, frame_samples)
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    speech = 20 * np.log10(rms) >= silence_db
    min_silence_frames = max(1, round(min_silence / 0.02))
    boundaries = [0]
    silent_start: int | None = None
    for index, is_speech in enumerate(speech):
        if not is_speech and silent_start is None:
            silent_start = index
        elif is_speech and silent_start is not None:
            if index - silent_start >= min_silence_frames:
                boundaries.append((silent_start + index) // 2)
            silent_start = None
    boundaries.append(len(frames))

    intervals: list[tuple[float, float]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        active = np.flatnonzero(speech[left:right])
        if not len(active):
            continue
        start_frame = left + int(active[0])
        end_frame = left + int(active[-1]) + 1
        duration = (end_frame - start_frame) * 0.02
        if duration >= min_duration:
            intervals.append((start_frame * 0.02, duration))
    return intervals


def quality_reason(audio: AudioData, start: float, duration: float) -> str | None:
    first = round(start * audio.sample_rate)
    last = round((start + duration) * audio.sample_rate)
    samples = audio.samples[first:last]
    if not len(samples) or float(np.max(np.abs(samples))) < 1e-4:
        return "empty_or_silent"
    if float(np.mean(np.abs(samples) >= 0.999)) > 0.005:
        return "clipping"
    if len(samples) >= audio.sample_rate * 2:
        half_second = audio.sample_rate // 2
        blocks = samples[: len(samples) // half_second * half_second].reshape(-1, half_second)
        if len(blocks) >= 4:
            rounded = [hashlib.sha256(block.tobytes()).digest() for block in blocks]
            if len(set(rounded)) <= len(rounded) // 2:
                return "repeated_audio_loop"
    return None
