"""Manifest entries and padded training batches."""

import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class AudioPrompt:
    path: str
    start: float
    duration: float
    transcript: str | None = None


@dataclass
class Entry:
    path: str
    duration: float
    transcript: str
    words: list[dict[str, Any]] | None = None  # [{"word", "start", "end"}] from align_data
    start: float = 0.0  # offset of the utterance inside the audio file (long recordings)
    latents_file: str | None = None
    prompt: AudioPrompt | None = None


@dataclass
class Batch:
    audio: torch.Tensor  # [B, 1, samples], zero-padded
    num_audio_frames: torch.Tensor  # [B] valid codec frames per sample
    text_tokens: list[torch.Tensor]  # ragged, one [L_b] long tensor per sample
    voice_audio: torch.Tensor  # [B, 1, prompt_samples]
    num_voice_prompt_frames: torch.Tensor  # [B] valid codec frames of each voice prompt
    tail_latents: torch.Tensor | None = None
    prompt_latents: torch.Tensor | None = None
