"""Whole-sentence supervision without target leakage or word alignment."""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import sphn
import torch

from pocket_tts.models.mimi import MimiModel
from training.args import DataArgs
from training.dataloader import DataLoader, encode_batch
from training.dataloader.loader import InvalidSentenceEntry

SR = 24000
TEXT = "今天下午我們一起去市場買水果。"


def make_loader(tmp_path: Path, rows: list[dict[str, Any]], **kwargs: Any) -> DataLoader:  # noqa: ANN401
    path = tmp_path / "sentences.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return DataLoader(
        str(path),
        lambda text: list(map(ord, text)),
        len(rows),
        SR,
        12.5,
        kwargs.pop("max_duration_sec", 30),
        5,
        0,
        1,
        shuffle=False,
        sampling_mode=kwargs.pop("sampling_mode", "sentence"),
        **kwargs,
    )


def write_audio(tmp_path: Path, name: str = "target.wav", frequency: int = 220) -> str:
    t = np.arange(3 * SR) / SR
    wav = (0.2 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    path = tmp_path / name
    sphn.write_wav(str(path), wav, SR)
    return str(path)


def row(path: str, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
    return {"path": path, "start": 0, "duration": 2, "transcript": TEXT, **kwargs}


class Codec:
    """Mimi test double restricted to the encoder boundary used by encode_batch."""

    def __init__(self):
        self.calls = []

    def encode_to_latent(self, audio: torch.Tensor) -> torch.Tensor:
        assert audio.shape[-1] > 0
        self.calls.append(audio.shape)
        return torch.ones(audio.shape[0], audio.shape[-1] // 1920, 4)


def test_full_chinese_sentence_and_no_prompt(tmp_path: Path):
    loader = make_loader(tmp_path, [row(write_audio(tmp_path))])
    batch = next(iter(loader))
    assert batch.text_tokens[0].tolist() == list(map(ord, TEXT))
    assert batch.audio.shape == (1, 1, 2 * SR)
    assert batch.num_audio_frames.tolist() == [25]
    assert batch.voice_audio.shape == (1, 1, 0)
    assert batch.num_voice_prompt_frames.tolist() == [0]
    codec = Codec()
    latents, mask, prompt, lengths = encode_batch(
        cast(MimiModel, codec), batch, torch.device("cpu")
    )
    assert len(codec.calls) == 1
    assert latents.shape == (1, 25, 4)
    assert mask.all()
    assert prompt.shape == (1, 0, 4)
    assert lengths.tolist() == [0]


def test_mixed_prompts_encode_only_present_rows(tmp_path: Path):
    target = write_audio(tmp_path)
    prompt = {"path": write_audio(tmp_path, "prompt.wav", 440), "start": 0, "duration": 1}
    batch = next(iter(make_loader(tmp_path, [row(target), row(target, prompt=prompt)])))
    codec = Codec()
    _, _, voice, lengths = encode_batch(cast(MimiModel, codec), batch, torch.device("cpu"))
    assert codec.calls == [(2, 1, 2 * SR), (1, 1, SR)]
    assert lengths.tolist() == [0, 12]
    assert torch.count_nonzero(voice[0]) == 0
    assert torch.all(voice[1] == 1)


@pytest.mark.parametrize("prompt_start", [0, 1.5])
def test_overlapping_prompt_is_rejected(tmp_path: Path, prompt_start: float):
    target = write_audio(tmp_path)
    prompt = {"path": target, "start": prompt_start, "duration": 1}
    with pytest.raises(InvalidSentenceEntry, match="overlaps"):
        next(iter(make_loader(tmp_path, [row(target, prompt=prompt)])))


def test_adjacent_prompt_is_allowed(tmp_path: Path):
    target = write_audio(tmp_path)
    prompt = {"path": target, "start": 2, "duration": 1}
    batch = next(iter(make_loader(tmp_path, [row(target, prompt=prompt)])))
    assert batch.num_voice_prompt_frames.tolist() == [12]
    assert batch.audio.shape[-1] == 2 * SR


@pytest.mark.parametrize("same_transcript", [False, True])
def test_duplicate_content_is_rejected(tmp_path: Path, same_transcript: bool):
    target = write_audio(tmp_path)
    prompt = {
        "path": write_audio(tmp_path, "copy.wav", 440 if same_transcript else 220),
        "start": 0,
        "duration": 2,
    }
    if same_transcript:
        prompt["transcript"] = TEXT
    with pytest.raises(InvalidSentenceEntry, match="duplicates|repeats"):
        next(iter(make_loader(tmp_path, [row(target, prompt=prompt)])))


@pytest.mark.parametrize("duration,cap,error", [(2, 1, "exceeds"), (4, 5, "shorter")])
def test_full_target_is_never_silently_truncated(
    tmp_path: Path, duration: float, cap: float, error: str
):
    with pytest.raises(InvalidSentenceEntry, match=error):
        next(
            iter(
                make_loader(
                    tmp_path, [row(write_audio(tmp_path), duration=duration)], max_duration_sec=cap
                )
            )
        )


def test_legacy_mode_still_requires_alignment(tmp_path: Path):
    with pytest.raises(ValueError, match="align_data"):
        next(iter(make_loader(tmp_path, [row(write_audio(tmp_path))], sampling_mode="aligned")))
    assert DataArgs().sampling_mode == "aligned"
    assert DataArgs().precompute is True


def test_sentence_mode_rejects_latent_cache(tmp_path: Path):
    with pytest.raises(ValueError, match="precompute"):
        DataArgs(sampling_mode="sentence")
    assert DataArgs(sampling_mode="sentence", precompute=False).sampling_mode == "sentence"
    with pytest.raises(InvalidSentenceEntry, match="cached latents"):
        next(
            iter(make_loader(tmp_path, [row(write_audio(tmp_path), latents_file="x.safetensors")]))
        )


def test_sentence_keeps_partial_final_codec_frame_and_punctuation(tmp_path: Path):
    text = TEXT + " OK."
    batch = next(
        iter(
            make_loader(
                tmp_path,
                [row(write_audio(tmp_path), duration=2.01, transcript=text)],
                final_punct_dropout=1,
            )
        )
    )
    assert batch.num_audio_frames.tolist() == [26]
    assert abs(batch.audio.shape[-1] - round(2.01 * SR)) <= 1
    assert batch.text_tokens[0].tolist() == list(map(ord, text))


def test_no_prompt_teacher_and_student_conditions_match():
    import copy

    from training.tests.test_smoke import LDIM, make_batch, tiny_model

    model = tiny_model("lsd")
    model.args.distill_cfg_coef = 1.0
    model.args.text_dropout = 0.0
    model.args.voice_dropout = 0.0
    teacher = copy.deepcopy(model.flow_lm).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    model.__dict__["distill_teacher"] = teacher
    latents, mask, text, _ = make_batch()
    voice = torch.zeros(latents.shape[0], 0, LDIM)
    lengths = torch.zeros(latents.shape[0], dtype=torch.long)
    loss, metrics = model(latents, mask, text, voice, num_voice_prompt_frames=lengths)
    assert "distill_mse" in metrics
    assert loss.item() < 1e-8
    loss.backward()
    grads = [p.grad for p in model.flow_lm.transformer.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
