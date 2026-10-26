"""Invariants for the bounded teacher-history perturbation."""

from typing import Any

import pytest
import torch

from training.args import TrainArgs, _from_dict
from training.modules.conditioner import build_sequences_with_conditions, teacher_forced_audio_inputs


def inputs(
    args: TrainArgs, latents: torch.Tensor, mask: torch.Tensor, bos: torch.Tensor, *, training: bool
) -> torch.Tensor:
    return teacher_forced_audio_inputs(args, latents, mask, bos, training)


def fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latents = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3) / 10
    # The final false slot is the padded terminal EOS position. The second row
    # has additional right padding, which must never receive perturbation.
    mask = torch.tensor([[True, True, True, True, False], [True, True, False, False, False]])
    bos = torch.full((2, 1, 3), -7.0)
    return latents, mask, bos


def test_zero_sigma_is_bitwise_the_existing_causal_shift():
    latents, mask, bos = fixture()
    expected = torch.cat([bos, latents[:, :-1]], dim=1)
    actual = inputs(TrainArgs(), latents, mask, bos, training=True)
    assert torch.equal(actual, expected)


def test_noise_changes_only_valid_predecessors_without_mutating_targets():
    latents, mask, bos = fixture()
    clean_targets = latents.clone()
    clean_mask = mask.clone()
    expected = torch.cat([bos, latents[:, :-1]], dim=1)
    history_mask = torch.cat([torch.zeros(2, 1, dtype=torch.bool), mask[:, :-1]], dim=1)

    torch.manual_seed(17)
    actual = inputs(TrainArgs(teacher_history_noise_std=0.05), latents, mask, bos, training=True)

    assert torch.equal(latents, clean_targets)
    assert torch.equal(mask, clean_mask)
    assert torch.equal(actual[:, 0], bos[:, 0])
    assert torch.equal(actual[~history_mask], expected[~history_mask])
    assert torch.all(actual[history_mask] != expected[history_mask])
    # Before noise, position t reads clean target t-1: BOS and the causal shift
    # themselves were not changed or replaced with a model rollout.
    assert torch.equal(expected[:, 1:], latents[:, :-1])


def test_evaluation_is_clean_even_when_noise_is_configured():
    latents, mask, bos = fixture()
    expected = torch.cat([bos, latents[:, :-1]], dim=1)
    actual = inputs(TrainArgs(teacher_history_noise_std=0.05), latents, mask, bos, training=False)
    assert torch.equal(actual, expected)


class TinyConditioner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(8, 3)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(tokens)


class TinyFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bos_emb = torch.nn.Parameter(torch.full((1, 3), -7.0))
        self.bos_before_voice = torch.nn.Parameter(torch.full((1, 1, 3), -9.0))
        self.input_linear = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.input_linear.weight.copy_(torch.eye(3))
        self.speaker_proj_weight = torch.nn.Parameter(torch.eye(3))
        self.conditioner = TinyConditioner()


def test_sequence_builder_uses_the_masked_history_inputs():
    latents, mask, _ = fixture()
    flow = TinyFlow()
    args = TrainArgs(teacher_history_noise_std=0.05)
    text = [torch.tensor([1]), torch.tensor([2])]
    voice = torch.empty(2, 0, 3)

    torch.manual_seed(23)
    noisy, prefix = build_sequences_with_conditions(
        args, latents, text, voice, False, flow, mask=mask, training=True
    )
    clean, clean_prefix = build_sequences_with_conditions(
        TrainArgs(), latents, text, voice, False, flow, mask=mask, training=True
    )

    assert torch.equal(prefix, clean_prefix)
    # Audio starts after BOS plus the one token for each text prompt. Its first
    # input is the clean BOS; its next input is the perturbed previous latent.
    assert torch.equal(noisy[:, prefix[0]], clean[:, clean_prefix[0]])
    assert torch.any(noisy[:, prefix[0] + 1] != clean[:, clean_prefix[0] + 1])
    assert torch.any(noisy[:, prefix[0] + 2] != clean[:, clean_prefix[0] + 2])


@pytest.mark.parametrize("sigma", [-0.1, float("nan"), float("inf")])
def test_invalid_noise_is_rejected(sigma: float):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        TrainArgs(teacher_history_noise_std=sigma)


def test_config_parser_converts_noise_to_float():
    args = _from_dict(TrainArgs, {"teacher_history_noise_std": "0.05"})
    assert args.teacher_history_noise_std == 0.05


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distill_cfg_coef": 1.0},
        {"distill_teacher_config": "teacher.yaml", "distill_teacher_weights": "teacher.pt"},
        {"head_distill": True},
    ],
)
def test_noise_rejects_teacher_and_head_distillation(kwargs: dict[str, Any]):
    with pytest.raises(ValueError, match="ordinary teacher training"):
        _from_dict(TrainArgs, {"teacher_history_noise_std": 0.05, **kwargs})
