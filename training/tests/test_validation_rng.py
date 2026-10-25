"""Validation must use comparable noise without advancing training's random stream."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from training import train
from training.args import TrainArgs


class NoisyModel:
    def __init__(self):
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.flow_lm = SimpleNamespace(
            conditioner=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text: [1]))
        )

    def eval(self) -> None:
        pass

    def __call__(self, latents: torch.Tensor, *args: object, **kwargs: object) -> tuple[Any, ...]:
        loss = (latents + torch.randn_like(latents) * self.weight).square().mean()
        return loss, {"loss": loss.detach()}


@pytest.fixture
def validation(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, ...]:
    model = NoisyModel()
    args = TrainArgs(num_valid_batches=2)
    mimi = SimpleNamespace(sample_rate=24000, frame_rate=12.5)
    batch = SimpleNamespace(text_tokens=[torch.tensor([1])])
    monkeypatch.setattr(train, "DataLoader", lambda *args, **kwargs: [batch, batch])

    def encode(*args: object, **kwargs: object) -> tuple[Any, ...]:
        return torch.rand(4), torch.ones(4, dtype=torch.bool), torch.empty(0), torch.tensor([0])

    monkeypatch.setattr(train, "encode_batch", encode)
    monkeypatch.setattr(train, "avg_across_ranks", lambda value: value)
    return model, mimi, args


def test_unchanged_model_has_same_validation_metrics(validation: tuple[Any, ...]) -> None:
    model, mimi, args = validation
    torch.random.default_generator.manual_seed(10)
    first = train.validate(model, mimi, args, torch.device("cpu"), 0, 1, 0)
    torch.rand(29)  # Training can consume arbitrary noise between evaluations.
    second = train.validate(model, mimi, args, torch.device("cpu"), 0, 1, 25)
    assert first == second


def test_validation_preserves_training_rng(validation: tuple[Any, ...]) -> None:
    model, mimi, args = validation
    torch.random.default_generator.manual_seed(71)
    before = torch.random.get_rng_state().clone()
    train.validate(model, mimi, args, torch.device("cpu"), 0, 1, 0)
    assert torch.equal(before, torch.random.get_rng_state())


def test_validation_failure_preserves_training_rng(
    validation: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, mimi, args = validation

    def fail(*args: object, **kwargs: object) -> None:
        torch.rand(3)
        raise RuntimeError("bad validation audio")

    monkeypatch.setattr(train, "encode_batch", fail)
    before = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="bad validation audio"):
        train.validate(model, mimi, args, torch.device("cpu"), 0, 1, 0)
    assert torch.equal(before, torch.random.get_rng_state())
