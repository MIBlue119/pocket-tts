from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from training.taigi import evaluate as evaluator
from training.taigi.evaluate import audio_metrics, read_prompts


def test_metrics_distinguish_empty_silent_and_invalid_audio():
    assert audio_metrics(np.array([], dtype=np.float32), 24000)["status"] == "empty"
    assert audio_metrics(np.zeros(24000, dtype=np.float32), 24000)["status"] == "silent"
    assert audio_metrics(np.array([np.nan]), 24000)["status"] == "nonfinite"
    metrics = audio_metrics(np.array([0.5, -1.1], dtype=np.float32), 24000)
    assert metrics["status"] == "audio"
    assert metrics["clipping_fraction"] == 0.5


def test_prompt_identity_is_unique_and_bounded(tmp_path: Path):
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"id":"negation","text":"我今天不去台北。"}\n')
    assert read_prompts(path)[0]["text"] == "我今天不去台北。"
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="unique"):
        read_prompts(path)
    path.write_text('{"id":"empty","text":" "}\n')
    with pytest.raises(ValueError, match="text"):
        read_prompts(path)


class _DeviceModule:
    frame_rate = 12.5

    def to(self: _DeviceModule, _device: object) -> _DeviceModule:
        return self

    def eval(self: _DeviceModule) -> _DeviceModule:
        return self


def _args(tmp_path: Path, cfg_coef: float) -> argparse.Namespace:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"id":"one","text":"測試。"}\n', encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("fixture: true\n", encoding="utf-8")
    return argparse.Namespace(
        prompts=prompts,
        max_frames=8,
        cfg_coef=cfg_coef,
        output=tmp_path / "output",
        config=config,
        checkpoint=None,
        device="cpu",
        seed=17,
        asr=False,
    )


def test_nondefault_cfg_coefficient_reaches_generate_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"fixture tokenizer")
    config = SimpleNamespace(
        seed=3, model_overrides={"flow_lm.lookup_table.tokenizer_path": str(tokenizer)}
    )
    observed: list[float] = []

    class Model(_DeviceModule):
        ldim = 2
        flow_lm = SimpleNamespace(
            conditioner=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda _text: [1, 2]))
        )

        def generate(
            self: Model,
            _tokens: object,
            _prompts: object,
            *,
            max_frames: int,
            temp: float,
            cfg_coef: float,
        ) -> list[torch.Tensor]:
            assert max_frames == 8 and temp == 0.3
            observed.append(cfg_coef)
            return [torch.zeros(4, self.ldim)]

    mimi = _DeviceModule()
    monkeypatch.setattr(evaluator, "load_args", lambda _path: config)
    monkeypatch.setattr(evaluator, "build_models", lambda _config: (Model(), mimi, None))

    report_path = evaluator.evaluate(_args(tmp_path, 2.0))

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert observed == [2.0]
    assert report["cfg_coef"] == 2.0


@pytest.mark.parametrize("cfg_coef", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_cfg_coefficient_rejected_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cfg_coef: float
) -> None:
    monkeypatch.setattr(
        evaluator,
        "build_models",
        lambda _config: pytest.fail("model must not load for invalid cfg-coef"),
    )

    with pytest.raises(ValueError, match="cfg-coef must be a finite positive float"):
        evaluator.evaluate(_args(tmp_path, cfg_coef))
    assert not (tmp_path / "output").exists()
