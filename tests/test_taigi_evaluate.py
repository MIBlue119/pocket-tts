from pathlib import Path

import numpy as np
import pytest

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
