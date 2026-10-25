from pathlib import Path
import json

import numpy as np
import pytest
import soundfile as sf

from training.taigi.audio import sha256_file
from training.taigi.enhance import enhance_file, overlap_enhance, REVISION


@pytest.mark.parametrize("length", [1, 1000, 4000, 4001, 9003, 40000])
def test_overlap_preserves_samples_and_bounds_model_input(length: int) -> None:
    samples = np.random.default_rng(0).normal(size=length).astype(np.float32)
    sizes = []

    def identity(clip: np.ndarray, rate: int) -> np.ndarray:
        sizes.append(len(clip))
        return clip.copy()

    result = overlap_enhance(samples, 8000, identity, 0.5)
    np.testing.assert_allclose(result, samples, atol=1e-6)
    assert max(sizes) <= 4000
    assert len(result) == length


def test_invalid_prediction_rejected() -> None:
    with pytest.raises(ValueError, match="invalid samples"):
        overlap_enhance(np.ones(1000), 8000, lambda clip, rate: clip[:-1])
    with pytest.raises(ValueError, match="invalid samples"):
        overlap_enhance(np.ones(1000), 8000, lambda clip, rate: clip * np.nan)


def test_file_provenance_and_bounded_interval(tmp_path: Path) -> None:
    source = tmp_path / "raw.wav"
    output = tmp_path / "enhanced.wav"
    sf.write(source, np.linspace(-0.2, 0.2, 24000), 8000, subtype="FLOAT")
    digest = sha256_file(source)
    path = enhance_file(source, output, lambda clip, rate: clip, start=1, duration=0.5)
    metadata = json.loads(path.read_text())
    samples, rate = sf.read(output)
    assert len(samples) == 4000
    assert rate == 8000
    assert metadata["source_start_sample"] == 8000
    assert metadata["source_audio_sha256"] == digest == sha256_file(source)
    assert metadata["audio_sha256"] == sha256_file(output)
    assert metadata["enhancement_revision"] == REVISION
    assert metadata["verification"] == "unverified_enhancement"
    with pytest.raises(FileExistsError):
        enhance_file(source, output, lambda clip, rate: clip)
