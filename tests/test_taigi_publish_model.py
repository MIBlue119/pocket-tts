import argparse
from pathlib import Path
from typing import BinaryIO, cast
from types import SimpleNamespace

from huggingface_hub import HfApi

import pytest

from training.scripts import publish_taigi_model as publisher


def test_package_refuses_missing_inference_weights(tmp_path: Path) -> None:
    args = publisher.parser().parse_args(
        [
            "--weights",
            str(tmp_path / "missing.safetensors"),
            "--config",
            "missing.yaml",
            "--tokenizer",
            "missing.json",
            "--provenance",
            "missing-provenance.json",
            "--evaluation",
            "missing-evaluation.json",
            "--package",
            str(tmp_path / "package"),
            "--candidate-id",
            "diagnostic-001",
            "--weights-variant",
            "raw",
            "--source-checkpoint-sha256",
            "a" * 64,
        ]
    )
    with pytest.raises(FileNotFoundError):
        publisher.package(args)
    assert not args.package.exists()


def fixture_inputs(tmp_path: Path) -> argparse.Namespace:
    import json

    import torch
    import yaml
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models

    config = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1] / "pocket_tts/config/english_2026-04_24l.yaml"
        ).read_text()
    )
    config["flow_lm"]["transformer"].update(d_model=8, num_heads=2, num_layers=1, hidden_scale=2)
    config["flow_lm"]["flow"].update(dim=8, depth=1)
    config["flow_lm"]["lookup_table"].update(dim=8, n_bins=3)
    config["mimi"].update(inner_dim=2, outer_dim=8)
    config["mimi"]["seanet"].update(dimension=8, n_filters=2, ratios=[2, 2])
    config["mimi"]["transformer"].update(
        d_model=8, num_heads=2, dim_feedforward=16, input_dimension=8, output_dimensions=[8]
    )
    config["mimi"]["quantizer"].update(dimension=2, output_dimension=8)
    tokenizer = tmp_path / "tokenizer.json"
    Tokenizer(models.WordLevel({"<unk>": 0, "你": 1, "好": 2}, unk_token="<unk>")).save(
        str(tokenizer)
    )
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(yaml.safe_dump(config))
    parsed = publisher.runtime_config(config_path, tokenizer)
    weights = tmp_path / "export.safetensors"
    save_file(
        {key: torch.zeros(shape) for key, shape in publisher.expected_shapes(parsed).items()},
        str(weights),
    )
    checkpoint = "a" * 64
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "weights_variant": "raw",
                "source_checkpoint_sha256": checkpoint,
                "exported_weights_sha256": publisher.sha256(weights),
            }
        )
    )
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "state": "complete",
                "weights": "raw_checkpoint",
                "checkpoint_sha256": checkpoint,
                "tokenizer_sha256": publisher.sha256(tokenizer),
                "quality_verified": False,
            }
        )
    )
    return publisher.parser().parse_args(
        [
            "--weights",
            str(weights),
            "--config",
            str(config_path),
            "--tokenizer",
            str(tokenizer),
            "--provenance",
            str(provenance),
            "--evaluation",
            str(evaluation),
            "--output",
            str(tmp_path / "package"),
            "--candidate-id",
            "pilot-001",
        ]
    )


def test_package_is_complete_allowlisted_and_runnable_example_uses_real_api(tmp_path: Path) -> None:
    import json

    import yaml

    args = fixture_inputs(tmp_path)
    package = publisher.package(args)
    manifest = publisher.validate_package(package)
    assert {p.name for p in package.iterdir()} == publisher.FILES
    assert manifest["evaluation_identity_matches"] is True
    assert manifest["quality_verified"] is False
    assert manifest["weights_variant"] == "raw"
    config = yaml.safe_load((package / "config.yaml").read_text())
    assert config["weights_path"] == "model.safetensors"
    assert config["flow_lm"]["lookup_table"]["tokenizer_path"] == "tokenizer.json"
    assert config["weights_path_without_voice_cloning"] is None
    compile((package / "inference.py").read_text(), "inference.py", "exec")
    assert "get_state_for_audio_prompt" in (package / "inference.py").read_text()
    assert json.loads((package / "evaluation.json").read_text())["weights"] == "raw_checkpoint"
    (package / "optim_001.pt").write_text("must not upload")
    with pytest.raises(ValueError, match="exactly"):
        publisher.validate_package(package)


def test_raw_evaluation_cannot_be_attached_to_ema_export(tmp_path: Path) -> None:
    args = fixture_inputs(tmp_path)
    args.weights_variant = "ema"
    with pytest.raises(ValueError, match="variant"):
        publisher.package(args)
    assert not args.package.exists()


def test_explicit_unevaluated_export_stays_experimental(tmp_path: Path) -> None:
    args = fixture_inputs(tmp_path)
    args.weights_variant = "ema"
    import json

    provenance = json.loads(args.provenance.read_text())
    provenance["weights_variant"] = "ema"
    args.provenance.write_text(json.dumps(provenance))
    args.evaluation.write_text(
        '{"state":"not_evaluated","reason":"EMA export; only raw checkpoint was evaluated"}'
    )
    directory = publisher.package(args)
    manifest = publisher.validate_package(directory)
    assert manifest["evaluation_identity_matches"] is False
    assert manifest["quality_status"] == "experimental_unverified"


def test_credentials_in_explicit_metadata_are_rejected() -> None:
    with pytest.raises(ValueError, match="credential"):
        publisher.reject_credentials({"nested": {"hf_token": "must not print"}})


class FakeHub:
    def __init__(self, directory: Path, private: bool = True):
        self.directory = directory
        self.private = private
        self.uploads = []
        self.corrupt = False
        self.public_after_upload = False

    def whoami(self) -> dict[str, str]:
        return {"name": "owner"}

    def repo_info(self, repo_id: str, **kwargs: object) -> SimpleNamespace:
        siblings = []
        if self.uploads:
            for path in self.directory.iterdir():
                siblings.append(
                    SimpleNamespace(
                        rfilename="candidates/pilot-001/" + path.name,
                        size=path.stat().st_size,
                        lfs={"sha256": "bad" if self.corrupt else publisher.sha256(path)},
                    )
                )
        return SimpleNamespace(
            private=self.private and not (self.uploads and self.public_after_upload),
            siblings=siblings,
            sha="parent-sha",
        )

    def upload_folder(self, **kwargs: object) -> SimpleNamespace:
        self.uploads.append(kwargs)
        return SimpleNamespace(oid="immutable-commit-sha")


def test_private_upload_uses_versioned_path_and_verifies_commit_hashes(tmp_path: Path) -> None:
    directory = publisher.package(fixture_inputs(tmp_path))
    hub = FakeHub(directory)
    receipt = publisher.publish(directory, "owner/pocket-taigi", cast(HfApi, hub))
    assert receipt["private"] is True
    assert receipt["revision"] == "immutable-commit-sha"
    assert set(receipt["verified_files"]) == publisher.FILES
    assert hub.uploads[0]["path_in_repo"] == "candidates/pilot-001"
    assert hub.uploads[0]["parent_commit"] == "parent-sha"
    assert set(cast(list[str], hub.uploads[0]["allow_patterns"])) == publisher.FILES
    with pytest.raises(ValueError, match="already exists"):
        publisher.publish(directory, "owner/pocket-taigi", cast(HfApi, hub))


def test_public_repository_refused_before_any_upload(tmp_path: Path) -> None:
    directory = publisher.package(fixture_inputs(tmp_path))
    hub = FakeHub(directory, private=False)
    with pytest.raises(ValueError, match="public"):
        publisher.publish(directory, "owner/pocket-taigi", cast(HfApi, hub))
    assert not hub.uploads


def test_personal_namespace_enforced_before_upload(tmp_path: Path) -> None:
    directory = publisher.package(fixture_inputs(tmp_path))
    hub = FakeHub(directory)
    with pytest.raises(ValueError, match="personal namespace"):
        publisher.publish(directory, "other/pocket-taigi", cast(HfApi, hub))
    assert not hub.uploads


@pytest.mark.parametrize("failure", ["hash", "privacy"])
def test_no_success_receipt_if_remote_verification_fails(tmp_path: Path, failure: str) -> None:
    directory = publisher.package(fixture_inputs(tmp_path))
    hub = FakeHub(directory)
    hub.corrupt = failure == "hash"
    hub.public_after_upload = failure == "privacy"
    with pytest.raises(ValueError, match="checksum|privacy"):
        publisher.publish(directory, "owner/pocket-taigi", cast(HfApi, hub))


@pytest.mark.parametrize("with_voice", [False, True])
@pytest.mark.parametrize("invalid_audio", [None, "empty", "nonfinite"])
def test_inference_example_resolves_package_paths_and_calls_actual_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_voice: bool, invalid_audio: str | None
) -> None:
    import runpy
    import sys

    import torch
    import yaml
    from pocket_tts import TTSModel
    from pocket_tts.modules import stateful_module

    directory = publisher.package(fixture_inputs(tmp_path))
    observed = []
    voice = tmp_path / "own-reference.wav"
    voice.touch()

    class Model:
        sample_rate = 24000
        flow_lm = object()

        def to(self, device: str) -> "Model":
            assert device == "cpu"
            return self

        def get_state_for_audio_prompt(self, path: Path) -> dict[str, str]:
            assert with_voice and path == voice
            return {"voice": "own"}

        def generate_audio(
            self, state: dict[str, str], text: str, frames_after_eos: int
        ) -> torch.Tensor:
            assert state == ({"voice": "own"} if with_voice else {"prompt": "zero"})
            assert text == "你好" and frames_after_eos == 1
            if invalid_audio == "empty":
                return torch.zeros(0)
            if invalid_audio == "nonfinite":
                return torch.tensor([float("nan"), float("inf"), 0.0])
            return torch.zeros(240)

    def load(
        config: Path,
        temp: float,
        eos_threshold: float,
        noise_clamp: None,
        sampler_decode_steps: int,
    ) -> Model:
        assert (temp, eos_threshold, noise_clamp, sampler_decode_steps) == (0.3, -1, None, 1)
        parsed = yaml.safe_load(config.read_text())
        observed.append(parsed)
        assert parsed["weights_path"] == str(directory / "model.safetensors")
        assert parsed["flow_lm"]["lookup_table"]["tokenizer_path"] == str(
            directory / "tokenizer.json"
        )
        return Model()

    monkeypatch.setattr(TTSModel, "load_model", load)
    monkeypatch.setattr(
        stateful_module,
        "init_states",
        lambda model, batch_size, sequence_length: {"prompt": "zero"}
        if batch_size == 1 and sequence_length == 0
        else pytest.fail(),
    )
    argv = ["inference.py", "--text", "你好", "--output", str(tmp_path / "candidate.wav")]
    if with_voice:
        argv.extend(["--voice", str(voice)])
    monkeypatch.setattr(sys, "argv", argv)
    if invalid_audio:
        with pytest.raises(ValueError, match=f"generated {invalid_audio} audio tensor"):
            runpy.run_path(str(directory / "inference.py"), run_name="__main__")
        assert not (tmp_path / "candidate.wav").exists()
    else:
        runpy.run_path(str(directory / "inference.py"), run_name="__main__")
        assert observed and (tmp_path / "candidate.wav").is_file()


def test_incomplete_tensor_bundle_is_refused(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    args = fixture_inputs(tmp_path)
    save_file({"flow_lm.unrelated": torch.zeros(1)}, str(args.weights))
    with pytest.raises(ValueError, match="incomplete/incompatible"):
        publisher.package(args)
    assert not args.package.exists()


def test_new_repository_creation_explicitly_requests_private(tmp_path: Path) -> None:
    from huggingface_hub.errors import RepositoryNotFoundError

    directory = publisher.package(fixture_inputs(tmp_path))

    class NewHub(FakeHub):
        created = False

        def repo_info(self, repo_id: str, **kwargs: object) -> SimpleNamespace:
            if not self.created:
                raise RepositoryNotFoundError("missing private repo")
            return super().repo_info(repo_id, **kwargs)

        def create_repo(self, repo_id: str, **kwargs: object) -> None:
            assert repo_id == "owner/pocket-taigi"
            assert kwargs == {"repo_type": "model", "private": True, "exist_ok": False}
            self.created = True

    hub = NewHub(directory)
    receipt = publisher.publish(directory, "owner/pocket-taigi", cast(HfApi, hub))
    assert hub.created and receipt["private"] is True


@pytest.mark.parametrize("failure", [None, "exit", "nonfinite", "empty", "timeout"])
def test_runtime_smoke_executes_zero_prompt_cpu_and_refuses_failed_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    import json
    import subprocess

    import numpy as np
    import soundfile as sf

    directory = publisher.package(fixture_inputs(tmp_path))
    output = tmp_path / "selected.runtime.wav"

    def execute(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert argv[1] == str(directory / "inference.py")
        assert "--voice" not in argv
        assert argv[argv.index("--device") + 1] == "cpu"
        assert argv[argv.index("--text") + 1] == "你好，今天天氣真好。"
        assert kwargs["timeout"] == 120
        env = cast(dict[str, str], kwargs["env"])
        assert env["PYTHONPATH"] == str(Path(__file__).resolve().parents[1])
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        cast(BinaryIO, kwargs["stdout"]).write(b"WARNING Maximum generation length reached\n")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 120)
        if failure == "exit":
            return subprocess.CompletedProcess(argv, 1)
        samples = np.zeros(0 if failure == "empty" else 240, dtype=np.float32)
        if failure == "nonfinite":
            samples[0] = np.nan
        sf.write(output, samples, 24000, subtype="FLOAT")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(publisher.subprocess, "run", execute)
    if failure:
        with pytest.raises((ValueError, RuntimeError, subprocess.TimeoutExpired)):
            publisher.runtime_smoke(directory, output, 120)
    else:
        report = publisher.runtime_smoke(directory, output, 120)
        assert report["duration_seconds"] == 0.01
        assert report["rms"] == 0
        assert report["quality_verified"] is False
        assert report["warnings"] == ["WARNING Maximum generation length reached"]
        assert report["audio_sha256"] == publisher.sha256(output)
    saved = json.loads(output.with_suffix(".json").read_text())
    assert saved["state"] == ("failed" if failure else "complete")
    assert saved["voice_prompt_frames"] == 0
    assert output.with_suffix(".log").is_file()
    assert {path.name for path in directory.iterdir()} == publisher.FILES


def test_publish_cli_cannot_upload_after_runtime_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    args = fixture_inputs(tmp_path)
    directory = publisher.package(args)
    monkeypatch.setattr(publisher, "package", lambda _: directory)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publisher",
            "--weights",
            "w",
            "--config",
            "c",
            "--tokenizer",
            "t",
            "--provenance",
            "p",
            "--evaluation",
            "e",
            "--output",
            "out",
            "--candidate-id",
            "candidate",
            "--repo-id",
            "owner/repo",
            "--publish",
        ],
    )

    def failed_runtime(*_args: object) -> None:
        raise RuntimeError("CPU runtime failed")

    monkeypatch.setattr(publisher, "runtime_smoke", failed_runtime)
    monkeypatch.setattr(publisher, "publish", lambda *_: pytest.fail())
    assert publisher.main() == 2
    assert not directory.with_name(directory.name + ".publication.json").exists()


def test_runtime_timeout_and_output_location_are_bounded(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at most 300"):
        publisher.runtime_smoke(tmp_path / "package", tmp_path / "runtime.wav", 301)
    with pytest.raises(ValueError, match="outside"):
        publisher.runtime_smoke(tmp_path, tmp_path / "inside.wav", 120)
