"""Fresh data adaptation preserves the parent while minting a new identity."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from training.scripts import continue_taigi_training as continuation
from training.scripts import fork_taigi_adaptation as adaptation
from training.scripts.run_experiment import digest, identity
from training.taigi.transcriber import BREEZE_ADAPTER_VERSION, BREEZE_MODEL, BREEZE_REVISION


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _parent(tmp_path: Path, *, sampling_mode: str = "aligned") -> Path:
    run = tmp_path / "parent-training"
    run.mkdir()
    train, valid = tmp_path / "old-train.jsonl", tmp_path / "old-valid.jsonl"
    tokenizer, model = tmp_path / "tokenizer.json", tmp_path / "model.yaml"
    train.write_text('{"transcript":"舊訓練"}\n', encoding="utf-8")
    valid.write_text('{"transcript":"舊保留"}\n', encoding="utf-8")
    tokenizer.write_text('{"vocab_size":4096}\n', encoding="utf-8")
    model.write_text("mimi:\n  sample_rate: 24000\n", encoding="utf-8")
    config = {
        "model_config": str(model),
        "model_overrides": {
            "flow_lm.lookup_table.tokenizer_path": str(tokenizer),
            "flow_lm.lookup_table.n_bins": 4096,
        },
        "start_from_pretrained": True,
        "reset_text_embedding": True,
        "data": {
            "train_jsonl": str(train),
            "valid_jsonl": str(valid),
            "precompute": False,
            "sampling_mode": sampling_mode,
        },
        "flow": {"type": "lsd", "kwargs": {}},
        "optim": {"lr": 3e-5, "warmup_steps": 100},
        "run_dir": str(run),
        "batch_size": 1,
        "grad_accum_steps": 8,
        "flow_batch_multiplier": 1,
        "max_steps": 3000,
    }
    config_path = tmp_path / "parent.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    recorded = identity(config_path, config, [str(tokenizer)])
    (run / "run_manifest.json").write_text(
        json.dumps({"schema_version": 1, "identity": recorded}), encoding="utf-8"
    )
    torch.save(
        {
            "step": 3000,
            "model": {"weight": torch.tensor([2.0])},
            "ema": {"weight": torch.tensor([1.0])},
        },
        run / "checkpoint_00003000.pt",
    )
    torch.save(
        {
            "optimizer": {
                "state": {0: {"exp_avg": torch.tensor([0.5])}},
                "param_groups": [{"lr": 3e-5}],
            }
        },
        run / "optim_00003000.pt",
    )
    return config_path


def _release(tmp_path: Path, *, overlap: str | None = None) -> Path:
    release = tmp_path / "merged" / "release-a"
    audio_dir = release / "audio"
    audio_dir.mkdir(parents=True)
    train_audio, valid_audio = audio_dir / "train.wav", audio_dir / "valid.wav"
    sf.write(train_audio, np.full(24000, 0.1, dtype=np.float32), 24000, subtype="FLOAT")
    if overlap == "audio":
        valid_audio = train_audio
    else:
        sf.write(valid_audio, np.full(24000, -0.1, dtype=np.float32), 24000, subtype="FLOAT")

    rows: list[dict[str, Any]] = []
    for split, audio, text in (
        ("train", train_audio, "今仔日天氣真好"),
        ("valid", valid_audio, "你食飽未"),
    ):
        group = "shared-group" if overlap == "group" else f"{split}-group"
        rows.append(
            {
                "utterance_id": f"merged-{split}",
                "source_id": f"source-{split}",
                "group_id": group,
                "published_audio_path": str(audio),
                "audio_sha256": digest(audio),
                "start": 0.0,
                "duration": 1.0,
                "input_text_zh": text,
                "split": split,
                "source_metadata": {"source_audio_sha256": ("a" if split == "train" else "b") * 64},
                "label_type": "asr_pseudo",
                "label_provenance": "Breeze-ASR-26",
                "label_model": BREEZE_MODEL,
                "label_model_revision": BREEZE_REVISION,
                "label_adapter_version": BREEZE_ADAPTER_VERSION,
                "verification": "unverified_weak_label",
            }
        )
    manifests = {
        split: [
            {
                "path": row["published_audio_path"],
                "start": row["start"],
                "duration": row["duration"],
                "transcript": row["input_text_zh"],
            }
            for row in rows
            if row["split"] == split
        ]
        for split in ("train", "valid")
    }
    _write_jsonl(release / "metadata.jsonl", rows)
    for split, manifest in manifests.items():
        _write_jsonl(release / f"{split}.jsonl", manifest)
    report = {
        "schema_version": "taigi-merged-weak-labels/v1",
        "release_id": release.name,
        "parents": [{"identity_sha256": "c" * 64}],
        "input_weak_label_count": 2,
        "accepted_weak_label_count": 2,
        "deduplicated_count": 0,
        "counts": {"train": 1, "valid": 1},
        "durations_seconds": {"train": 1.0, "valid": 1.0},
        "split_groups": {
            split: sorted({row["group_id"] for row in rows if row["split"] == split})
            for split in ("train", "valid")
        },
        "label_provenance": "Breeze-ASR-26 weak labels; unverified; no quality claim",
        "verified_labels": 0,
        "manifest_sha256": {
            name: digest(release / name)
            for name in ("metadata.jsonl", "train.jsonl", "valid.jsonl")
        },
    }
    (release / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return release


def test_fresh_fork_copies_full_state_and_records_only_allowed_recipe_changes(
    tmp_path: Path,
) -> None:
    parent, release, child = _parent(tmp_path), _release(tmp_path), tmp_path / "child"
    parent_config, _ = continuation.verify_parent(parent)

    config_path = adaptation.fork_adaptation(
        parent,
        release,
        child,
        4000,
        batch_size=16,
        grad_accum_steps=4,
        flow_batch_multiplier=4,
        learning_rate=2e-5,
        precompute=True,
    )

    config, child_identity = continuation.verify_parent(config_path)
    assert config["model_config"] == parent_config["model_config"]
    assert config["model_overrides"] == parent_config["model_overrides"]
    assert config["reset_text_embedding"] is True
    assert config["data"] == {
        **parent_config["data"],
        "train_jsonl": str((child / "data" / "train.jsonl").resolve()),
        "valid_jsonl": str((child / "data" / "valid.jsonl").resolve()),
        "precompute": True,
    }
    assert (config["batch_size"], config["grad_accum_steps"]) == (16, 4)
    assert config["flow_batch_multiplier"] == 4
    assert config["optim"]["lr"] == 2e-5
    assert config["max_steps"] == 4000

    parent_run, child_run = Path(parent_config["run_dir"]), Path(config["run_dir"])
    for name in ("checkpoint_00003000.pt", "optim_00003000.pt"):
        assert (child_run / name).read_bytes() == (parent_run / name).read_bytes()
    checkpoint = torch.load(child_run / "checkpoint_00003000.pt", weights_only=True)
    assert checkpoint["step"] == 3000 and checkpoint["ema"]["weight"].item() == 1.0

    lineage = json.loads((child / "lineage.json").read_text(encoding="utf-8"))
    assert lineage["quality_verified"] is False
    assert lineage["parent_step"] == 3000
    assert set(lineage["config_changes"]) == {
        "batch_size",
        "data.precompute",
        "data.train_jsonl",
        "data.valid_jsonl",
        "flow_batch_multiplier",
        "grad_accum_steps",
        "max_steps",
        "optim.lr",
        "run_dir",
    }
    recorded_paths = {item["path"] for item in child_identity["inputs"].values()}
    assert (
        str(Path(parent_config["model_overrides"]["flow_lm.lookup_table.tokenizer_path"]).resolve())
        in recorded_paths
    )
    assert str((child / "lineage.json").resolve()) in recorded_paths
    provenance = lineage["launcher_provenance"]
    assert provenance["parent_config"] == {"path": str(parent.resolve()), "sha256": digest(parent)}
    assert provenance["parent_checkpoint"] == {
        "path": str((parent_run / "checkpoint_00003000.pt").resolve()),
        "sha256": digest(parent_run / "checkpoint_00003000.pt"),
    }
    assert provenance["parent_optimizer"] == {
        "path": str((parent_run / "optim_00003000.pt").resolve()),
        "sha256": digest(parent_run / "optim_00003000.pt"),
    }
    assert provenance["release_report"] == {
        "path": str((release / "report.json").resolve()),
        "sha256": digest(release / "report.json"),
    }


def test_changed_tokenizer_refuses_fork(tmp_path: Path) -> None:
    parent, release = _parent(tmp_path), _release(tmp_path)
    config, _ = continuation.verify_parent(parent)
    Path(config["model_overrides"]["flow_lm.lookup_table.tokenizer_path"]).write_text("changed")
    child = tmp_path / "child"
    with pytest.raises(ValueError, match="parent input identity changed"):
        adaptation.fork_adaptation(parent, release, child, 4000)
    assert not child.exists()


def test_orphan_parent_checkpoint_refuses_fork(tmp_path: Path) -> None:
    parent, release = _parent(tmp_path), _release(tmp_path)
    config, _ = continuation.verify_parent(parent)
    Path(config["run_dir"], "optim_00003000.pt").unlink()
    child = tmp_path / "child"
    with pytest.raises(ValueError, match="optimizer sidecar"):
        adaptation.fork_adaptation(parent, release, child, 4000)
    assert not child.exists()


@pytest.mark.parametrize("changed", ["parent", "release"])
def test_changed_parent_or_release_manifest_refuses_fork(tmp_path: Path, changed: str) -> None:
    parent, release = _parent(tmp_path), _release(tmp_path)
    if changed == "parent":
        config = yaml.safe_load(parent.read_text(encoding="utf-8"))
        Path(config["data"]["train_jsonl"]).write_text("changed", encoding="utf-8")
        message = "parent input identity changed"
    else:
        with (release / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        message = "manifest failed integrity"
    child = tmp_path / "child"
    with pytest.raises(ValueError, match=message):
        adaptation.fork_adaptation(parent, release, child, 4000)
    assert not child.exists()


def test_endpoint_must_exceed_parent_step(tmp_path: Path) -> None:
    child = tmp_path / "child"
    with pytest.raises(ValueError, match="endpoint must exceed parent step"):
        adaptation.fork_adaptation(_parent(tmp_path), _release(tmp_path), child, 3000)
    assert not child.exists()


@pytest.mark.parametrize("overlap", ["audio", "group"])
def test_train_valid_identity_overlap_is_rejected(tmp_path: Path, overlap: str) -> None:
    child = tmp_path / "child"
    with pytest.raises(ValueError, match=f"train/valid {overlap} overlap"):
        adaptation.fork_adaptation(
            _parent(tmp_path), _release(tmp_path, overlap=overlap), child, 4000
        )
    assert not child.exists()


def test_config_diff_allowlist_rejects_architecture_or_tokenizer_change(tmp_path: Path) -> None:
    parent = yaml.safe_load(_parent(tmp_path).read_text(encoding="utf-8"))
    child = copy.deepcopy(parent)
    child["model_config"] = "different-architecture.yaml"
    child["model_overrides"]["flow_lm.lookup_table.tokenizer_path"] = "other-tokenizer.json"
    with pytest.raises(ValueError, match="unapproved config changes.*model_config"):
        adaptation.verify_config_changes(parent, child, {"run_dir", "max_steps"})


def test_cli_prepares_child_without_starting_training(tmp_path: Path) -> None:
    parent, release, child = _parent(tmp_path), _release(tmp_path), tmp_path / "child"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.scripts.fork_taigi_adaptation",
            "--parent-config",
            str(parent),
            "--release",
            str(release),
            "--child",
            str(child),
            "--endpoint",
            "4000",
            "--batch-size",
            "16",
            "--grad-accum-steps",
            "4",
            "--flow-batch-multiplier",
            "4",
            "--learning-rate",
            "0.00002",
            "--precompute",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == str(child / "training.yaml")
    assert not (child / "training" / "progress.jsonl").exists()


def test_sentence_sampling_rejects_precompute_before_child_creation(tmp_path: Path) -> None:
    child = tmp_path / "child"
    with pytest.raises(ValueError, match="sentence sampling requires data.precompute: false"):
        adaptation.fork_adaptation(
            _parent(tmp_path, sampling_mode="sentence"),
            _release(tmp_path),
            child,
            4000,
            precompute=True,
        )
    assert not child.exists()
