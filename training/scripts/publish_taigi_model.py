"""Package an explicit inference export; publish only to a private personal HF repo.

Default is local packaging only. Credentials come from Hugging Face's existing
local login. No optimizer, source audio, or training directory is uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import tempfile
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi

FILES = {
    "model.safetensors",
    "config.yaml",
    "tokenizer.json",
    "provenance.json",
    "evaluation.json",
    "README.md",
    "inference.py",
    "manifest.json",
}
INFERENCE = '''"""Run from the same pocket-tts checkout used for this candidate."""
import argparse
import tempfile
from pathlib import Path

import soundfile as sf
import torch
import yaml
from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import init_states

parser = argparse.ArgumentParser()
parser.add_argument("--text", required=True)
parser.add_argument("--voice", type=Path, help="Optional reference WAV; voice cloning is experimental")
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--output", type=Path, default=Path("candidate.wav"))
args = parser.parse_args()
root = Path(__file__).resolve().parent
config = yaml.safe_load((root / "config.yaml").read_text())
config["weights_path"] = str(root / "model.safetensors")
config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(root / "tokenizer.json")
with tempfile.TemporaryDirectory() as temporary:
    resolved = Path(temporary) / "config.yaml"
    resolved.write_text(yaml.safe_dump(config))
    torch.manual_seed(17)
    model = TTSModel.load_model(config=resolved, temp=0.3, eos_threshold=-1,
                               noise_clamp=None, sampler_decode_steps=1).to(args.device)
    state = (model.get_state_for_audio_prompt(args.voice.resolve()) if args.voice else
             init_states(model.flow_lm, batch_size=1, sequence_length=0))
    torch.manual_seed(17)
    audio = model.generate_audio(state, args.text, frames_after_eos=1)
    if audio.numel() == 0:
        raise ValueError("runtime generated empty audio tensor")
    if not bool(torch.isfinite(audio).all()):
        raise ValueError("runtime generated nonfinite audio tensor")
    sf.write(args.output, audio.detach().cpu().numpy(), model.sample_rate)
print(args.output)
'''


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    reject_credentials(value)
    return value


def reject_credentials(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {
                "token",
                "hf_token",
                "api_key",
                "password",
                "secret",
                "authorization",
                "credentials",
                "access_token",
            }:
                raise ValueError("metadata contains a credential field")
            reject_credentials(item)
    elif isinstance(value, list):
        for item in value:
            reject_credentials(item)
    elif isinstance(value, str) and (
        re.search(r"\bhf_[A-Za-z0-9]{20,}\b", value) or "PRIVATE KEY-----" in value
    ):
        raise ValueError("metadata appears to contain a credential")


def expected_shapes(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Build the real runtime architecture on meta, without weights or a GPU."""
    import torch

    from pocket_tts.models.flow_lm import FlowLMModel
    from pocket_tts.models.mimi import build_mimi
    from pocket_tts.utils.config import Config

    parsed = Config(**config)
    with torch.device("meta"):
        flow = FlowLMModel.from_pydantic_config(
            parsed.flow_lm,
            latent_dim=parsed.mimi.quantizer.dimension,
            insert_bos_before_voice=parsed.flow_lm.insert_bos_before_voice,
        )
        flow.speaker_proj_weight = torch.nn.Parameter(
            torch.empty(
                parsed.flow_lm.transformer.d_model,
                parsed.mimi.inner_dim or parsed.mimi.seanet.dimension,
            )
        )
        mimi = build_mimi(parsed.mimi)
    return {
        **{f"flow_lm.{key}": tuple(value.shape) for key, value in flow.state_dict().items()},
        **{f"mimi.{key}": tuple(value.shape) for key, value in mimi.state_dict().items()},
    }


def validate_weights(weights: Path, config: dict[str, Any]) -> None:
    from safetensors import safe_open

    expected = expected_shapes(config)
    with safe_open(weights, framework="pt", device="cpu") as tensors:
        observed = {key: tuple(tensors.get_slice(key).get_shape()) for key in tensors.keys()}
    if observed != expected:
        missing = sorted(expected.keys() - observed.keys())
        extra = sorted(observed.keys() - expected.keys())
        mismatched = sorted(
            key for key in expected.keys() & observed.keys() if expected[key] != observed[key]
        )
        raise ValueError(
            f"incomplete/incompatible inference weights: missing={missing[:5]}, extra={extra[:5]}, shapes={mismatched[:5]}"
        )


def runtime_config(config_path: Path, tokenizer: Path) -> dict[str, Any]:
    from tokenizers import Tokenizer

    from pocket_tts.utils.config import Config

    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("runtime config must be a mapping")
    reject_credentials(config)
    # A complete bundle has no external model or tokenizer fallback.
    config["weights_path"] = "model.safetensors"
    config["weights_path_without_voice_cloning"] = None
    config["flow_lm"]["weights_path"] = None
    config["mimi"]["weights_path"] = None
    table = config["flow_lm"]["lookup_table"]
    if table["tokenizer"] != "tokenizers":
        raise ValueError("publisher requires the converted tokenizer.json backend")
    table["tokenizer_path"] = str(tokenizer.resolve())
    parsed = Config(**config)
    if Tokenizer.from_file(str(tokenizer)).get_vocab_size() != parsed.flow_lm.lookup_table.n_bins:
        raise ValueError("tokenizer vocabulary size does not match runtime config")
    return parsed.model_dump()


def validate_package(directory: Path) -> dict[str, Any]:
    paths = list(directory.iterdir())
    if {path.name for path in paths} != FILES or any(
        not path.is_file() or path.is_symlink() for path in paths
    ):
        raise ValueError("package must contain exactly the approved inference files")
    manifest = json_object(directory / "manifest.json")
    if set(manifest["files"]) != FILES - {"manifest.json"}:
        raise ValueError("incomplete package manifest")
    for name, identity in manifest["files"].items():
        path = directory / name
        if path.stat().st_size != identity["size"] or sha256(path) != identity["sha256"]:
            raise ValueError(f"package integrity failure: {name}")
    config = runtime_config(directory / "config.yaml", directory / "tokenizer.json")
    validate_weights(directory / "model.safetensors", config)
    return manifest


def package(args: argparse.Namespace) -> Path:
    inputs = {
        key: Path(getattr(args, key)).resolve(strict=True)
        for key in ("weights", "config", "tokenizer", "provenance", "evaluation")
    }
    if inputs["weights"].suffix != ".safetensors":
        raise ValueError(
            "supply an exported combined .safetensors file, never a training checkpoint"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.candidate_id):
        raise ValueError("invalid candidate id")
    provenance = json_object(inputs["provenance"])
    evaluation = json_object(inputs["evaluation"])
    variant = args.weights_variant or provenance.get("weights_variant")
    checkpoint = (
        args.source_checkpoint_sha256
        or provenance.get("source_checkpoint_sha256")
        or provenance.get("checkpoint_sha256")
    )
    if (
        variant not in {"raw", "ema"}
        or not isinstance(checkpoint, str)
        or not re.fullmatch(r"[0-9a-f]{64}", checkpoint)
    ):
        raise ValueError(
            "explicit raw/ema weights variant and source checkpoint SHA256 are required"
        )
    if provenance.get("weights_variant", variant) != variant:
        raise ValueError("weights variant conflicts with supplied provenance")
    declared_checkpoint = provenance.get(
        "source_checkpoint_sha256", provenance.get("checkpoint_sha256", checkpoint)
    )
    if declared_checkpoint != checkpoint:
        raise ValueError("source checkpoint conflicts with supplied provenance")
    if evaluation.get("state") not in {"complete", "not_evaluated"}:
        raise ValueError("evaluation must be complete or explicitly not_evaluated")
    if evaluation.get("state") != "not_evaluated":
        if evaluation.get("weights") != f"{variant}_checkpoint":
            raise ValueError("evaluation weights variant does not match export provenance")
        if evaluation.get("checkpoint_sha256") != checkpoint:
            raise ValueError("evaluation checkpoint does not match export provenance")
        if evaluation.get("tokenizer_sha256") != sha256(inputs["tokenizer"]):
            raise ValueError("evaluation tokenizer does not match export")
    config = runtime_config(inputs["config"], inputs["tokenizer"])
    validate_weights(inputs["weights"], config)
    weight_hash = sha256(inputs["weights"])
    if provenance.get("exported_weights_sha256", weight_hash) != weight_hash:
        raise ValueError("exported weights do not match provenance")
    matched = (
        evaluation.get("weights") == f"{variant}_checkpoint"
        and evaluation.get("checkpoint_sha256") == checkpoint
        and evaluation.get("tokenizer_sha256") == sha256(inputs["tokenizer"])
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": args.candidate_id,
        "quality_status": "experimental_unverified",
        "quality_verified": False,
        "weights_variant": variant,
        "source_checkpoint_sha256": checkpoint,
        "exported_weights_sha256": weight_hash,
        "evaluation_identity_matches": matched,
        "evaluation_identity_meaning": "declared checkpoint and tokenizer match; export derivation is caller-supplied provenance",
    }
    destination = args.package.resolve()
    if destination.exists():
        raise FileExistsError("package destination already exists; use a new candidate directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".taigi-package-", dir=destination.parent))
    try:
        shutil.copyfile(inputs["weights"], temporary / "model.safetensors")
        if sha256(temporary / "model.safetensors") != weight_hash:
            raise ValueError("inference weights changed during packaging")
        shutil.copyfile(inputs["tokenizer"], temporary / "tokenizer.json")
        config["flow_lm"]["lookup_table"]["tokenizer_path"] = "tokenizer.json"
        (temporary / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (temporary / "provenance.json").write_text(
            json.dumps(
                {"source": provenance, "publication": manifest}, indent=2, ensure_ascii=False
            )
            + "\n"
        )
        (temporary / "evaluation.json").write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n"
        )
        (temporary / "inference.py").write_text(INFERENCE)
        (temporary / "README.md").write_text(
            f"# Taiwanese speech candidate: {args.candidate_id}\n\n"
            "Experimental Mandarin-text-to-Taiwanese-speech adaptation. Quality, pronunciation, and semantic accuracy remain unverified. "
            "Breeze consistency is not independent proof of Taiwanese speech. This is a candidate, not a promoted production model.\n\n"
            f"Weights: **{variant}**. Evaluation identity matches supplied checkpoint/tokenizer: **{matched}**. "
            "Export derivation is recorded from supplied provenance; raw-checkpoint evaluation must not be claimed for EMA exports.\n\n"
            "Training uses weak labels; consult provenance for source permissions and limitations. No training audio or optimizer state is included.\n\n"
            "Run from the same pocket-tts checkout as the candidate with its dependencies installed. Default inference uses no audio prompt, matching the training condition:\n\n"
            "```bash\npython inference.py --device cpu --text '你好，今天天氣真好。' --output candidate.wav\n```\n\n"
            "The script resolves the packaged weights and custom tokenizer before calling TTSModel.load_model(config=...). "
            "It uses seed 17, temperature 0.3, EOS threshold -1, no noise clamp, one decode step, and one frame after EOS. "
            "Optional --voice /path/to/reference.wav enables experimental voice conditioning. "
            "The public streaming runtime differs from the offline diagnostic evaluator; this example is not proof of equivalent output or quality.\n"
        )
        manifest["files"] = {
            path.name: {"sha256": sha256(path), "size": path.stat().st_size}
            for path in sorted(temporary.iterdir())
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        validate_package(temporary)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def runtime_smoke(directory: Path, output: Path, timeout: float = 300) -> dict[str, Any]:
    """Execute the selected package on CPU; this checks execution, not speech quality."""
    import numpy as np
    import soundfile as sf

    if not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise ValueError("runtime smoke timeout must be positive and at most 300 seconds")
    directory, output = directory.resolve(), output.resolve()
    if output.is_relative_to(directory) or output.suffix != ".wav":
        raise ValueError("runtime WAV must be outside the inference package")
    report_path, log = output.with_suffix(".json"), output.with_suffix(".log")
    if any(path.exists() for path in (output, report_path, log)):
        raise FileExistsError("runtime smoke outputs already exist")
    validate_package(directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    report: dict[str, Any] = {
        "state": "running",
        "quality_verified": False,
        "device": "cpu",
        "voice_prompt_frames": 0,
        "text": "你好，今天天氣真好。",
        "timeout_seconds": timeout,
        "package_manifest_sha256": sha256(directory / "manifest.json"),
        "started_at": time.time(),
        "log": str(log),
        "meaning": "public runtime execution only; no independent speech-quality verification",
    }

    def save() -> None:
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
        temporary.replace(report_path)

    save()
    try:
        with log.open("wb") as stream:
            result = subprocess.run(
                [
                    sys.executable,
                    str(directory / "inference.py"),
                    "--device",
                    "cpu",
                    "--text",
                    report["text"],
                    "--output",
                    str(output),
                ],
                cwd=root,
                env={**os.environ, "PYTHONPATH": str(root), "CUDA_VISIBLE_DEVICES": ""},
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        report["exit_code"] = result.returncode
        if result.returncode != 0:
            raise RuntimeError("selected-package public runtime command failed")
        audio, rate = sf.read(output, dtype="float32", always_2d=True)
        if audio.size == 0 or rate <= 0:
            raise ValueError("selected-package runtime produced empty audio")
        if not np.isfinite(audio).all():
            raise ValueError("selected-package runtime produced nonfinite audio")
        report.update(
            state="complete",
            audio=str(output),
            audio_sha256=sha256(output),
            sample_rate=rate,
            frames=len(audio),
            duration_seconds=len(audio) / rate,
            rms=float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
        )
        return report
    except BaseException as error:
        report.update(state="failed", error_type=type(error).__name__)
        raise
    finally:
        report["ended_at"] = time.time()
        if log.exists():
            report["log_sha256"] = sha256(log)
            report["warnings"] = [
                line[-500:]
                for line in log.read_text(errors="replace").splitlines()
                if "warn" in line.lower() or ("max" in line.lower() and "generat" in line.lower())
            ][-20:]
        save()


def publish(directory: Path, repo_id: str, api: HfApi | None = None) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError

    manifest = validate_package(directory)
    api = api or HfApi(token=True)
    owner = api.whoami()["name"]
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id)
        or repo_id.split("/")[0] != owner
    ):
        raise ValueError("publication requires the logged-in user's personal namespace")
    try:
        info = api.repo_info(repo_id, repo_type="model", files_metadata=True)
    except RepositoryNotFoundError:
        api.create_repo(repo_id, repo_type="model", private=True, exist_ok=False)
        info = api.repo_info(repo_id, repo_type="model", files_metadata=True)
    if info.private is not True:
        raise ValueError("refusing publication: existing repository is public")
    prefix = f"candidates/{manifest['candidate_id']}"
    if any(item.rfilename.startswith(prefix + "/") for item in (info.siblings or [])):
        raise ValueError("candidate version already exists; refusing overwrite")
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(directory),
        path_in_repo=prefix,
        commit_message=f"Add experimental candidate {manifest['candidate_id']}",
        allow_patterns=sorted(FILES),
        parent_commit=info.sha,
    )
    revision = commit.oid
    uploaded = api.repo_info(repo_id, revision=revision, repo_type="model", files_metadata=True)
    if uploaded.private is not True:
        raise ValueError("repository privacy verification failed after upload")
    siblings = {item.rfilename: item for item in uploaded.siblings or []}
    verified = {}
    for name in sorted(FILES):
        local = directory / name
        remote = siblings.get(f"{prefix}/{name}")
        if remote is None or remote.size != local.stat().st_size:
            raise ValueError(f"uploaded file missing or size mismatch: {name}")
        expected = sha256(local)
        lfs = remote.lfs
        lfs_hash = (
            (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None))
            if lfs
            else None
        )
        if lfs_hash:
            if lfs_hash != expected:
                raise ValueError(f"uploaded LFS checksum mismatch: {name}")
        else:
            downloaded = Path(
                hf_hub_download(
                    repo_id, f"{prefix}/{name}", revision=revision, repo_type="model", token=True
                )
            )
            if sha256(downloaded) != expected:
                raise ValueError(f"uploaded checksum mismatch: {name}")
        verified[name] = expected
    return {
        "repo_id": repo_id,
        "private": True,
        "revision": revision,
        "path": prefix,
        "quality_status": "experimental_unverified",
        "verified_files": verified,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "config", "tokenizer", "provenance", "evaluation"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--output", "--package", dest="package", type=Path, required=True)
    result.add_argument("--candidate-id", required=True)
    result.add_argument("--weights-variant", choices=("raw", "ema"))
    result.add_argument("--source-checkpoint-sha256")
    result.add_argument("--repo-id")
    result.add_argument("--publish", action="store_true")
    result.add_argument(
        "--dry-run", action="store_true", help="package only; never contacts Hugging Face"
    )
    return result


def main() -> int:
    args = parser().parse_args()
    if args.publish and (args.dry_run or not args.repo_id):
        raise SystemExit("--publish requires --repo-id and is incompatible with --dry-run")
    try:
        directory = package(args)
        print(f"Validated experimental package: {directory}")
        if args.publish:
            runtime_output = directory.with_name(directory.name + ".runtime.wav")
            runtime_smoke(directory, runtime_output)
            receipt = publish(directory, args.repo_id)
            receipt["runtime_report"] = str(runtime_output.with_suffix(".json"))
            receipt["runtime_report_sha256"] = sha256(runtime_output.with_suffix(".json"))
            receipt_path = directory.with_name(directory.name + ".publication.json")
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            print(f"Private publication verified: {receipt['repo_id']}@{receipt['revision']}")
        return 0
    except Exception as error:
        # SDK exceptions may include request headers or user metadata; do not print them.
        print(
            f"Publication stopped ({type(error).__name__}); no verified publication receipt was created."
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
