"""Wait for a frozen corpus pilot, then gate bounded raw-weight continuations.

Run in a persistent systemd unit with Restart=no and KillMode=control-group.
The external backup watcher owns VM stopping and private publication.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from training.scripts.run_experiment import (
    ROOT,
    checkpoint_pair,
    digest,
    identity,
    run_lock,
    write_json,
)
from training.scripts.run_taigi_corpus_job import command
from training.scripts.watch_taigi_cloud_job import POLL_SCRIPT, ended_invocation
from training.taigi.pipeline import transcript_quality_reason
from training.taigi.transcriber import BREEZE_MODEL, BREEZE_REVISION, BREEZE_ADAPTER_VERSION


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def remaining(deadline: float, reserve: float = 0) -> float:
    value = deadline - time.time() - reserve
    if value <= 0:
        raise TimeoutError("absolute deadline reserve reached")
    return value


def file_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": digest(path)}


def source_identity(deployment: Path) -> dict[str, Any]:
    receipt = read(deployment)
    script_hash = digest(Path(__file__))
    if (
        not re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("source_commit", "")))
        or receipt.get("controller_sha256") != script_hash
    ):
        raise ValueError("deployment receipt does not match controller source")
    return {
        "runtime_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "runtime_git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
        ),
        "controller_sha256": script_hash,
        "inference_api_sha256": digest(ROOT / "pocket_tts/models/tts_model.py"),
        "deployment_receipt": file_record(deployment),
        "deployment": receipt,
    }


def verify_parent(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text())
    run_dir = (ROOT / config["run_dir"]).resolve()
    recorded = read(run_dir / "run_manifest.json")["identity"]
    if recorded["config_sha256"] != digest(config_path):
        raise ValueError("parent config identity changed")
    for item in recorded["inputs"].values():
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("parent input identity changed")
    for key in ("train_jsonl", "valid_jsonl"):
        expected = (ROOT / config["data"][key]).resolve()
        item = recorded["inputs"].get(key, {})
        if item.get("path") != str(expected) or item.get("sha256") != digest(expected):
            raise ValueError("dataset is absent from parent identity")
    tokenizer = (ROOT / config["model_overrides"]["flow_lm.lookup_table.tokenizer_path"]).resolve()
    if not any(
        item["path"] == str(tokenizer) and item["sha256"] == digest(tokenizer)
        for item in recorded["inputs"].values()
    ):
        raise ValueError("tokenizer is absent from parent identity")
    return config, recorded


def verify_pair(run_dir: Path) -> tuple[Path, Path, int]:
    import torch

    pair = checkpoint_pair(run_dir)
    if pair is None:
        raise ValueError("no complete parent checkpoint")
    payload = torch.load(pair[0], map_location="cpu", weights_only=True)
    optimizer = torch.load(pair[1], map_location="cpu", weights_only=True)
    step = payload.get("step")
    if type(step) is not int or step <= 0 or int(pair[0].stem.split("_")[-1]) != step:
        raise ValueError("checkpoint step mismatch")
    if not payload.get("model") or not optimizer.get("optimizer", {}).get("param_groups"):
        raise ValueError("checkpoint or optimizer state missing")

    def finite(value: object) -> None:
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite checkpoint/optimizer tensor")
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                finite(item)

    finite(payload)
    finite(optimizer)
    return pair[0], pair[1], step


def fork_run(parent_config: Path, child: Path, endpoint: int) -> Path:
    config, parent_identity = verify_parent(parent_config)
    parent_run = (ROOT / config["run_dir"]).resolve()
    with run_lock(parent_run):
        checkpoint, optimizer, step = verify_pair(parent_run)
        if endpoint <= step:
            raise ValueError("continuation endpoint must exceed parent step")
        child.mkdir(parents=True, exist_ok=False)
        run_dir = child / "training"
        run_dir.mkdir()
        lineage = {
            "parent_config": file_record(parent_config),
            "parent_identity": parent_identity,
            "checkpoint": file_record(checkpoint),
            "optimizer": file_record(optimizer),
            "parent_step": step,
            "endpoint": endpoint,
            "allowed_config_changes": ["run_dir", "max_steps"],
            "resume_semantics": "model/optimizer/EMA restored; data-loader RNG restarts by step",
        }
        for source in (checkpoint, optimizer):
            target = run_dir / source.name
            shutil.copyfile(source, target)
            if digest(target) != digest(source):
                raise ValueError("checkpoint changed during child preparation")
        lineage_path = child / "lineage.json"
        write_json(lineage_path, lineage)
        config.update(run_dir=str(run_dir), max_steps=endpoint)
        config_path = child / "training.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        tokenizer = config["model_overrides"]["flow_lm.lookup_table.tokenizer_path"]
        current = identity(config_path, config, [tokenizer, str(lineage_path)])
        write_json(run_dir / "run_manifest.json", {"schema_version": 1, "identity": current})
    return config_path


def progress(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def loss_at(records: list[dict[str, Any]], step: int) -> float:
    values = [
        r["metrics"]["loss"] for r in records if r.get("type") == "valid" and r.get("step") == step
    ]
    if not values or not math.isfinite(values[-1]):
        raise ValueError(f"missing finite fixed validation loss at step {step}")
    return float(values[-1])


def summarize(
    evaluation: Path,
    records: list[dict[str, Any]],
    step: int,
    config: Path,
    checkpoint: Path | None,
    prompts: Path,
) -> dict[str, Any]:
    report = read(evaluation)
    expected = [json.loads(line) for line in prompts.read_text().splitlines() if line.strip()]
    actual = report.get("prompts", [])
    asr = report.get("asr", {})
    if (
        asr.get("model") != BREEZE_MODEL
        or asr.get("revision") != BREEZE_REVISION
        or asr.get("adapter") != BREEZE_ADAPTER_VERSION
    ):
        raise ValueError("evaluation ASR identity mismatch")
    training_config, _ = verify_parent(config)
    tokenizer = Path(training_config["model_overrides"]["flow_lm.lookup_table.tokenizer_path"])
    if (
        report.get("state") != "complete"
        or report.get("step") != step
        or report.get("weights")
        != ("raw_checkpoint" if checkpoint else "seeded_pretrained_with_reset_text_embeddings")
        or report.get("tokenizer_sha256") != digest(tokenizer)
        or report.get("config_sha256") != digest(config)
        or report.get("checkpoint_sha256") != (digest(checkpoint) if checkpoint else None)
        or [(r["id"], r["text"]) for r in actual] != [(r["id"], r["text"]) for r in expected]
    ):
        raise ValueError("evaluation identity, completion, or prompt order mismatch")
    for row in actual:
        if row.get("status") == "audio" and not row.get("audio_path"):
            raise ValueError("healthy evaluation audio is missing")
        if row.get("audio_path") and digest(row["audio_path"]) != row["audio_sha256"]:
            raise ValueError("evaluation audio hash mismatch")
    training = [r for r in records if r.get("type") == "train"]
    finite = bool(training) or step == 0
    for row in records:
        if row.get("type") not in {"train", "valid"}:
            continue
        numbers = list(row.get("metrics", {}).values())
        if row["type"] == "train":
            numbers.append(row.get("grad_norm"))
        finite &= all(isinstance(v, (int, float)) and math.isfinite(v) for v in numbers)
    similarities: list[float] = []
    score_sources = []
    for row in actual:
        measured = row.get("character_sequence_similarity")
        frames, seconds = row.get("generated_frames"), row.get("seconds")
        if (
            row.get("status") == "audio"
            and isinstance(row.get("asr_text"), str)
            and type(measured) in (int, float)
            and math.isfinite(measured)
            and 0 <= measured <= 1
        ):
            score = float(measured)
            source, reason = "measured_asr", "finite_asr_character_sequence_similarity"
        elif (
            row.get("status") == "too_few_frames_for_decoder"
            and type(frames) is int
            and 0 <= frames < 8
            and type(seconds) in (int, float)
            and math.isfinite(seconds)
            and seconds >= 0
            and (seconds == 0) == (frames == 0)
            and not any(
                key in row
                for key in (
                    "audio_path",
                    "audio_sha256",
                    "asr_text",
                    "character_sequence_similarity",
                )
            )
        ):
            # This evaluator branch never decodes audio or calls ASR. Zero is a
            # derived task-failure score, not a fabricated ASR measurement.
            score = 0.0
            source, reason = "decoder_unavailable", "too_few_frames_for_decoder"
        else:
            score = None
            source, reason = "missing_or_invalid_diagnostic", "no_verified_task_consistency_score"
        score_sources.append({"id": row["id"], "source": source, "reason": reason, "score": score})
        if score is not None:
            similarities.append(score)
    comparable = len(actual) == len(similarities) == 12
    return {
        "step": step,
        "loss": loss_at(records, step),
        "finite": bool(finite),
        "healthy": sum(
            r.get("status") == "audio" and r.get("generated_frames", 0) >= 8 for r in actual
        ),
        "capped": sum(bool(r.get("hit_frame_limit")) for r in actual),
        "repetitive": sum(
            isinstance(r.get("asr_text"), str)
            and transcript_quality_reason(r["asr_text"]) == "repetitive_transcript"
            for r in actual
        ),
        "similarities": similarities if comparable else [],
        "failure_adjusted_task_consistency_scores": similarities if comparable else [],
        "score_sources": score_sources,
        "prompts": len(actual),
        "evaluation": str(evaluation),
        "config": str(config),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "settings": {
            k: report.get(k)
            for k in ("seed", "max_frames", "temperature", "cfg_coef", "voice_prompt_frames")
        },
    }


def gate(
    previous: dict[str, Any],
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    train_rows: int,
    effective_batch: int,
) -> tuple[bool, str]:
    if train_rows <= 0 or effective_batch <= 0:
        return False, "invalid_coverage"
    for record in (baseline, previous, candidate):
        if (
            not record["finite"]
            or not math.isfinite(record["loss"])
            or record["prompts"] != 12
            or len(record["similarities"]) != 12
            or any(not math.isfinite(v) for v in record["similarities"])
        ):
            return False, "missing_or_nonfinite_diagnostics"
    if candidate.get("settings") != baseline.get("settings"):
        return False, "generation_settings_changed"
    reference = baseline if candidate["step"] == 200 else previous
    if (
        candidate["healthy"] < reference["healthy"]
        or candidate["capped"] > reference["capped"]
        or candidate["repetitive"] > reference["repetitive"]
    ):
        return False, "generation_regressed"
    gains = [
        a - b for a, b in zip(candidate["similarities"], previous["similarities"], strict=True)
    ]
    text_gain = (
        statistics.median(candidate["similarities"]) - statistics.median(previous["similarities"])
        >= 0.05
        or sum(value >= 0.10 for value in gains) >= 3
    )
    if candidate["loss"] <= 0.99 * previous["loss"] and text_gain:
        return True, "directional_improvement"
    if (
        candidate["step"] == 200
        and 200 * effective_batch / train_rows < 1
        and candidate["loss"] <= 1.10 * baseline["loss"]
    ):
        return True, "undercoverage_bridge"
    return False, "no_joint_validation_and_text_improvement"


def select_candidate(candidates: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    candidates = [c for c in candidates if c["finite"] and math.isfinite(c["loss"])]
    if not candidates:
        raise ValueError("no finite evaluated candidate is available")
    healthy = [
        c
        for c in candidates
        if c["healthy"] >= baseline["healthy"]
        and c["capped"] <= baseline["capped"]
        and c["repetitive"] <= baseline["repetitive"]
    ]
    pool = healthy or candidates
    return min(
        pool,
        key=lambda c: (c["loss"] if healthy else -c["healthy"], c["capped"], c["loss"], c["step"]),
    )


def export_selection(
    selection: dict[str, Any],
    baseline: dict[str, Any],
    output: Path,
    gates: list[dict[str, Any]],
    source_code: dict[str, Any] | None = None,
) -> None:
    import torch
    from training.args import load_args
    from training.checkpointing import export_pocket_safetensors, load_checkpoint
    from training.modules.builders import build_models

    target = output / "selection"
    target.mkdir(exist_ok=False)
    config_path, checkpoint = Path(selection["config"]), Path(selection["checkpoint"])
    config, _ = verify_parent(config_path)
    source_hash = digest(checkpoint)
    args = load_args(str(config_path))
    torch.manual_seed(args.seed)
    model, mimi, runtime = build_models(args)
    restored = load_checkpoint(checkpoint, model)
    if restored != selection["step"] or digest(checkpoint) != source_hash:
        raise ValueError("selected checkpoint changed")
    weights = target / "model.safetensors"
    export_pocket_safetensors(weights, model.flow_lm, mimi, ema=None)
    tokenizer = target / "tokenizer.json"
    original_tokenizer = Path(config["model_overrides"]["flow_lm.lookup_table.tokenizer_path"])
    shutil.copyfile(original_tokenizer, tokenizer)
    if digest(tokenizer) != digest(original_tokenizer):
        raise ValueError("tokenizer changed during export")
    runtime_dict = runtime.model_dump(mode="json")
    runtime_dict["weights_path"] = str(weights)
    runtime_dict["weights_path_without_voice_cloning"] = str(weights)
    runtime_dict["flow_lm"]["lookup_table"]["tokenizer_path"] = str(tokenizer)
    runtime_path = target / "config.yaml"
    runtime_path.write_text(yaml.safe_dump(runtime_dict, sort_keys=False))
    evaluation = target / "evaluation.json"
    evaluation.write_text(Path(selection["evaluation"]).read_text())
    provenance = target / "provenance.json"
    write_json(
        provenance,
        {
            "weights_variant": "raw",
            "source_code": source_code,
            "source_checkpoint_sha256": source_hash,
            "exported_weights_sha256": digest(weights),
            "source_checkpoint": str(checkpoint),
            "source_config": file_record(config_path),
            "selection": selection,
            "baseline": baseline,
            "gates": gates,
            "quality_verified": False,
            "selection_rule": "minimum validation loss among nonregressing healthy raw candidates; otherwise least unhealthy diagnostic",
            "validation_improved_over_initialization": selection["loss"] < baseline["loss"],
            "inference_condition": "zero voice prompt; experimental Chinese-to-Taiwanese conditioning",
        },
    )
    files = {
        key: file_record(path)
        for key, path in {
            "weights": weights,
            "config": runtime_path,
            "tokenizer": tokenizer,
            "evaluation": evaluation,
            "provenance": provenance,
        }.items()
    }
    write_json(
        output / "selected.json",
        {
            "state": "complete",
            "quality_verified": False,
            "candidate_id": f"step-{selection['step']}-raw",
            "files": files,
        },
    )


def wait_parent(args: argparse.Namespace, report: dict[str, Any], report_path: Path) -> None:
    invocation = None
    changed = False
    while True:
        timeout = min(30, remaining(args.absolute_deadline_epoch))
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    POLL_SCRIPT,
                    args.parent_unit,
                    str(args.parent_job / "job.json"),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            status = json.loads(result.stdout)
            service = status["service"]
            current = service.get("InvocationID")
            if current:
                changed |= bool(invocation and invocation != current)
                invocation = current
            report["parent_status"] = status
            write_json(report_path, report)
        except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
            # The parent is in another cgroup. A polling failure must not make
            # our backup watcher mistake its still-changing output for final data.
            report["last_poll_error"] = f"{type(error).__name__}: {error}"
            write_json(report_path, report)
        else:
            if ended_invocation(status):
                if changed:
                    raise ValueError("parent service invocation changed")
                if (
                    service.get("Result") != "success"
                    or (status.get("job") or {}).get("state") != "complete"
                ):
                    raise ValueError("parent service/job did not complete successfully")
                return
        time.sleep(min(args.poll_seconds, remaining(args.absolute_deadline_epoch)))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("INVOCATION_ID"):
        raise ValueError("requires persistent systemd service with cgroup containment")
    for name in (
        "absolute_deadline_epoch",
        "reserve_seconds",
        "chunk_seconds",
        "diagnostic_seconds",
        "poll_seconds",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"invalid {name}")
    if args.reserve_seconds < 1800 or args.chunk_seconds > 3600:
        raise ValueError("reserve must be at least 1800 seconds; chunk at most 3600")
    args.parent_job, args.output = args.parent_job.resolve(), args.output.resolve()
    if not args.output.is_relative_to(args.parent_job.parent) or args.output == args.parent_job:
        raise ValueError("continuation output must be a separate directory within production")
    deployment = args.deployment_receipt or args.parent_job.parent / "continuation-deployment.json"
    source_code = source_identity(deployment)
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / "job.json"
    report: dict[str, Any] = {
        "state": "running",
        "stage": "waiting_parent",
        "quality_verified": False,
        "source_code": source_code,
        "started_at": time.time(),
        "deadline": args.absolute_deadline_epoch,
        "gates": [],
    }
    write_json(report_path, report)
    try:
        wait_parent(args, report, report_path)
        config_path = args.parent_job / "training.yaml"
        config, _ = verify_parent(config_path)
        parent_run = Path(config["run_dir"])
        checkpoint, _, step = verify_pair(parent_run)
        if step != 200:
            raise ValueError("initial parent must have completed exactly 200 updates")
        prompts = args.parent_job / "prompts.jsonl"
        prompt_hash = digest(prompts)
        records = progress(parent_run / "progress.jsonl")
        baseline = summarize(
            args.parent_job / "baseline/report.json", records, 0, config_path, None, prompts
        )
        candidate = summarize(
            args.parent_job / "final/report.json", records, step, config_path, checkpoint, prompts
        )
        candidates = [candidate]
        previous = baseline
        train_rows = sum(
            bool(line.strip())
            for line in Path(config["data"]["train_jsonl"]).read_text().splitlines()
        )
        effective = config["batch_size"] * config["grad_accum_steps"]
        report.update(
            stage="gating",
            baseline=baseline,
            candidates=candidates,
            train_rows=train_rows,
            effective_batch=effective,
            prompt_sha256=prompt_hash,
        )
        for endpoint in (500, 1000, 2000):
            admit, reason = gate(previous, candidate, baseline, train_rows, effective)
            decision = {
                "from_step": candidate["step"],
                "endpoint": endpoint,
                "admit": admit,
                "reason": reason,
                "approximate_passes": candidate["step"] * effective / train_rows,
            }
            report["gates"].append(decision)
            write_json(report_path, report)
            if not admit:
                break
            available = args.absolute_deadline_epoch - time.time() - args.reserve_seconds
            if available <= args.diagnostic_seconds + 120:
                decision.update(admit=False, reason="insufficient_time_reserve")
                break
            seconds = min(args.chunk_seconds, available - args.diagnostic_seconds - 120)
            if digest(prompts) != prompt_hash:
                raise ValueError("fixed prompts changed")
            child = args.output / f"step-{endpoint}"
            config_path = fork_run(Path(candidate["config"]), child, endpoint)
            child_config = yaml.safe_load(config_path.read_text())
            report.update(stage="training", active_child=str(child))
            write_json(report_path, report)
            tokenizer = child_config["model_overrides"]["flow_lm.lookup_table.tokenizer_path"]
            code = command(
                [
                    sys.executable,
                    "-m",
                    "training.scripts.run_experiment",
                    "run",
                    str(config_path),
                    "--max-seconds",
                    str(seconds),
                    "--grace-seconds",
                    "60",
                    "--wandb",
                    "offline",
                    "--wandb-finish-seconds",
                    "10",
                    "--provenance",
                    tokenizer,
                    "--provenance",
                    str(child / "lineage.json"),
                ],
                seconds + 90,
                child / "training.log",
            )
            decision["training_exit_code"] = code
            if code == 124:
                decision["reason"] = "bounded_training_timeout_prior_candidates_preserved"
                break
            if code != 0:
                raise RuntimeError(f"continuation trainer failed: {code}")
            run_dir = Path(child_config["run_dir"])
            checkpoint, _, actual_step = verify_pair(run_dir)
            if actual_step != endpoint:
                raise ValueError("continuation did not reach its declared endpoint")
            report["stage"] = "evaluation"
            write_json(report_path, report)
            seconds = min(
                args.diagnostic_seconds,
                remaining(args.absolute_deadline_epoch, args.reserve_seconds),
            )
            code = command(
                [
                    sys.executable,
                    "-m",
                    "training.taigi.evaluate",
                    str(config_path),
                    str(prompts),
                    str(child / "evaluation"),
                    "--checkpoint",
                    str(checkpoint),
                    "--asr",
                ],
                seconds,
                child / "evaluation.log",
            )
            if code:
                raise RuntimeError("continuation evaluation failed")
            previous = candidate
            candidate = summarize(
                child / "evaluation/report.json",
                progress(run_dir / "progress.jsonl"),
                endpoint,
                config_path,
                checkpoint,
                prompts,
            )
            candidates.append(candidate)
            report["candidates"] = candidates
        report["stage"] = "export"
        write_json(report_path, report)
        remaining(args.absolute_deadline_epoch, 1800)
        chosen = select_candidate(candidates, baseline)
        export_selection(chosen, baseline, args.output, report["gates"], source_code)
        report.update(
            state="complete",
            stage="complete",
            selected=str(args.output / "selected.json"),
            selected_step=chosen["step"],
            validation_improved=chosen["loss"] < baseline["loss"],
        )
    except BaseException as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["ended_at"] = time.time()
        write_json(report_path, report)
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parent_job", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--deployment-receipt", type=Path)
    p.add_argument("--parent-unit", default="taigi-full-pipeline.service")
    p.add_argument("--absolute-deadline-epoch", type=float, default=1793023200)
    p.add_argument("--reserve-seconds", type=float, default=5400)
    p.add_argument("--chunk-seconds", type=float, default=3600)
    p.add_argument("--diagnostic-seconds", type=float, default=600)
    p.add_argument("--poll-seconds", type=float, default=30)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
