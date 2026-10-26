"""Owner-authorized local 1000/3000/6000-update exploration, never quality promotion.

Run in a persistent user systemd service with KillMode=control-group, Restart=no,
OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1. The separately bounded first1000 service
must finish before this controller admits any model process.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from training.scripts import continue_taigi_training as continuation
from training.scripts.run_experiment import ROOT, digest, run_lock, write_json
from training.scripts.run_taigi_corpus_job import command
from training.scripts.watch_taigi_cloud_job import POLL_SCRIPT, ended_invocation

INITIAL_BASELINE_LOSS = 0.5278603


def source_record() -> dict[str, Any]:
    return {
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10
        ).strip(),
        "files": {
            name: digest(ROOT / name)
            for name in (
                "training/scripts/extend_taigi_training.py",
                "training/scripts/continue_taigi_training.py",
                "training/scripts/run_experiment.py",
                "training/scripts/run_taigi_corpus_job.py",
                "training/scripts/watch_taigi_cloud_job.py",
                "training/scripts/publish_taigi_model.py",
                "training/taigi/evaluate.py",
                "training/checkpointing.py",
                "training/modules/builders.py",
            )
        },
    }


def durable_first_completion(args: argparse.Namespace) -> tuple[dict[str, str], tuple[Path, Path]]:
    """Recover completed user units whose volatile systemd history was collected."""
    config, identity = continuation.verify_parent(args.first_job / "training.yaml")
    run_dir = Path(config["run_dir"]).resolve()
    if run_dir != (args.first_job / "training").resolve():
        raise ValueError("first service run directory mismatch")
    with run_lock(run_dir):
        attempts = []
        for path in (run_dir / "attempts").glob("*.json"):
            record = continuation.read(path)
            started = record.get("started_at")
            if (
                isinstance(started, bool)
                or not isinstance(started, (int, float))
                or not math.isfinite(started)
                or started <= 0
            ):
                raise ValueError("invalid first service attempt timestamp")
            attempts.append((started, path, record))
        if not attempts:
            raise ValueError("collected first service has no durable attempt receipt")
        attempts.sort(key=lambda item: item[0])
        started, path, record = attempts[-1]
        if len(attempts) > 1 and attempts[-2][0] == started:
            raise ValueError("ambiguous latest first service attempt")
        ended = record.get("ended_at")
        if (
            record.get("state") != "succeeded"
            or type(record.get("exit_code")) is not int
            or record["exit_code"] != 0
            or type(record.get("child_exit_code")) is not int
            or record["child_exit_code"] != 0
            or isinstance(ended, bool)
            or not isinstance(ended, (int, float))
            or not math.isfinite(ended)
            or not started <= ended <= time.time()
            or record.get("identity") != identity
        ):
            raise ValueError("latest first service attempt is not verified successful")
        verified = validate_first(args.parent_200 / "training.yaml", args.first_job)
        return {"path": str(path), "sha256": digest(path)}, verified


def wait_first(
    args: argparse.Namespace, report: dict[str, Any], path: Path
) -> tuple[Path, Path] | None:
    invocation = None
    script = POLL_SCRIPT.replace('["systemctl", "show",', '["systemctl", "--user", "show",')
    while True:
        seconds = min(
            30, continuation.remaining(args.absolute_deadline_epoch, args.reserve_seconds)
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", script, args.first_unit, str(args.first_job / "job.json")],
                capture_output=True,
                text=True,
                check=True,
                timeout=seconds,
            )
            status = json.loads(result.stdout)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            report["last_poll_error"] = type(error).__name__
        else:
            service = status["service"]
            current = service.get("InvocationID")
            if current and invocation and current != invocation:
                raise ValueError("first service invocation changed")
            invocation = current or invocation
            report["first_service_status"] = status
            if ended_invocation(status):
                if service.get("Result") != "success":
                    raise ValueError(
                        "first1000 service did not finish successfully; partial checkpoints retained"
                    )
                # run_experiment has no job.json; the exact checkpoint/optimizer
                # and lineage checks after this wait establish its completion.
                write_json(path, report)
                return
            if (
                service.get("LoadState") == "loaded"
                and service.get("ActiveState") == "inactive"
                and service.get("Restart") == "no"
                and service.get("Result") == "success"
                and service.get("MainPID") == "0"
                and service.get("ControlPID") == "0"
                and service.get("CGroupProcessesCount") == "0"
                and service.get("ExecMainStartTimestampMonotonic") == "0"
                and not current
            ):
                receipt, verified = durable_first_completion(args)
                report["first_completion_receipt"] = receipt
                write_json(path, report)
                return verified
        write_json(path, report)
        time.sleep(
            min(
                args.poll_seconds,
                continuation.remaining(args.absolute_deadline_epoch, args.reserve_seconds),
            )
        )


def validate_first(parent_config: Path, first_job: Path) -> tuple[Path, Path]:
    parent, parent_identity = continuation.verify_parent(parent_config)
    parent_checkpoint, parent_optimizer, parent_step = continuation.verify_pair(
        Path(parent["run_dir"])
    )
    if parent_step != 200:
        raise ValueError("source parent must be exactly200 updates")
    first_config = first_job / "training.yaml"
    first, _ = continuation.verify_parent(first_config)
    ignored = {"run_dir", "max_steps"}
    if {k: v for k, v in parent.items() if k not in ignored} != {
        k: v for k, v in first.items() if k not in ignored
    }:
        raise ValueError("first1000 changed parent training hyperparameters or data")
    lineage = continuation.read(first_job / "lineage.json")
    if (
        lineage.get("parent_step") != 200
        or lineage.get("endpoint") != 1000
        or lineage.get("parent_config", {}).get("sha256") != digest(parent_config)
        or lineage.get("parent_identity") != parent_identity
        or first.get("max_steps") != 1000
        or lineage.get("checkpoint", {}).get("sha256") != digest(parent_checkpoint)
        or lineage.get("optimizer", {}).get("sha256") != digest(parent_optimizer)
        or lineage.get("checkpoint", {}).get("path") != str(parent_checkpoint.resolve())
        or lineage.get("optimizer", {}).get("path") != str(parent_optimizer.resolve())
    ):
        raise ValueError("first1000 source parent chain mismatch")
    checkpoint, _, step = continuation.verify_pair(Path(first["run_dir"]))
    if step != 1000:
        raise ValueError(
            "first service did not complete exactly1000 updates; partial checkpoints retained"
        )
    return first_config, checkpoint


def evaluate(
    config: Path, checkpoint: Path, prompts: Path, output: Path, step: int, args: argparse.Namespace
) -> dict[str, Any]:
    seconds = min(
        args.diagnostic_seconds,
        continuation.remaining(args.absolute_deadline_epoch, args.reserve_seconds),
    )
    code = command(
        [
            sys.executable,
            "-m",
            "training.taigi.evaluate",
            str(config),
            str(prompts),
            str(output),
            "--checkpoint",
            str(checkpoint),
            "--asr",
        ],
        seconds,
        output.parent / "evaluation.log",
    )
    if code:
        raise RuntimeError("longer-training fixed-prompt evaluation failed")
    cfg, _ = continuation.verify_parent(config)
    return continuation.summarize(
        output / "report.json",
        continuation.progress(Path(cfg["run_dir"]) / "progress.jsonl"),
        step,
        config,
        checkpoint,
        prompts,
    )


def complete_diagnostics(record: dict[str, Any]) -> bool:
    scores = record.get("similarities", [])
    return (
        record.get("finite") is True
        and type(record.get("loss")) in (int, float)
        and math.isfinite(record["loss"])
        and len(scores) == 12
        and all(
            type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
            for value in scores
        )
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.owner_authorized_longer_training:
        raise ValueError("explicit --owner-authorized-longer-training is required")
    if not os.environ.get("INVOCATION_ID"):
        raise ValueError("requires persistent user systemd service with cgroup containment")
    for name in (
        "absolute_deadline_epoch",
        "reserve_seconds",
        "diagnostic_seconds",
        "poll_seconds",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"invalid {name}")
    if args.reserve_seconds < 1800 or not re.fullmatch(
        r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", args.first_unit
    ):
        raise ValueError("requires1800-second reserve and explicit service unit")
    today22 = (
        datetime.now(ZoneInfo("Asia/Taipei"))
        .replace(hour=22, minute=0, second=0, microsecond=0)
        .timestamp()
    )
    if args.absolute_deadline_epoch > today22:
        raise ValueError("deadline must not exceed22:00 Asia/Taipei today")
    continuation.remaining(args.absolute_deadline_epoch, args.reserve_seconds)
    args.parent_200, args.first_job, args.output = (
        args.parent_200.resolve(),
        args.first_job.resolve(),
        args.output.resolve(),
    )
    if (
        args.output in (args.parent_200, args.first_job)
        or args.output.is_relative_to(args.first_job)
        or args.output.is_relative_to(args.parent_200)
    ):
        raise ValueError("output must be a fresh separate directory")
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "job.json"
    report: dict[str, Any] = {
        "state": "running",
        "stage": "verify_parent",
        "quality_verified": False,
        "owner_authorized_longer_training": True,
        "started_at": time.time(),
        "deadline": args.absolute_deadline_epoch,
        "initial_baseline_loss": INITIAL_BASELINE_LOSS,
        "divergence_limit": 1.5 * INITIAL_BASELINE_LOSS,
        "divergence_check": "evaluated stage boundaries only; not continuous",
        "candidates": [],
        "gates": [],
    }
    candidates: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    baseline = None
    source = None
    failure = None
    first_ended = False
    unsafe_stop = False
    write_json(path, report)
    try:
        source = source_record()
        report["source_code"] = source
        parent_config = args.parent_200 / "training.yaml"
        parent, _ = continuation.verify_parent(parent_config)
        parent_checkpoint, _, parent_step = continuation.verify_pair(Path(parent["run_dir"]))
        if parent_step != 200:
            raise ValueError("frozen initial parent must be exactly200 updates")
        prompts = args.parent_200 / "prompts.jsonl"
        prompt_hash = digest(prompts)
        records = continuation.progress(Path(parent["run_dir"]) / "progress.jsonl")
        baseline = continuation.summarize(
            args.parent_200 / "baseline/report.json", records, 0, parent_config, None, prompts
        )
        if not complete_diagnostics(baseline):
            raise ValueError("baseline diagnostics missing or nonfinite")
        if not math.isclose(baseline["loss"], INITIAL_BASELINE_LOSS, rel_tol=0, abs_tol=1e-6):
            raise ValueError(
                "initial baseline does not match authorized longer-training experiment"
            )
        initial = continuation.summarize(
            args.parent_200 / "final/report.json",
            records,
            200,
            parent_config,
            parent_checkpoint,
            prompts,
        )
        if not complete_diagnostics(initial):
            raise ValueError("initial parent diagnostics missing or nonfinite")
        candidates.append(initial)
        if initial["loss"] <= report["divergence_limit"]:
            eligible.append(initial)
        report.update(
            stage="waiting_first1000",
            baseline=baseline,
            candidates=candidates,
            prompt_sha256=prompt_hash,
        )
        write_json(path, report)
        verified_first = wait_first(args, report, path)
        first_ended = True
        config, checkpoint = verified_first or validate_first(parent_config, args.first_job)
        for endpoint in (1000, 3000, 6000):
            if digest(prompts) != prompt_hash:
                raise ValueError("fixed prompts changed")
            report.update(stage="evaluation", active_step=endpoint)
            write_json(path, report)
            child = args.first_job if endpoint == 1000 else args.output / f"step-{endpoint}"
            candidate = evaluate(config, checkpoint, prompts, child / "evaluation", endpoint, args)
            if not complete_diagnostics(candidate):
                raise ValueError("missing_or_invalid_consistency_diagnostics")
            candidates.append(candidate)
            report["candidates"] = candidates
            if candidate["loss"] > report["divergence_limit"]:
                report["halt_reason"] = "validation_diverged_above1.5_times_initial_baseline"
                break
            eligible.append(candidate)
            if endpoint == 6000:
                report["halt_reason"] = "completed6000_update_ceiling"
                break
            next_step = 3000 if endpoint == 1000 else 6000
            available = (
                args.absolute_deadline_epoch
                - time.time()
                - args.reserve_seconds
                - args.diagnostic_seconds
                - 120
            )
            if available <= 0:
                report["halt_reason"] = "deadline_reserve_no_new_training"
                break
            report["gates"].append(
                {
                    "from_step": endpoint,
                    "endpoint": next_step,
                    "admit": True,
                    "reason": "owner_authorized_longer_exploration_finite_below_divergence_limit",
                }
            )
            child = args.output / f"step-{next_step}"
            config = continuation.fork_run(config, child, next_step)
            child_config, _ = continuation.verify_parent(config)
            report.update(stage="training", active_step=next_step, active_child=str(child))
            write_json(path, report)
            code = command(
                [
                    sys.executable,
                    "-m",
                    "training.scripts.run_experiment",
                    "run",
                    str(config),
                    "--max-seconds",
                    str(available),
                    "--grace-seconds",
                    "60",
                    "--wandb",
                    "offline",
                    "--wandb-finish-seconds",
                    "10",
                    "--provenance",
                    child_config["model_overrides"]["flow_lm.lookup_table.tokenizer_path"],
                    "--provenance",
                    str(child / "lineage.json"),
                ],
                available + 90,
                child / "training.log",
            )
            report["gates"][-1]["training_exit_code"] = code
            if code == 124:
                report["halt_reason"] = "bounded_training_timeout_partial_child_retained"
                break
            if code != 0:
                raise RuntimeError("longer-training trainer failed")
            checkpoint, _, actual = continuation.verify_pair(Path(child_config["run_dir"]))
            if actual != next_step:
                raise ValueError("longer-training child did not reach declared endpoint")
    except TimeoutError:
        report["halt_reason"] = "absolute_deadline_reserve"
        if not any(c["step"] >= 1000 for c in candidates):
            failure = "first1000_not_verified_complete"
        unsafe_stop = not first_ended
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        unsafe_stop = not first_ended or isinstance(
            error, (InterruptedError, SystemExit, KeyboardInterrupt, subprocess.TimeoutExpired)
        )
    if unsafe_stop:
        report.update(
            state="failed",
            stage="failed",
            ended_at=time.time(),
            error=failure or "unsafe_interruption",
            cleanup_boundary="exit service for KillMode=control-group cleanup; no further model/export/publication",
        )
        write_json(path, report)
        return report
    try:
        if eligible and baseline is not None:
            report["stage"] = "export"
            write_json(path, report)
            chosen = continuation.select_candidate(eligible, baseline)
            request = args.output / "export-request.json"
            write_json(
                request,
                {
                    "selection": chosen,
                    "baseline": baseline,
                    "gates": report["gates"],
                    "source": source,
                },
            )
            script = (
                "import json,sys;from pathlib import Path;"
                "from training.scripts.continue_taigi_training import export_selection;"
                "r=json.loads(Path(sys.argv[1]).read_text());"
                'export_selection(r["selection"],r["baseline"],Path(sys.argv[2]),r["gates"],r["source"])'
            )
            seconds = min(900, continuation.remaining(args.absolute_deadline_epoch, 60))
            code = command(
                [sys.executable, "-c", script, str(request), str(args.output)],
                seconds,
                args.output / "export.log",
            )
            if code or not (args.output / "selected.json").is_file():
                raise RuntimeError("bounded selected export failed")
            report.update(selected_step=chosen["step"], selected=str(args.output / "selected.json"))
        else:
            failure = failure or "no_finite_nondivergent_evaluated_candidate"
    except BaseException as error:
        failure = failure or f"{type(error).__name__}: {error}"
    if not failure and args.publish_repo and report.get("selected"):
        report["stage"] = "publication"
        write_json(path, report)
        try:
            selected = continuation.read(Path(report["selected"]))
            if selected.get("state") != "complete" or set(selected.get("files", {})) != {
                "weights",
                "config",
                "tokenizer",
                "provenance",
                "evaluation",
            }:
                raise ValueError("selected inference contract incomplete")
            argv = [sys.executable, "-m", "training.scripts.publish_taigi_model"]
            for key, item in selected["files"].items():
                artifact = Path(item["path"]).resolve()
                if not artifact.is_relative_to(args.output) or digest(artifact) != item["sha256"]:
                    raise ValueError("selected inference artifact changed")
                argv.extend(["--" + key, str(artifact)])
            package = args.output / "package"
            argv.extend(
                [
                    "--output",
                    str(package),
                    "--candidate-id",
                    f"{args.output.name}-{selected['candidate_id']}",
                    "--repo-id",
                    args.publish_repo,
                    "--publish",
                ]
            )
            code = command(
                argv,
                min(1200, continuation.remaining(args.absolute_deadline_epoch, 30)),
                args.output / "publish.log",
            )
            receipt = package.with_name(package.name + ".publication.json")
            published = continuation.read(receipt) if receipt.is_file() else {}
            if (
                code
                or published.get("private") is not True
                or published.get("repo_id") != args.publish_repo
            ):
                raise RuntimeError("private publication not verified")
            report["publication"] = {
                "state": "complete",
                "receipt": str(receipt),
                "sha256": digest(receipt),
            }
        except BaseException as error:
            report["publication"] = {
                "state": "failed",
                "error_type": type(error).__name__,
                "meaning": "local training and selected export remain preserved",
            }
    report.update(
        state="failed" if failure else "complete",
        stage="failed" if failure else "complete",
        ended_at=time.time(),
    )
    if failure:
        report["error"] = failure
    write_json(path, report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("first_job", type=Path)
    result.add_argument("parent_200", type=Path)
    result.add_argument("output", type=Path)
    result.add_argument("--owner-authorized-longer-training", action="store_true")
    result.add_argument("--first-unit", default="taigi-local-long-first.service")
    result.add_argument("--absolute-deadline-epoch", type=float, required=True)
    result.add_argument("--reserve-seconds", type=float, default=1800)
    result.add_argument("--diagnostic-seconds", type=float, default=600)
    result.add_argument("--poll-seconds", type=float, default=30)
    result.add_argument(
        "--publish-repo", help="Optional private personal HF publication after selected export"
    )
    return result


def main() -> int:
    try:
        report = run(parser().parse_args())
        return 0 if report["state"] == "complete" else 1
    except (ValueError, OSError, TimeoutError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
