"""Run a frozen full corpus, then a fresh bounded teacher adaptation.

Run inside a systemd service with KillMode=control-group and RuntimeMaxSec at
most the external VM watchdog. Nested corpus/trainer processes create sessions;
process-group cancellation here is graceful best effort, not cgroup containment.
Only upstream corpus stages resume. Any existing job directory is refused: use
another directory after interruption, retaining the same corpus output.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from types import FrameType

import yaml

from training.scripts.run_experiment import ROOT, checkpoint_pair, digest, write_json

PROMPTS = [
    "你好，今天天氣真好。",
    "我今天想去台北。",
    "我今天不想去台北。",
    "請給我一杯水。",
    "請給我兩杯水。",
    "這個東西多少錢？",
    "我們明天早上再見。",
    "阿母正在厝裡煮飯。",
    "火車幾點會到？",
    "我想去台南看朋友。",
    "我想去高雄看朋友。",
    "謝謝你幫我的忙。",
]


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def remaining(deadline: float, reserve: float = 0) -> float:
    seconds = deadline - time.monotonic() - reserve
    if seconds <= 0:
        raise TimeoutError("job deadline leaves no time for this stage")
    return seconds


def command(argv: list[str], seconds: float, log: Path) -> int:
    """Wait for graceful exit before escalation; systemd owns nested sessions."""
    child = None
    previous = {}

    def interrupted(signum: int, frame: FrameType | None) -> None:
        raise InterruptedError(f"job received signal {signum}")

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, interrupted)
        with log.open("ab") as stream:
            child = subprocess.Popen(
                argv,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env={**os.environ, "WANDB_MODE": "offline", "PYTHONUNBUFFERED": "1"},
                start_new_session=True,
            )
            return child.wait(timeout=seconds)
    finally:
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if child.poll() is None:
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def completed_releases(inventory: Path, output: Path) -> tuple[list[Path], dict[str, int]]:
    from training.taigi.corpus import normalize_inventory, worker_artifacts

    plan = json.loads((output / "plan.json").read_text())
    state = json.loads((output / "state.json").read_text())
    queue = normalize_inventory(inventory)
    if state.get("reason") or state.get("active"):
        raise ValueError("corpus stopped with an active stage or unresolved failure reason")
    if plan["inventory_sha256"] != digest(inventory) or plan["queue"] != queue:
        raise ValueError("corpus does not match frozen full inventory")
    if set(state["videos"]) != {item["key"] for item in queue}:
        raise ValueError("corpus queue is incomplete")
    counts = {name: 0 for name in ("complete", "no_candidates", "failed")}
    releases = []
    for item in queue:
        entry = state["videos"][item["key"]]
        if entry["state"] not in counts:
            raise ValueError("partial corpus queue: every video must have a terminal outcome")
        counts[entry["state"]] += 1
        if entry["state"] == "complete":
            video = output / "videos" / item["key"]
            worker_artifacts(video, "transcribe")
            result = json.loads((video / "stage/result.json").read_text())
            if result["state"] != "complete":
                raise ValueError("complete queue entry disagrees with worker result")
            releases.append(Path(result["release_path"]))
    if not releases:
        raise ValueError("corpus has no successful releases")
    return releases, counts


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("INVOCATION_ID"):
        raise ValueError("requires systemd service with KillMode=control-group and RuntimeMaxSec")
    for name in ("max_seconds", "training_seconds", "backup_seconds", "diagnostic_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if min(args.vocab_size, args.max_steps, args.warmup_steps, args.valid_cap) < 1:
        raise ValueError("vocabulary, steps, warmup and validation cap must be positive")
    deadline = time.monotonic() + args.max_seconds
    job = args.job.resolve()
    job.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"state": "running", "stage": "corpus", "started_at": time.time()}
    report_path = job / "job.json"
    write_json(report_path, report)

    def stage(name: str, argv: list[str], limit: float, reserve: float = 0) -> int:
        report["stage"] = name
        write_json(report_path, report)
        return command(argv, min(limit, remaining(deadline, reserve)), job / f"{name}.log")

    python = sys.executable
    try:
        # Corpus identity includes its config. Reuse the exact frozen settings.
        plan = json.loads((args.corpus / "plan.json").read_text())
        corpus_argv = [
            python,
            "-m",
            "training.taigi.corpus",
            "run",
            str(args.inventory),
            str(args.corpus),
        ]
        for key, value in plan["config"].items():
            if value is not None:
                corpus_argv.extend(["--" + key.replace("_", "-"), str(value)])
        reserve = args.training_seconds + 2 * args.diagnostic_seconds + args.backup_seconds
        corpus_seconds = remaining(deadline, reserve)
        if corpus_seconds <= 60:
            raise TimeoutError("insufficient corpus cancellation margin")
        corpus_argv.extend(["--max-seconds", str(corpus_seconds - 60)])
        corpus_code = stage("corpus", corpus_argv, corpus_seconds, reserve)
        # The corpus CLI exits 1 for explicit failed sources as well as partial queues.
        # Only the complete frozen queue check below may admit that outcome.
        if corpus_code not in (0, 1):
            raise RuntimeError("corpus subprocess failed")
        releases, counts = completed_releases(args.inventory, args.corpus)
        report["video_outcomes"] = counts
        report["inventory_sha256"] = digest(args.inventory)
        from training.taigi.merge import merge_releases

        release = merge_releases(
            releases, job / "releases", valid_groups=set(args.valid_group) or None
        )
        train, valid = release / "train.jsonl", release / "valid.jsonl"
        train_rows, valid_rows = rows(train), rows(valid)
        if not train_rows or not valid_rows:
            raise ValueError(
                "training requires nonempty source-separated train and validation sets"
            )
        report["dataset"] = {
            "release": str(release),
            "train_sha256": digest(train),
            "valid_sha256": digest(valid),
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
        }
        prefix = job / "tokenizer"
        argv = [
            python,
            "-m",
            "training.scripts.train_tokenizer",
            str(prefix),
            str(train),
            "--vocab-size",
            str(args.vocab_size),
        ]
        if stage("tokenizer", argv, remaining(deadline, args.backup_seconds)) != 0:
            raise RuntimeError("tokenizer training failed; requested vocabulary was not adjusted")
        parity = job / "parity.jsonl"
        parity.write_text(train.read_text() + valid.read_text())
        tokenizer = job / "tokenizer.json"
        if (
            stage(
                "conversion",
                [
                    python,
                    "-m",
                    "training.scripts.convert_tokenizer",
                    str(prefix.with_suffix(".model")),
                    "--out",
                    str(tokenizer),
                    "--texts",
                    str(parity),
                ],
                remaining(deadline, args.backup_seconds),
            )
            != 0
        ):
            raise RuntimeError("tokenizer parity conversion failed")
        vocab = json.loads(tokenizer.read_text())["model"]["vocab"]
        actual_vocab = len(vocab)
        report["tokenizer"] = {
            "training_input": str(train),
            "training_sha256": digest(train),
            "requested_vocab": args.vocab_size,
            "actual_vocab": actual_vocab,
            "sha256": digest(tokenizer),
        }
        config = yaml.safe_load((ROOT / "training/configs/taigi_smoke.yaml").read_text())
        config["model_overrides"] = {
            "flow_lm.lookup_table.tokenizer_path": str(tokenizer),
            "flow_lm.lookup_table.n_bins": actual_vocab,
        }
        config["data"].update(train_jsonl=str(train), valid_jsonl=str(valid))
        config["optim"].update(lr=1e-4, warmup_steps=args.warmup_steps)
        config.update(
            run_dir=str(job / "training"),
            batch_size=1,
            grad_accum_steps=8,
            max_steps=args.max_steps,
            valid_freq=100,
            num_valid_batches=min(len(valid_rows), args.valid_cap),
            ckpt_freq=100,
            num_ckpt_keep=2,
            sample_sentences=[],
            freeze_head=False,
            sample_cfg_coef=1.0,
        )
        report["validation"] = {
            "rows": len(valid_rows),
            "batch_size": 1,
            "coverage_cap": args.valid_cap,
            "batches": config["num_valid_batches"],
        }
        config_path = job / "training.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        report["config_sha256"] = digest(config_path)
        prompts = job / "prompts.jsonl"
        prompts.write_text(
            "".join(
                json.dumps({"id": f"fixed-{i:02d}", "text": text}, ensure_ascii=False) + "\n"
                for i, text in enumerate(PROMPTS)
            )
        )
        evaluation = [python, "-m", "training.taigi.evaluate", str(config_path), str(prompts)]
        if (
            stage(
                "baseline",
                [*evaluation, str(job / "baseline")],
                args.diagnostic_seconds,
                args.backup_seconds + args.training_seconds + args.diagnostic_seconds,
            )
            != 0
        ):
            raise RuntimeError("baseline generation failed")
        train_seconds = min(
            args.training_seconds,
            remaining(deadline, args.backup_seconds + args.diagnostic_seconds + 90),
        )
        train_code = stage(
            "training",
            [
                python,
                "-m",
                "training.scripts.run_experiment",
                "run",
                str(config_path),
                "--max-seconds",
                str(train_seconds),
                "--grace-seconds",
                "30",
                "--wandb",
                "offline",
                "--wandb-finish-seconds",
                "10",
                "--provenance",
                str(tokenizer),
            ],
            train_seconds + 60,
            args.backup_seconds + args.diagnostic_seconds,
        )
        report["training_exit_code"] = train_code
        if train_code not in (0, 124):
            raise RuntimeError("training failed")
        pair = checkpoint_pair(job / "training")
        if pair is None:
            raise RuntimeError("training produced no complete checkpoint")
        if (
            stage(
                "final",
                [*evaluation, str(job / "final"), "--checkpoint", str(pair[0])],
                args.diagnostic_seconds,
                args.backup_seconds,
            )
            != 0
        ):
            raise RuntimeError("final generation failed")
        report.update(
            state="complete",
            checkpoint=str(pair[0]),
            checkpoint_sha256=digest(pair[0]),
            training_bounded=train_code == 124,
        )
        return report
    except BaseException as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["ended_at"] = time.time()
        write_json(report_path, report)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("inventory", type=Path)
    result.add_argument("corpus", type=Path)
    result.add_argument("job", type=Path)
    result.add_argument("--valid-group", action="append", default=[])
    result.add_argument("--max-seconds", type=float, default=19 * 3600)
    result.add_argument("--training-seconds", type=float, default=4 * 3600)
    result.add_argument("--backup-seconds", type=float, default=600)
    result.add_argument("--diagnostic-seconds", type=float, default=600)
    result.add_argument("--vocab-size", type=int, default=4096)
    result.add_argument("--max-steps", type=int, default=2000)
    result.add_argument("--warmup-steps", type=int, default=100)
    result.add_argument("--valid-cap", type=int, default=100)
    return result


def main() -> int:
    try:
        run(parser().parse_args())
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        print(f"Corpus job stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
