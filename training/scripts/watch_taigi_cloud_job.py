"""Mirror a bounded cloud job, verify its files, and invoke an explicit stop helper.

No disk is deleted. Run this controller durably on the local machine; its own
lifecycle is separate from the remote service and VM watchdog. A missing helper
means observation and backup only, and is recorded as stop_not_configured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

POLL_SCRIPT = r"""
import json, subprocess, sys
from pathlib import Path
unit, job = sys.argv[1:]
result = subprocess.run(["systemctl", "show", unit, "--property=ActiveState,SubState"],
                        capture_output=True, text=True, timeout=15, check=True)
properties = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
try:
    record = json.loads(Path(job).read_text())
except (OSError, ValueError):
    record = None
print(json.dumps({"service": properties, "job": record}))
"""
HASH_SCRIPT = r"""
import hashlib, json, os, stat, sys
from pathlib import Path
root = Path(sys.argv[1])
if not root.is_dir():
    raise ValueError("remote output directory missing")
for directory, dirs, names in os.walk(root, followlinks=False):
    dirs.sort()
    for name in sorted(names):
        path = Path(directory) / name
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            continue
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        after = path.lstat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("remote file changed while hashing")
        print(json.dumps({"path": str(path.relative_to(root)), "size": after.st_size,
                          "sha256": h.hexdigest()}), flush=True)
"""


def atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def time_left(args: argparse.Namespace) -> float:
    remaining = args.absolute_deadline_epoch - time.time()
    if remaining <= 0:
        raise TimeoutError("absolute cloud deadline reached")
    return min(args.command_timeout, remaining)


def ssh(args: argparse.Namespace, script: str, arguments: list[str]) -> list[str]:
    remote = shlex.join(["python3", "-c", script, *arguments])
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", args.host, remote]


def poll(args: argparse.Namespace) -> dict[str, Any]:
    result = subprocess.run(
        ssh(args, POLL_SCRIPT, [args.remote_systemd_unit, args.remote_job_json]),
        check=True,
        capture_output=True,
        text=True,
        timeout=time_left(args),
    )
    value = json.loads(result.stdout)
    if not isinstance(value, dict) or not isinstance(value.get("service"), dict):
        raise ValueError("invalid remote service response")
    return value


def mirror(args: argparse.Namespace) -> None:
    subprocess.run(
        [
            "rsync",
            "-a",
            "--partial",
            "--protect-args",
            "-e",
            "ssh -o BatchMode=yes -o ConnectTimeout=10",
            "--",
            f"{args.host}:{args.remote_output.rstrip('/')}/",
            str(args.local_output) + "/",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=time_left(args),
    )


def verify(args: argparse.Namespace, manifest: Path) -> int:
    temporary = manifest.with_suffix(".partial")
    try:
        with temporary.open("wb") as stream:
            subprocess.run(
                ssh(args, HASH_SCRIPT, [args.remote_output]),
                check=True,
                stdout=stream,
                stderr=subprocess.PIPE,
                timeout=time_left(args),
            )
        count = 0
        root = args.local_output.resolve()
        with temporary.open() as stream:
            for line in stream:
                time_left(args)
                item = json.loads(line)
                relative = Path(item["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("remote hash manifest contains an unsafe path")
                target = root / relative
                if target.is_symlink() or not target.resolve().is_relative_to(root):
                    raise ValueError(
                        f"local mirror path is a symlink or escapes output: {relative}"
                    )
                if not target.is_file() or target.stat().st_size != item["size"]:
                    raise ValueError(f"local mirror file missing or size mismatch: {relative}")
                h = hashlib.sha256()
                with target.open("rb") as local:
                    for block in iter(lambda: local.read(1024 * 1024), b""):
                        time_left(args)
                        h.update(block)
                if h.hexdigest() != item["sha256"]:
                    raise ValueError(f"local mirror checksum mismatch: {relative}")
                count += 1
        if not count:
            raise ValueError("remote output contains no regular files")
        temporary.replace(manifest)
        return count
    finally:
        temporary.unlink(missing_ok=True)


def stop(args: argparse.Namespace, receipt: dict[str, Any], path: Path) -> None:
    receipt["stop"] = "requested" if args.stop_command else "stop_not_configured"
    atomic(path, receipt)
    if args.stop_command:
        try:
            result = subprocess.run(
                [str(args.stop_command), "--stop-now"],
                check=False,
                capture_output=True,
                timeout=args.stop_timeout,
            )
            receipt["stop"] = "succeeded" if result.returncode == 0 else "failed"
            receipt["stop_exit_code"] = result.returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            receipt.update(stop="failed", stop_error=type(error).__name__)
        atomic(path, receipt)


def run(args: argparse.Namespace) -> dict[str, Any]:
    for name in ("absolute_deadline_epoch", "interval", "command_timeout", "stop_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", args.host):
        raise ValueError("host must be an explicit SSH hostname or user@hostname")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", args.remote_systemd_unit):
        raise ValueError("invalid systemd unit")
    for value in (args.remote_output, args.remote_job_json):
        if not value.startswith("/") or "\x00" in value or "\n" in value:
            raise ValueError("remote paths must be absolute")
    args.local_output = args.local_output.resolve()
    args.local_output.mkdir(parents=True, exist_ok=True)
    if args.stop_command:
        args.stop_command = args.stop_command.resolve(strict=True)
        if not args.stop_command.is_file() or not os.access(args.stop_command, os.X_OK):
            raise ValueError("stop-command must be an existing executable file")
    receipt_path = args.local_output.with_name(args.local_output.name + ".watch.json")
    manifest = args.local_output.with_name(args.local_output.name + ".remote-sha256.jsonl")
    receipt: dict[str, Any] = {
        "state": "watching",
        "backup": "incomplete",
        "started_at": time.time(),
        "deadline_epoch": args.absolute_deadline_epoch,
        "host": args.host,
        "remote_output": args.remote_output,
        "local_output": str(args.local_output),
        "stop_helper": str(args.stop_command) if args.stop_command else None,
    }
    atomic(receipt_path, receipt)
    while True:
        try:
            time_left(args)
            # Poll failure never implies a completed job or successful backup.
            status = poll(args)
            receipt["remote_status"] = status
            receipt["last_polled_at"] = time.time()
            mirror(args)
            receipt["last_mirrored_at"] = time.time()
            inactive = status["service"].get("ActiveState") in {"inactive", "failed"}
            job = status.get("job")
            terminal = isinstance(job, dict) and job.get("state") in {"complete", "failed"}
            if inactive and terminal:
                # Recheck inactivity after the final mirror before hashing stable output.
                final_status = poll(args)
                if final_status["service"].get("ActiveState") not in {"inactive", "failed"}:
                    raise ValueError("service restarted during final backup")
                count = verify(args, manifest)
                receipt.update(
                    state="complete",
                    backup="verified",
                    verified_files=count,
                    manifest=str(manifest),
                    ended_at=time.time(),
                )
                stop(args, receipt, receipt_path)
                return receipt
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            receipt["last_error"] = f"{type(error).__name__}: {str(error)[:500]}"
            receipt["last_error_at"] = time.time()
        if time.time() >= args.absolute_deadline_epoch:
            receipt.update(state="deadline", ended_at=time.time())
            stop(args, receipt, receipt_path)
            return receipt
        atomic(receipt_path, receipt)
        if args.once:
            return receipt
        wake = min(time.time() + args.interval, args.absolute_deadline_epoch)
        while time.time() < wake:
            time.sleep(min(60, wake - time.time()))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--host", required=True)
    result.add_argument("--remote-output", required=True)
    result.add_argument("--local-output", type=Path, required=True)
    result.add_argument("--remote-job-json", required=True)
    result.add_argument("--remote-systemd-unit", required=True)
    result.add_argument("--absolute-deadline-epoch", type=float, required=True)
    result.add_argument("--stop-command", type=Path)
    result.add_argument("--interval", type=float, default=300)
    result.add_argument("--command-timeout", type=float, default=240)
    result.add_argument("--stop-timeout", type=float, default=60)
    result.add_argument("--once", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        result = run(args)
        print(json.dumps(result, indent=2))
        return 0 if result["backup"] == "verified" and result.get("stop") != "failed" else 1
    except (ValueError, OSError) as error:
        print(f"Cloud watcher rejected request: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
