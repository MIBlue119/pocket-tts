from __future__ import annotations

import argparse
from pathlib import Path

from training.taigi.pipeline import prepare_candidates, transcribe_release


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m training.taigi")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="segment and quality-check declared Taigi audio")
    prepare.add_argument("source_manifest", type=Path)
    prepare.add_argument("output_dir", type=Path)
    prepare.add_argument("--train-only", action="store_true")
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--silence-db", type=float, default=-42.0)
    prepare.add_argument("--min-silence", type=float, default=0.35)
    prepare.add_argument("--min-duration", type=float, default=0.8)
    prepare.add_argument("--max-duration", type=float, default=20.0)
    prepare.add_argument("--allow-playlist", action="store_true")
    transcribe = commands.add_parser("transcribe", help="create a pinned Breeze weak-label release")
    transcribe.add_argument("candidates", type=Path)
    transcribe.add_argument("releases_dir", type=Path)
    transcribe.add_argument("--device", default="cpu")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_candidates(
            args.source_manifest,
            args.output_dir,
            train_only=args.train_only,
            seed=args.seed,
            silence_db=args.silence_db,
            min_silence=args.min_silence,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            allow_playlist=args.allow_playlist,
        )
    else:
        result = transcribe_release(args.candidates, args.releases_dir, device=args.device)
    print(result)
    return 0
