from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def acquire_url(url: str, output_dir: Path) -> tuple[Path, str]:
    executable = shutil.which("yt-dlp")
    if executable is None:
        raise RuntimeError("URL acquisition requires the `yt-dlp` executable")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = subprocess.run(
        [executable, "--dump-single-json", "--skip-download", url],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(metadata.stdout)
    if info.get("_type") == "playlist":
        raise ValueError("playlist URLs are not supported; provide one explicit video URL")
    video_id = str(info.get("id") or "")
    if not video_id:
        raise ValueError("yt-dlp did not return a stable video ID")
    archive = output_dir / "download-archive.txt"
    template = str(output_dir / "%(id)s.%(ext)s")
    subprocess.run(
        [
            executable,
            "--no-playlist",
            "--download-archive",
            str(archive),
            "-x",
            "--audio-format",
            "flac",
            "-o",
            template,
            url,
        ],
        check=True,
    )
    matches = sorted(output_dir.glob(f"{video_id}.*"))
    if not matches:
        raise RuntimeError(f"yt-dlp completed without an audio file for {video_id}")
    return matches[0].resolve(), video_id
