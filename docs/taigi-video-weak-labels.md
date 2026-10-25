# Build weak labels from one Taigi video

Run this command with the Python environment that contains Breeze ASR, `yt-dlp`, and the data pipeline dependencies. Supply the separate RE-USE environment as `--enhancer-python`. Both subprocess environments import this checkout. FFmpeg and Node must be on PATH. `--js-runtime deno` selects Deno instead.

```bash
/path/to/asr/bin/python -m training.taigi.build_video \
  'https://www.youtube.com/watch?v=VIDEO_ID_11' \
  data/taigi/video-attempt-001 \
  --enhancer-python /mnt/HDD2/proj_tts/.venvs/taigi-enhance/bin/python \
  --group-id speaker-or-program-group --train-only \
  --start 10 --duration 20 --timeout 300 \
  --cache-dir /mnt/HDD2/proj_bark/pretrained_model/huggingface/hub
```

Replace the URL with one individual video. The destination must not exist. Reusing a destination fails without overwriting its files. Choose a shared `group-id` for related source material so a later combined dataset can keep related audio in the same split. `--train-only` is mandatory because this command processes only one source group. It produces no independent validation set.

Supplying a source declares that its speech is Taiwanese (`nan-tw`). The command does not verify the language or infer usage permission. Source metadata retains any reported license, or records `unknown`. It records the original URL, video ID, group ID, raw WAV hash, exact frame interval, enhancement lineage, and enhanced audio hash.

The command performs these stages under one subprocess deadline.

1. Inspect metadata for one finite video. Playlists, live streams, and sources longer than `--max-source-seconds` are rejected. The source limit defaults to 300 seconds and cannot exceed 300.
2. Download the complete audio stream with a compressed media cap of `--max-download-mib`, default and maximum 20 MiB. Retain the compressed file, decode it to WAV, and check its actual duration and the requested interval. The decoded WAV can exceed 20 MiB. No browser cookies or user yt-dlp configuration are used.
3. Run RE-USE on the exact requested sample interval with five-second overlap chunks. `--duration` defaults to 20 seconds and cannot exceed 20. The wrapper verifies the output frame count and enhancement sidecar against the source. It does not use approximate `yt-dlp --download-sections` cuts. Enhancement uses cached model artifacts unless `--download-model` is supplied.
4. Invoke the existing candidate preparation CLI with `--train-only`, then the Breeze transcription CLI. `--device` controls Breeze and defaults to `cuda:0`. The enhancer uses its own CLI's default CUDA device.

`--timeout` defaults to 300 seconds and accepts at most 3600. Each external stage receives only the remaining time. Expiry kills the stage's process group, including FFmpeg or model subprocesses. Local file verification and report writes occur between stages and can add filesystem overhead. No training or cloud resource allocation follows this command.

Each stage has a stdout `.log` and a separate `.stderr.log`. `run-report.json` records the final state and failing stage. `weak_labels_ready` requires a published release with at least one accepted weak label and a nonempty `train.jsonl`. If silence, quality checks, or empty transcripts leave no usable labels, the run fails and preserves its diagnostics. Inspect the reported release's `metadata.jsonl` and `report.json`. Labels remain `asr_pseudo` and `unverified_weak_label`; successful processing does not establish transcription accuracy or speech quality.

Raw media and intermediate files remain available in the attempt directory. The published release contains its own audio copy. There is no queue, playlist expansion, automatic upload, or automatic cleanup.
