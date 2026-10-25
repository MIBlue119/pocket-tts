# Prepare Chinese-semantic and Taigi speech pairs

This command builds sentence-level weak labels for direct Chinese text to Taigi speech training. It does not create verified transcripts or detect the spoken language.

## Install the data tools

Run the commands in the repository environment. Local WAV and FLAC input needs `soundfile`. Breeze transcription needs `transformers`. URL acquisition also needs the `yt-dlp` executable and FFmpeg.

The development dependency group already includes `soundfile` and `transformers`. The project does not declare `yt-dlp` because local files are the default input.

## Create a source manifest

Write one JSON object per source. Declare `language` as `nan-tw`. The command rejects unknown or inferred languages.

```json
{"audio_path":"/data/show-a.wav","source_id":"show-a-episode-1","group_id":"show-a-episode-1","language":"nan-tw"}
{"audio_path":"/data/show-b.flac","source_id":"show-b-episode-3","group_id":"show-b-episode-3","language":"nan-tw"}
```

`group_id` defaults to `source_id`. You can set `split` to `train` or `valid`. All sources in one group must use the same split. The command assigns any remaining groups before segmentation, so one source cannot cross the train and validation split.

For a bounded YouTube download, replace `audio_path` with one explicit `url`. The command records yt-dlp's stable video ID in the source ID and uses `download-archive.txt`. It rejects playlists unless you pass `--allow-playlist`.

## Prepare candidates

```bash
/mnt/HDD1/miniconda3/bin/python -m training.taigi prepare sources.jsonl data/taigi-work
```

The command finds deterministic RMS silence boundaries. It quarantines an interval longer than 20 seconds instead of inventing a cut without a pause. These boundaries are pauses, not semantic or punctuation claims. It writes:

- `candidates.jsonl` with accepted audio candidates, offsets, source hashes, groups, splits, and `source_declared` language provenance.
- `quarantine.jsonl` with empty, silent, clipped, repeated-loop, or too-short intervals.
- `prepare-report.json` with counts and the segmentation settings.

A single source group cannot produce a real validation split. For a pipeline smoke test only, pass `--train-only`.

## Generate Breeze weak labels

```bash
/mnt/HDD1/miniconda3/bin/python -m training.taigi transcribe \
  data/taigi-work/candidates.jsonl data/taigi-releases --device cuda:0
```

The transcriber pins `MediaTek-Research/Breeze-ASR-26` to revision `7b992682e7f5ceedd0a41ebec240f01ba469d19e`. It reads existing Hugging Face authentication and does not accept credentials as arguments. Each complete selected interval gets one Chinese semantic pseudo label.

The command writes an immutable content-addressed directory under `data/taigi-releases`. A repeated run with the same outputs returns the existing release. Each release contains:

- `train.jsonl` and `valid.jsonl` with only `path`, `start`, `duration`, and `transcript`.
- `metadata.jsonl` with source provenance, hashes, the pinned model revision, and `unverified_weak_label` status.
- `quarantine.jsonl` for empty ASR output.
- `report.json` with accepted and quarantined counts. `verified_labels` is always zero.

RE-USE or another enhancer can create a later audio variant. This pipeline records only `raw` audio and does not require an enhancer.
