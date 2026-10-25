# Prepare Chinese-semantic and Taigi speech pairs

This command builds sentence-level weak labels for direct Chinese text to Taigi speech training. It does not create verified transcripts or detect the spoken language.

## Install the data tools

Run the commands in the repository environment. Local WAV and FLAC input needs `soundfile`. Breeze transcription needs `transformers`. URL acquisition also needs the `yt-dlp` executable and FFmpeg.

The `taigi` dependency group includes `soundfile`, the tested Transformers version,
and `yt-dlp`. Use the isolated environment in [taigi-pilot.md](taigi-pilot.md);
FFmpeg is a separate system dependency for URL acquisition.

## Create a source manifest

Write one JSON object per source. Declare `language` as `nan-tw`. The command rejects unknown or inferred languages.

```json
{"audio_path":"/data/show-a.wav","source_id":"show-a-episode-1","group_id":"show-a-episode-1","language":"nan-tw"}
{"audio_path":"/data/show-b.flac","source_id":"show-b-episode-3","group_id":"show-b-episode-3","language":"nan-tw"}
```

`group_id` defaults to `source_id`. You can set `split` to `train` or `valid`. All sources in one group must use the same split. Sources with identical audio bytes also stay in one split, even when their source or group IDs differ. The command preserves extra fields such as `license`, `repo`, and `revision` in each candidate's `source_metadata`.

Set `audio_variant` to `enhanced` for RE-USE output. It defaults to `raw`. Put lineage such as `raw_sha256`, the enhancer name, its revision, and its settings in additional source fields. The command retains those fields in `source_metadata`.

For a bounded YouTube download, replace `audio_path` with one explicit video `url`. The command records yt-dlp's stable video ID in the source ID and uses `download-archive.txt`. It rejects playlist URLs.

## Prepare candidates

```bash
.venv-train/bin/python -m training.taigi prepare sources.jsonl data/taigi-work
```

The command finds deterministic RMS silence boundaries. It quarantines an interval longer than 20 seconds instead of inventing a cut without a pause. These boundaries are pauses, not semantic or punctuation claims. It writes:

- `candidates.jsonl` with accepted audio candidates, offsets, source hashes, groups, splits, and `source_declared` language provenance.
- `quarantine.jsonl` with empty, silent, clipped, repeated-loop, or too-short intervals.
- `prepare-report.json` with counts and the segmentation settings.

A single source group cannot produce a real validation split. For a pipeline smoke test only, pass `--train-only`.

## Generate Breeze weak labels

```bash
.venv-train/bin/python -m training.taigi transcribe \
  data/taigi-work/candidates.jsonl data/taigi-releases --device cuda:0
```

The transcriber pins `MediaTek-Research/Breeze-ASR-26` to revision `7b992682e7f5ceedd0a41ebec240f01ba469d19e`. Adapter `breeze-resample/v2` reads the exact source interval and resamples it to the model feature extractor rate before inference. Release metadata records that rate. The command reads existing Hugging Face authentication and does not accept credentials as arguments. Each complete selected interval gets one Chinese semantic pseudo label.

Before transcription, the command verifies each source file against the hash recorded during preparation. It stops before loading or calling ASR if a source changed.

The command writes an immutable content-addressed directory under `data/taigi-releases`. It copies accepted source files to hash-named files in the release and points the training manifests at those copies. A repeated run verifies the published copies before returning the existing release. Each release contains:

- `train.jsonl` and `valid.jsonl` with only `path`, `start`, `duration`, and `transcript`.
- `metadata.jsonl` with source provenance, hashes, the pinned model revision, and `unverified_weak_label` status.
- `quarantine.jsonl` for empty ASR output.
- `report.json` with accepted and quarantined counts. `verified_labels` is always zero.

RE-USE or another enhancer can create a later audio variant. This pipeline records only `raw` audio and does not require an enhancer.
