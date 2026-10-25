* _2026-10-25 19:02:00 +0800 (codex/unknown)_

# Process every supplied playlist video

The owner corrected the scope: “為什麼 我給你的 youtube 清單 不是有超多影片 你應該要都處理”.
All six playlists have now been enumerated without entry limits:290 unique videos,
26.1 known source hours, longest3200seconds. The earlier25-video subset was only a
pilot and must not be represented as fulfillment of this request.

## Requirements

- R6: Every enumerated video receives a durable queue record and explicit outcome;
  no first-five, short-video-only or arbitrary sample restriction.
- R7: Download each full source once; process its complete decoded interval in
  bounded chunks, preserve exact source provenance, and reuse verified stage work.
- R8: Keep RE-USE resident across a video's chunks and Breeze resident across its
  candidates. Release one model process before admitting the next; no GPU overlap.
- R9: Automatically persist partial progress, empty/no-speech outcomes and failures;
  account for private/unavailable sources and pauses due to budget or disk limits.
- R10: Freeze accepted releases for training and use Astra's diagnostics. Long runs
  must survive terminal loss; cloud spend, if used, remains within totalUSD50.

## Contracts

`training.taigi.corpus_worker` owns two commands: `enhance SPEC_JSON` and
`transcribe SPEC_JSON`. Spec keys: raw_audio (absolute path), video_id, group_id,
original_url, license, output_dir (absolute per-video stage directory), cache_dir,
device, asr_batch_size (1 by default, opt-in 2–4). Enhance loads predictor once, divides actual decoded frames into contiguous
chunks at most600seconds using exact source sample offsets, reuses verified complete
chunks and preserves partial diagnostics. It writes `sources.jsonl`, candidates and
an enhancement report including total covered frames, model identity and hashes.
Source IDs may include chunk offsets; original video/hash/group remain in metadata.
Preparation remains train-only per video; final grouping occurs during merge.
Transcribe reads those candidates and uses per-segment text caches keyed by audio
hash/interval/model+adapter/code/batch-size identity, then publishes via existing transcribe_release.
Load Breeze lazily once per video. A durable `result.json` reports state
`complete` or `no_candidates`, release path if present, counts and provenance.
Labels remain weak; long/silent/clipped segments are counted in quarantine rather
than hidden. Whole-source processing does not mean every sample is training-eligible.

`training.taigi.corpus` normalizes the complete inventory into a frozen finite
video queue, unions duplicate membership groups and retains all original playlists.
It downloads each finite available source once through yt-dlp, stores verified raw
WAV and metadata, and supervises worker stages using existing batch.execute/token
cleanup. It persists per-video stage state, hashes, capped retry history and reports
atomically under flock. Resume verifies prior artifacts and avoids re-download or
model work for completed stages. Controlled cancellation stops the current owned
process before new admission. Runtime, source-size and disk-reserve limits pause
or explicitly fail work, never silently remove it. Defaults must admit every known
source (max3200seconds); no inherited300second cap. Unknown duration requires real
metadata inspection, not omission. Read-only status reports totals and active stage.

Output SPEC has one stable per-video directory. Acquisition attempts may have
separate directories; completed raw audio is promoted to a stable source.wav.
Enhance writes `candidates/candidates.jsonl`; transcribe writes `result.json`.
Partial model outputs remain outside completed artifacts until validation/promotion.

## Invariants

- INV6: Every inventory video maps to one queue item with its memberships and an
  explicit pending/running/complete/no_candidates/failed result; none silently vanish.
- INV7: Chunk intervals tile the actual decoded source frames without gaps/overlap;
  incomplete or changed artifact hashes refuse reuse. Original audio remains available.
- INV8: At most one GPU model stage is active, including after interruption/restart.
- INV9: Whole-corpus output cannot change a frozen training manifest; model/dataset
  identity and weak-label uncertainty remain explicit.

## Acceptance criteria

Enumerate all290videos and compare queue IDs exactly. Test full coverage including
fractional final chunks, resident model load count, stage resume, corruption refusal,
empty speech, interrupted processing and explicit failures. Process one complete
long episode on the execution host before scaling, inspect coverage/hashes and actual weak labels.
Confirm durable execution/status and resource shutdown rules before cloud use.
Reuse the established merger, deterministic validation and generated-audio checks.

The authorized Nebius host is the execution host for the whole-episode acceptance.
Its actual 52.5-minute trial covered every decoded frame and produced 760 weak
labels. A corrected cache environment and CPU-only verified resume recovered the
initial short-video ASR setup failure; original attempts remain recorded.

The owner's acceleration request adds a measured batching acceptance: compare
single and four-item Breeze inference on unequal durations and a partial final
batch, preserve order, and prove cache resume uses no model load or call. The
resident benchmark found 1.87× throughput for batch four; duplicate RE-USE
processes were slower, so production keeps one GPU stage active.

The first full-data training stage is a 200-update diagnostic, not automatic
promotion: train-only BPE with exact conversion parity, learning rate 3e-5,
100 warmup updates, validation/checkpoints every 50 and four retained raw
candidates. Save fixed generated audio and same-ASR consistency before and after.
Quarantine obvious repeated-character labels while retaining evidence. Every
sentence must have a positive terminal EOS target, excluded from flow targets;
the real small-data experiment exposed and regression-tested this missing target.
Old validation losses are not comparable across that objective correction.

## Minimality check

Raising single-video limits alone would repeatedly download sources and reload
models for every20seconds. Two modules reuse the existing enhancement, ASR,
preparation, publication and lifecycle helpers while adding only resident stage
processing and a per-video queue. No external database, new ML model or service API
is needed. A durable local service or bounded cloud job may run the same CLI.

Self-check: This supersedes the pilot's source restriction under explicit owner direction; budget and quality claims remain bounded.
