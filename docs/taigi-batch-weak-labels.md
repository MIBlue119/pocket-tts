# Resumable Taiwanese weak-label batches

The batch command runs the tested yt-dlp → RE-USE → Breeze path on a finite list
of short video intervals. It saves each attempt and skips verified successful jobs
on restart. The merge command creates a fixed train/validation release; training
never reads a manifest that a producer is still changing.

The labels remain unverified Chinese semantic supervision. Processing success,
validation loss and Breeze agreement do not establish Taiwanese pronunciation.

## Prepare and run a finite batch

Create JSONL with one individual YouTube URL per interval:

```json
{"url":"https://www.youtube.com/watch?v=T_X7HGDuMYc","group_id":"youtube-playlist-PL9X_7mTn8zvVmd2vyiWdneNZhtkGzogI-","start":10,"duration":20}
```

Use the same group for related episodes or programs. Each interval is at most
20 seconds; each full source is at most 300 seconds and 20 MiB of compressed media.
The current implementation downloads the short source separately for each interval.
For long interviews and whole playlists, use the full-corpus command below instead
of raising the interval-batch timeout. The committed source registry contains metadata,
not downloaded audio or proof of a dataset license.

Run from the feature checkout with the existing model environments:

```bash
USE_TF=0 USE_FLAX=0 /mnt/HDD2/proj_tts/.venvs/taigi-pipeline/bin/python \
  -m training.taigi.batch /absolute/path/jobs.jsonl /absolute/path/batch-output \
  --enhancer-python /mnt/HDD2/proj_tts/.venvs/taigi-enhance/bin/python \
  --cache-dir /mnt/HDD2/proj_bark/pretrained_model/huggingface/hub \
  --max-seconds 2700 --timeout 180 --max-attempts 2 --min-free-gib 10
```

Repeat the exact command to resume. Configuration changes require a new output
folder. Completed artifacts are checked before reuse; corrupted files stop the run.
Attempts are capped across restarts. Each invocation has its own overall deadline.
A failed or incomplete batch exits nonzero and preserves `state.json`,
`batch-report.json`, per-attempt logs and diagnostics. Some intervals can contain
no usable speech; do not hide those failures by retrying indefinitely.

Linux x86-64 with PID-descriptor support, readable same-user `/proc` and `flock` are required. Attempt ownership tokens are persisted before launch, so cleanup can identify detached stages after wrapper death. Only the named token is compared; process environments are not logged. A second writer is refused. Cancellation
signals the active wrapper and waits for its subprocess cleanup; escalation can
add up to 20 seconds beyond the deadline. Keep original attempt directories:
release paths and checksums refer to them. Disk admission checks preserve a free
space threshold before each bounded job, not a filesystem quota.

## Freeze one dataset version

Collect the release paths only from completed jobs. The following prints an
explicit list; preserve that list with the experiment rather than using a glob
that changes while production runs:

```python
import json
from pathlib import Path

root = Path('/absolute/path/batch-output')
state = json.loads((root / 'state.json').read_text())
releases = []
for job in state['jobs'].values():
    if job['state'] == 'complete':
        attempt = Path(job['attempts'][-1]['path'])
        releases.append(json.loads((attempt / 'run-report.json').read_text())['release'])
print('\n'.join(releases))
```

Pass the frozen paths as separate arguments:

```bash
/path/to/asr/bin/python -m training.taigi.merge RELEASE_1 RELEASE_2 \
  --output-dir /absolute/path/merged-releases --valid-group HELD_OUT_PROGRAM
```

The merger requires corrected Breeze `breeze-resample/v2` inputs, checks manifests
and copied audio, deduplicates matching original intervals, and rejects conflicting
labels. Group links are computed before duplicate removal. Video IDs, source
hashes and audio hashes unite related groups; all linked material stays together.
This detects exact provenance/byte duplicates, not perceptual reuploads with
changed audio. Actual speaker overlap between programs remains unverified.

The output contains immutable copies of accepted audio, unique merged utterance
IDs, original parent identities, compact `train.jsonl` / `valid.jsonl`, metadata,
counts, durations and hashes. Both splits must contain rows unless `--train-only`
is explicitly selected. A held-out development set is not a final independent test.

## Train and compare generated speech

Train the tokenizer using only `train.jsonl`, then run the existing tokenizer
conversion parity check. Set vocabulary size to fit the actual character inventory
and byte fallback; freeze it for the entire run. A changed tokenizer requires a
fresh adaptation run, not silent checkpoint resume with different token IDs.

Astra's initial local diagnostic recipe is microbatch 1, gradient accumulation 8,
learning rate `1e-4`, warmup 10, at most 100 updates, and checkpoints/validation every
25 updates. Keep the 24-layer pretrained model, reset text embeddings, train the
backbone and flow head, freeze Mimi, use LSD and zero voice/text dropout. Use raw
audio sentence mode with precompute and compilation disabled. This recipe is a
bounded diagnostic, not a tuned optimum.

Set `num_valid_batches` to the verified held-out row count for batch size 1, after
checking every row loads. The current loader cycles; increasing this count does
not create more independent examples. Validation now uses isolated fixed RNG state
and logs a baseline at step 0 on fresh runs. Its score is batch-averaged flow/EOS
loss under teacher forcing, not generated-speech accuracy.

Use the existing bounded launcher with a 720-second training deadline, 120-second
checkpoint grace and W&B metrics. Save the fixed merged-release hashes in its
provenance JSON. Inspect the actual exit, step, checkpoint and tracker state.

For generated-audio comparison, save 1–20 JSONL prompts with unique `id` and `text`.
Include training examples, unseen Chinese, and pairs differing in negation or a
number. Use the same prompts/settings for initialization and raw checkpoints:

```bash
USE_TF=0 USE_FLAX=0 /path/to/asr/bin/python -m training.taigi.evaluate \
  train.yaml prompts.jsonl /absolute/path/baseline --max-frames 125 --asr

USE_TF=0 USE_FLAX=0 /path/to/asr/bin/python -m training.taigi.evaluate \
  train.yaml prompts.jsonl /absolute/path/checkpoint-evaluation \
  --checkpoint /absolute/path/checkpoint_00000100.pt --max-frames 125 --asr
```

Use an external process deadline when scheduling these diagnostics. Output folders
must be new. The evaluator rebuilds the seeded training initialization or loads
raw checkpoint weights, uses zero voice prompt, temperature 0.3, CFG 1, fixed seeds,
and at most 125 frames (10 seconds) per prompt by default. It records short/empty,
silent or nonfinite output, clipping, duration, frame-limit hits and optional Breeze
text. Float WAV preserves clipping evidence. ASR loads after generation models are
released. Character similarity measures consistency with the same labeling ASR;
it does not verify Taiwanese language, tones, translation or naturalness.

Only consider longer training or cloud spending when fixed generation and held-out
diagnostics show a useful signal. Preserve failed diagnostics as evidence. The
remaining budget is a total ceiling shared by compute and storage, not a reason
to run more steps on insufficient data.

## Process the complete six-playlist corpus

`training/configs/taigi_youtube_full_inventory.json` freezes 290 unique video IDs
from all six supplied playlists. The supervisor visits playlists in round-robin
order, downloads each full source once, and records an explicit outcome for every
video. Unavailable sources remain failures in the report. A successful download
does not mean every second is accepted as speech training data.

```bash
USE_TF=0 USE_FLAX=0 /path/to/asr/bin/python -m training.taigi.corpus run \
  training/configs/taigi_youtube_full_inventory.json /absolute/path/corpus \
  --enhancer-python /path/to/enhancer/bin/python --cache-dir /absolute/path/hf/hub \
  --max-seconds 43200 --min-free-gib 30

/path/to/asr/bin/python -m training.taigi.corpus status /absolute/path/corpus
```

Sources are limited to two hours and 512 MiB of compressed audio by default;
all enumerated durations fit that duration bound. RE-USE loads once per video
and processes the entire decoded source in chunks of at most ten minutes. Breeze
loads once per video when candidates exist. Completed chunks, transcripts and
releases have content hashes; restart verifies them before reuse. The plan also
freezes the processing code and model configuration. Changing those requires a
new output directory. Deadline and free-space admission limits can change on
resume; a separate cloud budget clock must never reset with an invocation.

For automatic training after corpus processing, first initialize the corpus plan
with the command above. A tiny positive `--max-seconds` can initialize the queue
without admitting a download. Then run the following inside an actual systemd
service with `KillMode=control-group`, `TimeoutStopSec=60`, and a `RuntimeMaxSec`
that ends before the external VM stop deadline:

```bash
/path/to/asr/bin/python -m training.scripts.run_taigi_corpus_job \
  training/configs/taigi_youtube_full_inventory.json /absolute/path/corpus \
  /absolute/path/new-training-job \
  --valid-group playlist:PL9X_7mTn8zvUHn5P_-ngeHVUgqi4-zMha \
  --max-seconds 64800 --training-seconds 14400
```

The orchestrator requires systemd's invocation marker, but the operator must
verify the real service containment properties. It resumes corpus processing
and refuses training while any source remains pending or interrupted. Explicit
terminal source failures are counted and can coexist with usable releases.
Training uses a fresh train-only BPE tokenizer with exact conversion parity,
the pretrained 24-layer teacher with reset text embeddings, batch 1 with eight
accumulation steps, and an initial 200 updates at learning rate `3e-5` with
100 warmup updates. Checkpoints and validation run every 50 updates; four raw
candidates are retained. Validation covers at most 100 held-out rows per check.
Twelve fixed generated prompts are saved before and after training; W&B runs offline so cloud
credentials are unnecessary. Training job directories cannot be resumed; use a
new job directory while retaining the resumable corpus directory.

The final-update checkpoint is an experiment candidate, not an automatically
promoted model. The small 100-update pilot worsened held-out loss relative to its
fixed baseline, so a 2,000-update run is not the default. Longer training needs
evidence from the full-corpus pilot's held-out and generated-speech diagnostics.
Empty labels and runs of at least 12 identical alphabetic/CJK characters are
quarantined; original text, audio references and reasons remain recorded. This
narrow rule does not detect every ASR error or establish Taiwanese correctness.

Keep a separate persistent backup controller and provider-level stop timer.
Stopping a VM does not delete its billable disk. Verify copied artifacts before
removing cloud storage. Source licensing, speaker separation and Taiwanese
pronunciation quality remain separate from pipeline execution success.
