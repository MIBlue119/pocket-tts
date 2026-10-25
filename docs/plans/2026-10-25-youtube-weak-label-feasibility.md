# YouTube → RE-USE → Breeze weak-label feasibility

The requested pipeline ran locally on RTX 3090 without manual listening or annotation.
A fresh one-command execution on user-supplied video `T_X7HGDuMYc` downloaded audio,
enhanced exactly seconds 10–30, and published four Chinese weak-label/audio pairs in
28.54 seconds. The run exited 0 with `weak_labels_ready`. This proves acquisition,
enhancement, labeling and training integration; label accuracy and synthesized
Taiwanese quality remain unverified.

## Artifacts and reproduction

- Entry point: `python -m training.taigi.build_video`; [copyable command and limits](../taigi-video-weak-labels.md).
- Source registry: [six user-provided playlists](../../training/configs/taigi_youtube_sources.json), five metadata entries per playlist, 30 entries total. Metadata inventory does not mean all videos were downloaded.
- Successful attempt: `data/taigi-enhancement-pilot/one-command-v2/run-report.json`.
- Release: `data/taigi-enhancement-pilot/one-command-v2/releases/f961f5fd869a24fa/`, four train rows, zero independent validation rows and zero verified labels.
- RE-USE revision: `31f82f2efad1095034492f236e8dac9239ba713b`; Breeze revision: `7b992682e7f5ceedd0a41ebec240f01ba469d19e`.
- Enhanced output: 960,000 float32 mono samples at 48 kHz, exactly 20 seconds. Each release row records enhanced audio, source identity, source interval, model revisions and hashes.
- Final targeted suite: 170 tests passed in 15.64 seconds. Scoped Ruff and type checks passed. An independent reviewer reproduced signal cleanup and verified the selected virtualenv interpreter.

The manually staged run published `4ac2724d60b54204`: four rows, 12.64 seconds after
segmentation. The wrapper run uses the same video interval and must not be added
as four new independent examples. Audio and intermediates remain local and ignored
by Git. The wrapper retains original audio and a verified copy in its release.

## What the labels mean

Training input is Chinese semantic text predicted by Breeze, paired with Taiwanese
audio, rather than required Taiwanese orthography. The schema records `asr_pseudo`,
`unverified_weak_label`, source-declared `nan-tw`, and `audio_variant: enhanced`.
Breeze output is weak supervision: it can contain mistranscriptions or Taiwanese
wording rather than standard Mandarin semantics. Producing Chinese characters alone
does not demonstrate correct Chinese-to-Taiwanese translation.

The provided YouTube metadata did not establish an open dataset license, so the
registry records `unknown`. These sources are local experimental inputs, distinct
from the ADI open dataset. No corpus or model weights have been published.

## Measurements

| Video and exact interval | RE-USE inference | ASR raw + enhanced | ASR peak allocated | Raw/enhanced character disagreement |
| --- | ---: | ---: | ---: | ---: |
| `F4J5hCDbEYA`, 30–50 s | 7.00 s | 14.74 s | 8.376 GiB | 4.82% |
| User video `T_X7HGDuMYc`, 10–30 s | 6.97 s | 11.05 s | 8.395 GiB | 13.41% |

RE-USE peak allocated memory was 3.063 GiB, reserved 4.92 GiB. Measurements are
separate warmed feasibility runs, not throughput or concurrent training benchmarks.
Character disagreement compares two ASR predictions; it is not CER against a gold
transcript and does not establish that enhancement improved quality.

## Defects found by actual execution

1. Approximate yt-dlp section cutting returned 39.97 seconds for a requested
   30-second clip. The bounded wrapper downloads the full short source and uses
   exact sample indices for the requested interval.
2. The installed Transformers pipeline's resampling path left higher-rate audio
   effectively interpreted as 16 kHz. Our `breeze-resample/v2` adapter explicitly
   resamples before pipeline invocation. A real 48 kHz sine fixture verifies
   duration and frequency preservation. Old ADI release `2066a2e8654ec2f5` is
   superseded by `cf0ad740136ec22e`; old training runs prove execution only.
3. Resolving the enhancement Python symlink escaped its virtualenv and loaded an
   incompatible Mamba extension. The wrapper now preserves the interpreter path;
   a real virtualenv test verifies `sys.prefix`.
4. Default SIGTERM handling could leave a GPU child alive. Scoped TERM/INT handlers
   now clean the entire active stage process group, write failure evidence and
   restore prior handlers. Real cancellation tests verify descendants stop.

## Training integration

Corrected ADI release `cf0ad740136ec22e` and manual YouTube release
`4ac2724d60b54204` form 38 rows totaling 81.596 seconds. The actual sentence loader
read all four enhanced rows with zero prompt frames. A new 512-token Chinese BPE
passed 47 conversion checks, and teacher training completed 20 steps with finite
loss and gradients, final loss 0.83427459, peak allocated GPU memory 5.9 GiB, complete
checkpoint/optimizer and inference export.

The [W&B run](https://forge.coreweave.com/wandb/weirenlan/pocket-tts-taigi/runs/dfe6b9296f774f9dace3fe9782a6c070)
was independently read as finished at step 20. Artifacts are in
`data/taigi-pilot-v2/` and `runs/taigi-smoke-v2/`. No held-out quality claim follows
from this deliberately tiny train-only experiment.

The earlier Nebius pilot completed 50 steps on superseded labels. Its checkpoint
was retrieved and verified, and dedicated VMs/disks were deleted. The existing OCR
VM remained stopped. Estimated total pilot cost is below US$0.70 before tax, not
an invoice. See [pilot evidence](2026-10-25-taigi-pilot-results.md).

## Expansion path

The current command handles one finite video of at most 300 seconds / 20 MiB
compressed media, and one interval of at most 20 seconds. Long interview episodes
in the sixth playlist exceed this bound. Durable queues, full episode chunking,
content-addressed reuse and automatic cross-release deduplication are not yet built.

For the next cloud run, freeze training release A while bounded producers build
release B. Use one GPU preprocessing slot shared by RE-USE and Breeze, separate
processes, and measured admission limits alongside the teacher. Benchmark teacher
alone and each sidecar for 50–100 representative steps; require no OOM, at most 10%
median step slowdown, and free VRAM of at least max(12 GiB, 20% of capacity). These
are proposed gates, not measured results. Adopt release B only in a new run at a
verified checkpoint boundary, with fixed held-out source/speaker groups.

Automated quality expansion should add overlap/language checks and raw/enhanced
consistency diagnostics. Current execution and basic filters do not establish all
of those checks. Human listening is not required to operate this weak-label path;
uncertainty stays explicit in the dataset.
