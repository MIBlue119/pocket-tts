# Taiwanese Pocket TTS pilot evidence

This report distinguishes pipeline execution from synthesis quality. The pilot
accepts ordinary Chinese semantic text and targets Taiwanese speech; it does not
require Taiwanese orthographic labels or fabricated word timestamps.

## Label correction and current usable smoke

The original release `2066a2e8654ec2f5` is superseded. Inspection of the installed
Transformers ASR pipeline found that its resampling call left 32/48 kHz input at
its original rate while treating it as 16 kHz. Our adapter now explicitly resamples
each exact interval before calling Breeze, records `breeze-resample/v2`, and has a
sample-count/frequency regression test. The old local and cloud runs below remain
checkpoint, restoration and infrastructure evidence only; their labels must not
be used for subsequent training.

Corrected ADI release `cf0ad740136ec22e` contains 34 weak-label rows. Combining it
with four enhanced YouTube rows from `4ac2724d60b54204` produced 38 pairs totaling
81.596 seconds. A new local run completed 20 steps with finite loss and gradients,
final loss 0.83427459, peak allocated GPU memory 5.9 GiB, checkpoint, optimizer and
inference export. The actual sentence loader consumed all four YouTube rows with
zero prompt frames. The new 512-token tokenizer passed 47 conversion parity checks.
The [corrected-data W&B run](https://forge.coreweave.com/wandb/weirenlan/pocket-tts-taigi/runs/dfe6b9296f774f9dace3fe9782a6c070)
was read back as finished at step 20, with an SDK finish acknowledgement.
Artifacts are `data/taigi-pilot-v2/` and `runs/taigi-smoke-v2/`.

The complete yt-dlp → RE-USE → Breeze wrapper also passed a fresh real-video run:
see [weak-label feasibility evidence](2026-10-25-youtube-weak-label-feasibility.md).
These are unverified Chinese weak labels, with no held-out quality score.

## Historical local execution on superseded labels

| Check | Observed result |
| --- | --- |
| Source | ADI Hokkien training shard, revision `ffcc36b9afe31ec50f7742c0c4c614f1a5b40151` |
| Acquired audio | 32 clips, 132.048 seconds, 21 source speaker IDs |
| Weak-label release | `2066a2e8654ec2f5`; 34 segments, 68.956 seconds, train-only |
| ASR | Breeze-ASR-26 revision `7b992682e7f5ceedd0a41ebec240f01ba469d19e` |
| Label quality | 0 independently verified labels; 0 quarantine entries in this small sample |
| Tokenizer | 512-token Chinese BPE; 43 conversion parity checks passed |
| Teacher | Released English 24-layer model; 312.4M trainable parameters, fixed Mimi |
| GPU | Local RTX 3090; peak allocated training memory 5.9 GiB |
| First attempt | 35-second execution limit; graceful checkpoint at step 11, exit 124 |
| Restore | Restored model and optimizer at step 11; continued through step 20, exit 0 |
| Final training loss | 0.75598395; finite loss and gradients throughout logged steps |
| Weight update | First transformer feed-forward weight changed; maximum absolute difference `3.0321e-5` versus pretrained export |
| Export loading | Public inference API loaded the exported model and Chinese tokenizer |
| Audio execution | Chinese input produced 0.88 seconds of finite 24 kHz audio; pronunciation and semantics unverified |
| Backup | Complete model/optimizer snapshot, checksums and final `READY` marker |

The short synthesized output is execution evidence only. The tiny corpus, mostly
short read-speech items, cannot establish general Chinese-to-Taiwanese quality.
There is no held-out quality score for this explicitly train-only smoke.

Local artifacts are under `data/taigi-pilot/` and `runs/taigi-smoke/`, excluded from
Git. The resumable backup is `data/taigi-pilot/checkpoints/local-step20/`:

- Model checkpoint SHA256: `c9f071c55852eb5b382b6ff3c39170604fc982005342279ed25c13e3ff8d1fa3`.
- Optimizer checkpoint SHA256: `efc8f2d5e1be87349a9f358db2db0c53b14b7c2496e63128662ad78b1f4acd60`.

## Experiment tracking

The `weirenlan/pocket-tts-taigi` W&B project is private. Remote API reads confirmed
both local attempts finished and received steps and loss:

- [Bounded attempt through step 11](https://forge.coreweave.com/wandb/weirenlan/pocket-tts-taigi/runs/57e67fea0e554b54ae89f0c338e3d9da).
- [Restored attempt through step 20](https://forge.coreweave.com/wandb/weirenlan/pocket-tts-taigi/runs/7f0b631b20a5428d96f00a816bbe8158).

The first online attempt exposed an insufficient two-second tracker shutdown wait.
It was closed through the SDK, verified remotely, and the launcher now has an
explicit bounded finish allowance and acknowledgement state. Raw audio, transcripts
and model weights were not uploaded to W&B.

## Verification and corrections

The final targeted training, runner, enhancement and video-wrapper suite passed
170 tests. Scoped Ruff and type checks pass. An independent same-environment comparison found 165 identical
full-repository type diagnostics on upstream baseline `44cb63e` and the reviewed
implementation, with zero new diagnostics.

Independent review reproduced and fixed duplicate audio crossing splits and stale
audio hashes. Releases now validate source bytes, group duplicate sources, copy
verified audio, and verify existing releases. Broken playlist support was removed.
Actual execution also exposed mismatched inherited torchaudio and a Transformers
development regression; the tested runtime uses matching torch/torchaudio 2.9.1
CUDA 12.8 and Transformers 4.57.1 in a separate environment.

## Cloud execution

The authorized total pilot budget is US$50. The first on-demand request failed with
an explicit capacity error; its dedicated VM and disk were removed. A dedicated
RTX PRO 6000 Blackwell Spot VM with a 150 GiB disk then reached RUNNING. An external
persistent watchdog was armed before creation with an absolute 13:30 UTC deadline,
unchanged across the retry. The existing stopped OCR VM was not needed.

The cloud GPU passed a CUDA-versus-CPU matrix multiplication check with
torch/torchaudio 2.9.1+cu128. The uploaded teacher matched local SHA256
`8fecefe7206ccfb02fe61a1bd1496d3238f534056f19e5dc0421049cbfbfa4df`.
The clean cloud code revision was `b29667cbec8c31fa39a34d8cf61abfb54f7edc0e`.

The first training attempt revealed an undeclared `jiwer` dependency in the minimal
cloud setup. It is now declared in the `taigi` group and checked by importing the
actual training entrypoint during setup. The retry completed 50 consecutive steps,
exit 0, final loss 0.68437773, final gradient norm 2.040977, peak GPU memory 5.9 GiB,
and a step-50 model/EMA checkpoint, optimizer sidecar and inference export. All logged
losses and gradients were finite. Different minibatches mean the first and last
training loss do not establish improvement.

Cloud metrics were imported after execution through local credentials. The
[cloud W&B run](https://forge.coreweave.com/wandb/weirenlan/pocket-tts-taigi/runs/nebius-20261025-step50)
was independently read back as `finished` with trainer step 50. Raw corpus and model
files were not uploaded. Local evidence is under `data/taigi-pilot/cloud-results/`.
All 6,329,608,365 backup bytes were copied locally and verified against checksums;
the primary agent independently rehashed all six snapshot entries and the inference
export and loaded the step-50 model/EMA checkpoint. Snapshot files are under
`data/taigi-pilot/cloud-results/checkpoints/cloud-step50/`.

Provider reads at 09:53:46 UTC confirmed both pilot VMs and their disks absent.
The unrelated OCR VM remained STOPPED with resource version 14. The watchdog timer
was disabled after cleanup. Conservative pilot cost is below US$0.70 before tax;
this is an estimate, not an invoice. No pilot compute or disk remains allocated.
Cleanup evidence is `data/taigi-pilot/cloud/cleanup-receipt.json`.

## Next experiment

Expand to sentence-rich Taiwanese recordings, keep speaker/source groups held out,
and retain Chinese ASR labels as explicit weak supervision. Train a larger
training-only tokenizer, increase effective batch size, and compare fixed prompts
and automatic intelligibility checks across checkpoints. Using the labeling ASR
again is a consistency check, not an independent pronunciation score. Local RE-USE plus Breeze feasibility is verified; large YouTube acquisition and
concurrent cloud preprocessing remain next-stage work. Distill a smaller model after
the teacher demonstrates useful behavior on held-out inputs.

Metadata analysis of the pinned ADI train manifest found 31,482 rows: 23,158 declared
`nan-tw`, 2,792 `zh-CN`, and 5,532 without locale. A deterministic proxy requiring at
least six Han characters before parenthesized romanization yields 1,794 rows,
1,516 sentence IDs and 130 client IDs. Sort eligible rows by path, retain the first per sentence ID, then sort by
(client ID, sentence ID) and cap at 50 per client: this leaves 911 initial candidates. These are text-length candidates, not
verified complete spoken sentences. The TSV has no durations: decode and measure
accepted audio before claiming the proposed 1–5-hour next-stage target.
