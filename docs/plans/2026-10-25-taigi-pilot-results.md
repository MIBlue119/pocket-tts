# Taiwanese Pocket TTS pilot evidence

This report distinguishes pipeline execution from synthesis quality. The pilot
accepts ordinary Chinese semantic text and targets Taiwanese speech; it does not
require Taiwanese orthographic labels or fabricated word timestamps.

## Verified local execution

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

The training and runner suite passed 141 tests before the two tracker-finish tests
were added; the expanded runner suite separately passed all 18 tests. Scoped Ruff
and type checks pass. An independent same-environment comparison found 165 identical
full-repository type diagnostics on upstream baseline `44cb63e` and the reviewed
implementation, with zero new diagnostics.

Independent review reproduced and fixed duplicate audio crossing splits and stale
audio hashes. Releases now validate source bytes, group duplicate sources, copy
verified audio, and verify existing releases. Broken playlist support was removed.
Actual execution also exposed mismatched inherited torchaudio and a Transformers
development regression; the tested runtime uses matching torch/torchaudio 2.9.1
CUDA 12.8 and Transformers 4.57.1 in a separate environment.

## Cloud execution

The authorized total pilot budget is US$50. A dedicated RTX PRO 6000 VM with a
150 GiB disk was requested at 09:30 UTC. An external persistent watchdog was armed
before creation with an absolute 13:30 UTC deadline. At this report's initial
writing the provider was still starting the VM; no cloud training result is claimed
yet. Final execution and cleanup evidence will be appended after verification.

## Next experiment

Expand to sentence-rich Taiwanese recordings, keep speaker/source groups held out,
and retain Chinese ASR labels as explicit weak supervision. Train a larger
training-only tokenizer, increase effective batch size, and compare fixed prompts
and automatic intelligibility checks across checkpoints. Using the labeling ASR
again is a consistency check, not an independent pronunciation score. RE-USE and
large YouTube acquisition remain later experiments. Distill a smaller model after
the teacher demonstrates useful behavior on held-out inputs.
