# Taiwanese corpus and teacher pilot evidence

Updated 2026-10-25T19:52:50+08:00.

The full 290-video job has been launched. A Taiwan-only download restriction on
some sources required a controlled transport repair; full completion is pending.
The model is not ready for production speech synthesis.

## Source processing

All six playlists are frozen in `training/configs/taigi_youtube_full_inventory.json`:
290 unique videos and approximately 26.1 source hours. The original interval pilot
completed 54/56 windows and produced 195 weak pairs, 459.08 seconds. Two source failures
remain in its history. A new immutable filtered release retains 193 pairs: 165 training
rows, 381.14 seconds, and 28 held-out rows, 72.68 seconds. Two repeated-character labels
were quarantined; original text/audio and old releases were preserved.

A complete cloud trial covered 151,202,816 frames at 48 kHz (52.501 minutes) without
source gaps. It yielded 760 weak labels spanning 2318.72 seconds and one preparation
quarantine. The short full-video trial yielded 17 labels. Both trial queue entries
were verified complete after an explicit cache-environment recovery. A read-only
long-episode audit verified all 760 text hashes and candidate identities; no extreme
character-rate or repetitive-phrase anomalies appeared under the stated checks.
These checks do not establish transcript or Taiwanese-language correctness.

## Measured acceleration

The long enhancer used 97–99% GPU compute. Two resident enhancement processes were
8.9% slower than one; two Breeze processes gave no useful gain. One resident Breeze
model with batch 4 was 1.87 times faster on equal 20-second inputs.

The production-relevant comparison used 39 unequal clips totaling 127.54 seconds,
including hashing and per-segment cache writes: batch 1 took 15.360 seconds and
stable-duration-grouped batch 4 took 11.470 seconds (1.339 times throughput, 25.3% less wall
time). All 39 texts matched exactly, including a final 3-item batch. Resuming the
cache used zero model loads and zero inference calls. Peak allocated memory was
8.36 GiB versus 15.54 GiB. Production keeps one GPU stage active and uses batch 4 ASR.

## Training diagnosis and result

A first 100-update run exposed missing terminal EOS supervision for batch size 1.
The longest sentence filled every valid codec frame, so no positive stopping
target was trained. Commit `78bc38d` appends one invalid terminal position for EOS;
that position is excluded from flow targets and latent statistics. Regression
cases failed before the fix; 54 related CPU tests passed afterward, and a separate
review found no blocker.

Fresh 100-update run `diagnostic-v5-eos` used the same 193-row release and 813-token
train-only BPE tokenizer. Exact conversion parity passed. All six fixed generated
prompts now ended naturally at 1.20–2.16 seconds, whereas the old final candidate
hit the 10-second limit for all six. Generated text remained unreliable under
Breeze consistency checks, so the candidate is not promoted.

Corrected-objective validation on the 28-row held-out program:

| Update | Held-out total loss | EOS loss |
|---:|---:|---:|
| 0 | 0.554617 | 0.119969 |
| 25 | 0.567941 | 0.033948 |
| 50 | 0.563741 | 0.032471 |
| 75 | 0.587491 | 0.043507 |
| 100 | 0.576774 | 0.034270 |

The final loss did not improve on its own corrected baseline. Do not compare
these values directly to the old objective, or treat same-ASR agreement as an
independent speech-quality score. The final logged training loss is a last
microbatch metric, not an average over all training examples.

W&B sync completed for the [corrected run](https://forge.coreweave.com/wandb/weirenlan/pocket-tts-taigi/runs/01a037f62600460398dbbfd617f0b418).
Local artifacts are under `data/taigi-batch-v1/diagnostic-v5-eos/` in the original
checkout. The previous failure and comparison remain under `diagnostic-v3/` and
`diagnostic-v4/`.

The actual cloud teacher also completed 5 updates. Model, EMA, optimizer and export
tensors were finite; EOS weights changed. This verifies cloud execution, not
quality. Large compatibility checkpoint files remain on the VM pending cleanup.

## Full-data experiment and operations

The next experiment uses a fresh train-only BPE tokenizer, 24-layer pretrained
teacher with reset text embeddings, batch 1 with eight accumulation steps,
learning rate 3e-5, 100 warmup updates, and 200 updates initially. Validation and
checkpoints run every 50 updates; four raw candidates are retained. Fixed baseline
and final generation include Breeze consistency diagnostics. There is no automatic
promotion or automatic 2,000-update continuation.

Source commit 43735eb was verified against all 230 regular archived files on the
VM; the production output includes the archive and deployment hashes. Execution
uses a persistent enabled systemd service with control-group cleanup and a fixed
runtime. A local watcher incrementally copies outputs, verifies every remote
regular file after the service ends, and then stops the exact dedicated VM.
Stopped disks remain billable until verified cleanup; the OCR VM is excluded.

The original absolute stop remains 2026-10-26 15:06:12 Taipei. The USD 50 ceiling is
shared across prior and current compute/storage, not reset by retries. W&B stays
offline on the VM; no HF/W&B credentials were transferred.

The cloud initially could not access Taiwan-only Snoopy and drama sources. The
repair routes public downloads through the user's Taiwan machine using a
loopback-only reverse SSH SOCKS listener, while GPU processing stays on Nebius.
Original failed attempts remain recorded; verified source reports allow recovery
without resetting attempt history or changing the frozen corpus. Both restricted
sources were retrieved and verified. The backup watcher was
paused during the controlled repair and has resumed against the new running
`job-200-tw` orchestration job. The corpus directory, frozen plan, completed
artifacts and absolute deadline were preserved. The proxy listener is loopback-only
and stops with the verified completion helper or original absolute deadline.

## Phone previews

Telegram confirmed delivery of three MP4s: 20-second source audio, the same 20 seconds
after RE-USE, and a 10-second enhanced Snoopy clip. WAV originals and phone versions
are under `/home/weiren/tts-deliverables/2026-10-25-audio-preview/`. These are processed
source examples, not outputs from the trained TTS candidate.
