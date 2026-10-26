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

## Overnight continuation and private model delivery

The owner raised the **total** cap to USD100 and requested a usable Taiwanese
Pocket-TTS candidate by tomorrow evening. The operational interpretation is a hard
cloud cutoff of **2026-10-26 22:00 Asia/Taipei**, with90minutes reserved before it
for final artifact handling. The original compute start is unchanged. The recorded
conservative bound is USD56.114 including prior estimated spend and reserve; this
is an estimate, not an invoice or a commitment to spend the entire allowance.

The original full-corpus producer remains frozen at43735eb and running. A separate
persistent `taigi-training-continuation.service` waits for its200-update result,
then conditionally admits cumulative500,1000and2000-update stages. A single
undercoverage bridge is allowed under finite/non-regression guards; later stages
require joint directional validation and same-ASR evidence. These are spending
heuristics, not proof of correct Taiwanese. Each child preserves dataset/tokenizer
identity and restores model/optimizer/EMA, retaining the old run manifest. The
selected export is explicitly raw and matches its diagnostic checkpoint.

Deployment preflight exposed missing Git metadata in the cloud source archive.
Real metadata for43735eb was prepared separately and installed atomically without
rewriting any working source: all231tracked file hashes and modes were unchanged.
This also fixes the existing training launcher's otherwise-failing Git lookup.
The extension commit65490c4 and exact controller hash are recorded separately in
`production/continuation-deployment.json`; cloud Git accurately reports the frozen
base plus an untracked extension.

The backup watcher now follows the continuation service/job. Synchronous privileged
`ExecStopPost` stops the original producer before the umbrella fully exits, so a
failed controller cannot leave a writer running during final backup. A real dummy
parent/failing-guard fixture proved this ordering; the real producer stayed active.
At20:14Taipei the producer still had its originalPID21133 and11complete/1running/
278pending sources. Both independent deadline controls, the Taiwan proxy, backup
watcher and continuation service were active.

The private repository is **https://huggingface.co/weiren119/pocket-tts-taigi**.
Private visibility and README bytes were verified at revision
`1a208143b5486493ee6dabef6f0402f0292800bc`. No model weights have been uploaded yet.
Local `taigi-private-publish.service` waits for the final verified backup, rehashes
selected files, packages only inference artifacts/metadata, runs actual CPU
inference, and uploads to a versioned candidate directory. It refuses public repos,
checks raw/EMA/tokenizer identity, rejects empty/nonfinite tensors before PCM
conversion, and verifies remote files at the returned commit. HF credentials stay
local. Deadline cleanup terminates the publisher's entire process group.

Root checks:13continuation tests,25publisher tests and4readiness tests passed;
three fork tests were rerun after a type-only config update; scoped Ruff and type
checks passed. Independent review covers privacy, identity and process lifecycles.
Real packaging of the earlier diagnostic weights produced a2.4-second24kHz GPU
WAV. The final bounded CPU smoke helper produced a7.04-second24kHz WAV and retained
a maximum-generation warning from that known-poor tiny-data model. This establishes
runtime execution, not usable speech; that diagnostic package was not uploaded.
The future selected candidate receives its own runtime check before publication.

Remaining: finish all source outcomes, train/evaluate the full-data candidate,
apply continuation gates, verify final backup, privately publish the selected
model, sync offline W&B records and clean up the dedicated cloud resources.

## October26 morning recovery

At07:12Taipei all290sources had terminal outcomes:288completed and2explicit
regional-access failures. The resulting22754weak pairs contain17.9897audio hours.
The full-data parent stopped before training at03:57because four of22763tokenizer
checks differed on consecutive spaces. The guard correctly prevented inconsistent
IDs from entering training. Metaspace's default splitting prevented learned
whitespace-only BPE pieces from merging; commit1485064disables that splitting.
The original learned tokenizer and all text stayed unchanged during verification:
22763/22763production checks and3304additional whitespace cases passed independently.
The focused regression and offline tokenizer tests pass; cached shipped-model tests
were unavailable, and installed SentencePiece stubs prevent a clean scoped type check.

The final old backup was verified at04:38, but API stop failed because the Nebius
federation login expired. The VM remained idle until morning recovery. A guarded
SSH shutdown fallback now verifies the exact dedicated instance and can request
poweroff when due; it never claims provider-confirmed STOPPED or billing cessation.
The independent cloud22:00timer and original budget clock are unchanged. API
confirmation/cleanup still requires renewed login.

Fresh recovery output lives under `production/recovery-20261026`. Old failures and
verified-backup receipts remain intact. The original frozen corpus is reused; only
the converter source changed on cloud, and its complete parity check passed there
before restart. Parent/continuation services restarted at07:18; backup and private
publisher now follow the new subtree. Full-data training and useful speech are not
yet claimed by this recovery record.

While observing recovery, a second defect was reproduced in merge validation:
`setdefault(path, sha256_file(path))` eagerly rehashed shared full audio for every
segment; existing merged releases also hashed per row. Commit979ab7auses a cache
local to each validation pass and still checks every row's expected digest. All16
merge tests pass, including conflicts and changed-file detection. Actual metadata
implies22754audio hash calls become421per phase (about54times fewer calls), without
claiming a measured end-to-end speedup. Input/copy/manifest integrity remains checked.

Before training began, backup/publication were paused and the old umbrella/parent
were stopped with bothMainPID/ControlPIDzero. No checkpoint existed to discard.
The preserved first recovery attempt records this interruption. The corrected run
uses `production/recovery-20261026-r2`, with fresh parent/continuation and retargeted
local backup/private publication. Its deployment receipt binds both converter and
merge source hashes; none of the eight frozen corpus identity files changed.


### Full-data result on 2026-10-26 morning

The fresh r2 run completed200updates at07:35Taipei using22606training and148validation rows. Exact semantic row checks and unchanged871source attempts establish zero repeated downloading/enhancement/ASR during recovery. Validation loss fell from0.5278602999 to0.5000719985 (5.27%); step100 was0.4990983266. These are optimization diagnostics, not speech-quality evidence.

All12final prompts produced finite audio, but7hit the125-frame/10-second limit (baseline2). ASR repeatedly returned unrelated phrases, including「你給我回來」across different inputs. The candidate is unsuitable as a usable Chinese-to-Taiwanese model. Conditional continuation must not proceed on loss alone. The original200-step raw candidate is retained for experimental private delivery after verified backup.

A separate source fix records explicit decoder-too-short baseline failures as labelled zero task-consistency scores while retaining absent raw ASR measurements. All other missing/invalid cases and existing quality thresholds remain fail-closed. Root42focused continuation/readiness tests, Ruff and implementation typing pass; Astra independently reviewed the change. No replacement cloud controller was launched because the actual generation regression already prevents admission. Local backup/private-publisher services are rearmed against the original r2 selection. Provider API authentication remains expired; SSH shutdown acknowledgement must not be described as provider-confirmed STOPPED.


### Owner-requested longer continuation, 2026-10-26 09:55

The owner explicitly requested longer training after the200-step result. The updated exploratory policy preserves the same recipe and resumes model/optimizer/EMA through1000,3000and6000cumulative updates. Initial generation regression is not used as an admission gate for this authorized exploration. All results remain experimental; finite-state, complete diagnostics, stage-boundary divergence and22:00deadline guards remain.

The prior recovery backup was verified against488remote files. Exact-instance SSH shutdown was acknowledged08:38; providerSTOPPED cannot be confirmed because Nebius federation authentication expired. No new paid instance was started. LocalRTX3090 successfully resumed at201at09:55:39,LR3e-5; first20updates have finite losses/gradients and totalGPUusage about8GiB. Evidence: data/taigi-full-corpus/local-longer-20261026/resume-proof.json. First1000userunit has a6000-second trainer limit and cgroup containment.

The200-step raw candidate was uploaded after fixing the local publisher's missing credential-file environment. Actual HFAPI inspection confirms private=true and revisionbec7e32ccc5dbd6d3330954a531adb8999d4a698, with all8candidatefiles under candidates/step-200-raw, including1344708416-byte weights. The publisher also verified each remote file hash and runtime execution. This is experimental archival delivery, not a speech-quality pass. Earlier failed publication outputs are retained.
