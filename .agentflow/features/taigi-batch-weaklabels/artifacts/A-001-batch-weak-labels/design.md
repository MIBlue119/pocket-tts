* _2026-10-25 18:48:00 +0800 (codex/unknown)_

# Resumable weak-label expansion

The next working increment turns the existing single-video command into repeatable
batch data production, freezes a source-grouped dataset, and trains against it.
Labels stay unverified. This does not promise useful synthesis from a tiny corpus.

## Requirements and authority

Owner: “請繼續幫我完成任務 你可以用 agentflow 的godev來達成”. Earlier messages explicitly
request yt-dlp, NVIDIA RE-USE and Breeze-ASR-26 weak labels, little human intervention,
local feasibility, Git delivery and training, within US$50 total cloud pilot spend.
The existing single-video proof is ebf708f. Continuation authorizes implementation.

- R-1: Process an explicit finite job manifest and resume completed verified jobs without repeating expensive work.
- R-2: Bound attempts, individual subprocess time, overall run time and disk admission; preserve failed attempts and diagnostics.
- R-3: Freeze accepted releases together, deduplicate exact source intervals/audio, and keep related source groups entirely in one split.
- R-4: Run actual user-supplied sources through enhancement/ASR, train on the resulting release, validate on held-out source groups and record W&B evidence.
- R-5: Commit reviewed code and instructions to the existing personal branch; keep raw media, credentials and model weights out of Git.

## Discovery and codewalk

Shared coverage: discovery and codewalk. Facts from inspected ebf708f:
`training/taigi/build_video.py` already supervises bounded individual stages, restores
signal handlers, verifies exact 20-second intervals and artifact hashes, and emits
one release. `pipeline.py` writes compact train/valid manifests plus metadata and
hash-addressed audio. Its split helper unites exact duplicate audio within one
preparation call. It does not merge separately created releases or persist a batch
queue. `training.train` accepts sentence manifests and runs validation at valid_freq.

Edit boundaries: new batch orchestration and merge modules with focused tests;
existing single-video implementation stays intact unless a demonstrated integration
issue needs a correction. CLI interfaces use argparse and separate Python model
environments. Unit tests inject external work; real local runs prove GPU integration.
Unexamined: production-scale multilingual quality and cloud GPU concurrency.

## Specification

Batch input: JSONL rows `{url, group_id, start, duration}` referencing individual
YouTube URLs; duration remains at most 20 seconds. Reuse the existing command's
source bounds rather than add a second downloader. Optional planning reads the
committed registry and emits only finite short-source jobs; long interviews are
reported as deferred, not silently claimed processed.

Persist a versioned job plan/config identity and atomic state in a caller-selected
output directory, with an exclusive flock preventing two active writers. Each job
has a deterministic key from URL and interval plus group; duplicate video intervals
must be rejected or deduplicated deterministically, including conflicting groups.
Each attempt has a new directory; success requires child exit0, ready report and
validated release manifest/audio checksums. A resumed completed job verifies saved
artifact integrity and skips inference; stale running jobs after restart become
interrupted attempts. Configuration changes refuse resume. Failures remain visible,
with capped total attempts and no unbounded loop. TERM/INT forwards cancellation to
the running wrapper and waits for descendant cleanup, escalating only if necessary.
Do not use the single-stage helper's immediate SIGKILL against the wrapper since it
owns its own process groups. Enforce an overall deadline and disk reserve at job
admission. Save an explicit batch report and return nonzero if work remains failed.

Merge input: explicit release paths, each from corrected Breeze v2. Validate
metadata against compact manifests and verify source audio hashes before copying.
Keep one exact pair for duplicate source intervals, including raw/enhanced variants;
reject conflicting text or provenance rather than choose silently. Unite groups
linked by source ID, original source hash or copied audio hash, so related variants
cannot cross splits. Stable split by connected group, with one or more explicitly
held-out groups; require both train and valid unless train-only was explicitly
requested. Copy audio into a new content-addressed release atomically; record
parent release identity, counts, durations, label provenance and split assignments.
Existing release reuse validates both manifests and audio. Never mutate input releases.

Training: keep release fixed; train a tokenizer from training text only, load the
released 24-layer teacher, use sentence mode with no target-as-prompt, and run a
bounded local experiment with actual validation. Compare validation losses as
execution/fit diagnostics, not independent language quality. Save checkpoint,
optimizer/export, source manifest hashes and W&B status. Do not spend on another VM
until local expansion proves this path.

## Invariants

- INV-1: Given an unchanged plan and verified successful job, resume performs zero external work for that job. Missing/changed artifacts fail closed.
- INV-2: Given cancellation or deadline expiry, the active wrapper and its descendants terminate, and the attempt stays visibly failed/interrupted.
- INV-3: Given duplicate/related source material across releases, no connected source group crosses train/valid; duplicates do not inflate row count.
- INV-4: Given published inputs and a fixed plan, existing release bytes remain unchanged. Output publication happens only after integrity checks.
- INV-5: Given weak ASR text, records never claim verified labels or independent intelligibility scores. Secrets/media remain outside Git and W&B payloads.

## Acceptance criteria

- R-1/INV-1: Inject counted worker calls, run twice and observe unchanged call count; corrupt a completed artifact and observe refusal.
- R-2/INV-2: Real subprocess cancellation test proves wrapper descendants exit; retries, disk floor and deadlines stop admission. A real PTY CLI journey records output, exit and persisted state.
- R-3/INV-3/INV-4: Duplicate source variants across groups are deduplicated or rejected; related groups stay in one split; mutation/missing files fail before publish; output bytes remain stable on reuse.
- R-4/INV-5: Actual finite user-video batch publishes more than the prior four rows, a grouped train/valid release, finite training/validation metrics and a complete checkpoint. No human labeling is required.
- R-5: Independent reviewer inspects the final code, tests, instructions and evidence at a named commit; remote HEAD matches delivered branch.

## Minimality check

A shell loop alone is simpler but repeats GPU work after interruption and cannot
prove merged split integrity. Two small modules reuse the existing downloader,
enhancer, transcriber, training loader and bounded training runner. No service,
database, new ML model, hot data reload or additional cloud resource is necessary.

## Trigger decisions and gates

Requirements and specification are settled from prior authorization and existing
contracts. Explore/spike: skipped, since actual single-video/model compatibility
was already proven; bounded tests address new state behavior. Security: review
local subprocess cancellation, path handling and file publication in the final
independent pass. Learn: no separate stage; retain actual failures in the notebook.
No new owner decision remains. This is reversible local implementation, not a
production deployment or irreversible data migration.

## Reopened integrity/lifecycle contract after review

Astra reproduced compact-manifest tampering with refreshed local hashes and abrupt
wrapper death with a surviving separate-session stage. Reuse must reconstruct both
compact manifests from canonical expected metadata, rather than trust refreshed
self-declared hashes. Durable per-attempt ownership tokens must be written before
launch and cleanup must find owned descendants even after the wrapper disappears.
PID descriptors prevent reused PIDs receiving signals. These refine existing
INV-2/4; they do not add a new product outcome. A controlled real pause/resume will
verify completed jobs survive while the corrected runner is adopted.

Self-check: Scope follows the existing authorized weak-label/training task; limitations and observable acceptance are explicit.

## 2026-10-25 evening delivery extension

Owner raises the total pilot cap to US$100 and allows training until tomorrow evening
if results are promising; the desired deliverable is usable Chinese-input Taiwanese
Pocket-TTS, with model artifacts uploaded to the owner's private Hugging Face repo.
The operational cutoff is 2026-10-26 22:00 Asia/Taipei (14:00 UTC), preserving the
original compute start and previous spend. This supersedes the US$50/20-hour limit.
The current full-corpus producer remains frozen and uninterrupted.

- R-6: Prepare bounded continuation after the full-data 200-update diagnostic, gated
  on explicit finite-loss, generated-audio and validation evidence. Preserve every
  stage and stop when evidence does not justify spending more. Loss or same-ASR
  agreement alone does not establish Taiwanese correctness.
- R-7: Publish a reproducible inference package to `weiren119/pocket-tts-taigi`,
  private before any upload and verified private afterward. Include the exact
  selected weight variant, tokenizer, configuration, code revision and evaluation;
  no training media, optimizer state or credentials are published.
- INV-6: New deadlines preserve the original cost clock and cap total estimated
  spend, with independent cloud/local stop controls. Extension must not silently
  remove early-stop behavior or consume the entire budget regardless of quality.
- INV-7: Model publication checks personal namespace and private visibility before
  sending artifacts; an existing public destination is refused. A successful upload
  requires remote revision and file integrity verification.
- INV-8: A continued run keeps the dataset/tokenizer fixed and records parent
  checkpoint, optimizer and EMA identity. A changed training horizon is a child
  experiment, never a rewrite of the old run manifest. Evaluated weights must match
  the published raw/EMA variant.

Acceptance: control readbacks agree on the latest cutoff (INV-6); a public-repo
fixture fails before upload and a private fixture verifies revision/file checks
(INV-7); lineage and incompatible-input fixtures prove continuation rejects changed
inputs (INV-8). Actual artifact loading and generated WAVs, followed by private Hub
verification, are required before declaring model delivery. Usability remains an
empirical outcome and is not guaranteed by the deadline.

Minimality check: reuse the current producer, trainer, evaluator and backup watcher.
A small continuation supervisor and strict artifact publisher add only the missing
lifecycle and private delivery boundaries. No new VM, database or labeling platform
is needed. Retain all 290 source outcomes; no deadline-driven silent sample limit.

## 2026-10-26 morning failure recovery

All290source outcomes are now terminal:288successful and2region-restricted failures.
The exact parity gate correctly stopped before training because the converter's
Metaspace pre-tokenizer split adjacent whitespace markers before learned BPE merges.
The provider stop request also failed after its federation login expired. Existing
owner authorization covers repairing both failures and continuing within the same
USD100 total and22:00Taipei deadline; no budget clock is reset.

Recover with a minimal converter correction, proving the original22763parity cases
without changing the learned model or source transcripts. Keep all failed job and
backup records intact. A new production subtree `recovery-20261026` holds a fresh
parent job, continuation, backup receipt and publication attempt. The original
corpus and immutable outputs are reused; existing verified successes must perform
zero new download, enhancement or ASR attempts. No source failures are hidden.

The new subtree lets the existing selection-path contract and backup boundary work
without another feature or a rewrite of failed state. Required checks: converter
regression red/green, complete real parity, frozen corpus code identity unchanged,
correct service targets and stop-post containment, unchanged source-attempt counts,
and observed actual training progress. Stop-control fallback must never describe
SSH shutdown as provider-confirmed stopped or verified billing cessation.

Recovery exposed a second concrete defect before training: input validation used
`dict.setdefault(path, sha256_file(path))`, eagerly hashing the same full audio
for every segment; existing merged releases also rehashed every row. Use a cache
local to each validation call, keyed by resolved path, while still comparing every
row's expected digest. Preserve containment, bounds, manifests, metadata and copy
verification. Real metadata has22754rows referencing421paths; cache reuse removes
redundant hash calls, not integrity checks. Pause backup/publication before stopping
the pretraining parent, preserve that attempt, and restart in fresh
`production/recovery-20261026-r2`. No trained checkpoint is discarded, no corpus
identity file changes, and original budget/deadline remain in force.


## Morning continuation diagnostic correction — 2026-10-26

The actual baseline contains six explicitly recorded decoder-too-short failures, which intentionally have no audio or ASR measurement. Preserve the original reports; derive a labelled zero task-failure score only for this exact validated branch. Missing or contradictory diagnostics remain fail-closed. All finite-state, health, cap, repetition, deadline and improvement thresholds remain unchanged. Optional relative selection paths support fresh recovery subtrees while retaining verified-root containment and artifact hashes.

Minimality check: changing raw ASR reports or weakening the gate would misrepresent evidence. The smallest correction is a derived score with source labels and focused tests; the selection option avoids moving historical candidates. Actual step200 subsequently showed7/12caps versus2baseline, so no v2 continuation is launched and the original selection remains the publication target. Source correction does not authorize ignoring this generation regression.


## Owner-authorized longer adaptation — 2026-10-26

Owner instruction:「還是要訓練久一點」. Continue the verified step200 model, optimizer and EMA through1000,3000and6000 in separate immutable run directories. Keep the exact tokenizer, data split, batch1/accumulation8, constant3e-5learning rate and all other training parameters. Use available local RTX3090 because Nebius API authentication expired and the previous instance received shutdown after verified488-file backup. No paid instance or budget clock is reset.

The former early quality-improvement admission rule is not applied to this explicitly authorized exploration. Every endpoint still receives fixed-prompt evaluation and remains experimental; missing artifacts, nonfinite state or validation loss above1.5times the original baseline stop progression. Preserve absolute2026-10-26 22:00Taipei cutoff with1800seconds reserved for evaluation/export. No automatic training beyond6000. First1000 has a100-minute process bound and cgroup containment; actual resumed updates and memory use must be observed before claiming launch success.

Minimality check: reuse fork_run, run_experiment, evaluator and raw export instead of changing the trainer or rebuilding data. A separate small controller is needed because the old200-step pilot controller intentionally rejects the new exploratory policy and waits on a cloud-only parent service. Parent outputs stay immutable. Optional later Nebius restart needs authentication, but does not block local progress.

Invariants: INV-L1 preserves checkpoint/tokenizer/data hashes and optimizer/EMA restoration; any mismatch fails. INV-L2 preserves absolute time/budget and unrelated GPU processes; no additional paid compute is started. INV-L3 separates exploration from quality promotion; generated output and provenance remain truthful even when quality is poor.

Acceptance: real local training resumes at201 with LR3e-5 and finite gradients; automatic stages terminate at6000 or an explicit failure/time bound; all completed evaluation/checkpoint artifacts remain reviewable and private delivery never claims usability without evidence.


## No-reference inference prefix correction

The ongoing usable-model/private-delivery task authorizes correcting a reproduced inference-contract bug: generated inference.py builds an empty public state without the bos_before_voice prefix that training and the native evaluator use. Correct only the packaged no-reference branch, preserve optional-reference behavior and all sampling settings, and test actual prefix/state behavior. Existing published candidate bytes and the live training/evaluation code stay unchanged. The user's step1000「你好嗎？」sample came from the correctly prefixed native evaluator; this correction must not be represented as fixing that sample's poor speech.

Minimality check: prefill the existing public flow helper with the BOS embedding when configured, instead of changing the model API, adding artificial reference audio or restarting training. Verify the generated script in a fresh local package copy onCPU and record its provenance; new candidates will carry the corrected inference script. Existing private200candidate remains historical.
