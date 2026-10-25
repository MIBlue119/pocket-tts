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

Self-check: Scope follows the existing authorized weak-label/training task; limitations and observable acceptance are explicit.
