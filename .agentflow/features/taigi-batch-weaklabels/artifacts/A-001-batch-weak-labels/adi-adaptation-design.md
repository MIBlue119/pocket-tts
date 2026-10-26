# Fresh ADI data-adaptation fork

## Outcome and scope

Add a CPU-only configuration helper that forks a completed, identity-verified Taigi
training run onto one frozen merged weak-label release. The helper copies the latest
complete model/EMA checkpoint and optimizer sidecar, preserves tokenizer, model
architecture and Mimi configuration, and changes only the new run directory, endpoint,
data manifests, and explicitly requested recipe knobs. It does not launch training,
allocate a GPU, promote a model, or claim that any recipe or learning rate wins.

The CLI accepts `--parent-config`, `--release`, `--child`, and an endpoint above the
parent step. Optional explicit comparison knobs are batch size, gradient accumulation,
flow batch multiplier, learning rate, and precompute mode. Omitted knobs inherit the
parent exactly.

## Approach

1. Reuse `continue_taigi_training.verify_parent`, `verify_pair`, `run_lock`, launcher
   identity generation, Taigi JSONL parsing, and the merged release's frozen report.
2. Validate both nonempty manifests, their report hashes/counts, metadata correspondence,
   finite positive intervals within real audio bounds, referenced audio hashes, and
   train/valid separation by audio and available source/group identities.
3. Under the completed parent run lock, reverify the parent and checkpoint pair, copy the
   pair byte-for-byte, derive the child config, and reject any config diff outside the
   explicit allowlist.
4. Copy the frozen train/valid manifests into the child so optional latent precompute cannot
   write beside the source release. Record parent, checkpoint, optimizer, tokenizer, merged
   release and resolved recipe identities in lineage. The launcher identity hashes the copied
   manifests, unchanged tokenizer and lineage; the hashed lineage carries the parent pair,
   parent config and release report/manifests provenance without depending on an initial
   checkpoint that normal retention may later prune.

## Invariants

- **INV-1 — Parent immutability.** Starting condition: the parent config, recorded inputs,
  and latest checkpoint/optimizer pair verify. Preserved guarantee: the helper never edits
  them and copies the pair byte-for-byte. Failure condition: any parent identity mismatch,
  missing/orphan sidecar, invalid tensor, or copy hash mismatch aborts the fork.
- **INV-2 — Token/model continuity.** Starting condition: the parent identity includes its
  tokenizer. Preserved guarantee: tokenizer path/hash, token IDs, model config/overrides,
  architecture, Mimi settings, embedding-reset setting, checkpoint model/EMA state, and
  optimizer state remain unchanged. Failure condition: their config paths differ or their
  recorded files change.
- **INV-3 — Frozen, separated data.** Starting condition: a merged weak-label release has a
  matching immutable report. Preserved guarantee: train and valid are nonempty, schema-valid,
  bounded by real files, hash-valid, and disjoint by exact audio and source grouping where
  metadata supplies it. Failure condition: any report/manifest/audio mismatch, bad row, empty
  split, missing audio, or cross-split identity overlap.
- **INV-4 — Deliberate experiment identity.** Starting condition: the endpoint exceeds the
  parent checkpoint step. Preserved guarantee: only run/data/endpoint plus explicitly passed
  recipe fields change, and all actual diffs and provenance hashes are recorded in a fresh
  child. Failure condition: an unallowlisted config diff or changed provenance is observed.
- **INV-5 — No quality or resource action.** Starting condition: weak labels remain
  unverified. Preserved guarantee: the helper only prepares files, records no quality/LR
  winner, and starts no trainer, GPU, cloud job, publication, or budget action. Failure
  condition: it launches work or marks the child quality-approved.

## Acceptance criteria

- **AC-1 (INV-1, INV-2):** A small real Torch checkpoint/optimizer fixture forks with
  identical copied bytes and preserved EMA, tokenizer, architecture and optimizer state;
  altered tokenizer, parent manifest, nonfinite/missing/orphan pair, or same endpoint fails
  before a usable child is created.
- **AC-2 (INV-3):** A coherent merged release with real WAV files passes. Empty/malformed
  manifests, changed manifest bytes, missing/out-of-bounds audio, and train/valid exact-audio
  or source-group overlap fail.
- **AC-3 (INV-4):** The generated YAML differs only at the declared allowlist; an injected
  tokenizer/model change is rejected. Lineage records hashes for the parent config/identity,
  copied pair, release report/manifests, and tokenizer; `run_manifest.json` hashes the copied
  manifests, tokenizer, and that complete lineage.
- **AC-4 (INV-4):** Re-running the launcher identity check against unchanged child files
  matches, while changing a copied manifest, tokenizer, or lineage rejects identity. Parent or
  release tampering before the fork is created is rejected by the source verification gates.
- **AC-5 (INV-5):** CLI help and a CPU-only fixture invocation succeed without importing or
  launching the trainer; output states only the created config path.

## Minimality check

Keep `continue_taigi_training.verify_parent` unchanged. Add one purpose-specific script and
one focused test module; do not add a generic validation framework, alter dataset construction,
or modify GPU/job/publication code.

## Dictionary semantic release amendment

The ADI adaptation data is now an exact MOE dictionary-parallel weak-label release, not an
ASR-pseudo-label release. Build it by joining every admitted mapping to the already frozen ADI
candidate by normalized original Common Voice path plus sentence ID and verifying the native
reference. Preserve the prepared train/valid split, then give valid precedence when one normalized
Mandarin target would otherwise occur in both splits. The optional `diagnostic-64` selection writes
the 64 diagnostic cases only to a quarantined control manifest; controls never become acoustic
training gold. Copy audio and source evidence into the immutable child release, preserving the ADI
license, dictionary source bytes, mapping bytes, and their hashes.

- **INV-6 — Exact semantic join.** Every admitted source path resolves to exactly one prepared
  candidate whose sentence ID and native reference match the mapping. Missing, duplicate, ambiguous,
  or tampered mapping/provenance aborts the release.
- **INV-7 — Split and target isolation.** Prepared split assignments are retained. Exact audio,
  source group, native reference, and normalized Mandarin targets remain disjoint; valid wins any
  Mandarin-target collision and the excluded train rows are counted.
- **INV-8 — Self-contained licensed release.** Published audio is a regular in-release read-only
  copy, never an external symlink. ADI and dictionary source evidence is byte-identical and hashed.
  Labels use schema `taigi-dictionary-semantic-pairs/v1`, type `dictionary_parallel_weak`, and
  verification `dictionary_matched_not_audio_semantics_verified`; they are never described as
  Breeze/ASR labels or quality-promoted.

- **AC-6 (INV-6):** Real WAV/prepared-report fixtures build; changed join keys, mapping bytes,
  dictionary hash/license/proof, and post-build artifact tampering fail verification.
- **AC-7 (INV-7):** A shared Mandarin target across train and valid removes only the train row and
  records the valid-precedence exclusion; diagnostic controls remain outside train/valid.
- **AC-8 (INV-8):** Copied audio and evidence hashes match their sources, and the adaptation fork
  accepts this explicit dictionary schema while continuing to enforce the existing Breeze branch.
