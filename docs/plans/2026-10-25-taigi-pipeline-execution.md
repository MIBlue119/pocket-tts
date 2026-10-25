# Start the direct Chinese-to-Taigi training pipeline

## Done condition

Prepare versioned Chinese semantic text and real Taiwanese audio pairs, run a bounded teacher training smoke with a complete checkpoint, verify restoration and W&B metric delivery, and push reviewed code to the personal fork. A smoke run proves execution, not useful Taiwanese synthesis quality. Cloud pilot spending is capped at US$50, authorized by the user on 2026-10-25.

## pstack workflow

- [x] Read the Principles section of poteto-mode.
- [x] Phase A: Frame. Inspect repository, credentials, hardware, and existing research.
- [x] Phase B: Design the workflow. Reuse the researched sentence-level design. A second architecture bakeoff is skipped because the direct-task design was already compared with the cascade by the preceding explorer team.
- [ ] Phase C: Run the loop. Implement and verify each owned unit.
- [ ] Phase D: Keep the audit trail. Append actual decisions and evidence.
- [ ] Phase E: Verify and hand back. Inspect real data, training events, checkpoint restoration, tracker, and published branch.

## Work partition

| Owner | Separate worktree | Deliverable |
| --- | --- | --- |
| Astra sentence worker | pocket-tts-sentence | Explicit sentence loader, full text and audio, optional independent prompt, true zero prompt, behavioral tests |
| Sol data worker | pocket-tts-data | Local audio and bounded YouTube preparation, Breeze pseudo-labels, provenance, grouped manifests, offline tests |
| Astra runner worker | pocket-tts-runner | Bounded experiment launcher, W&B adapter, run identity, strict checkpoint validation and snapshots |
| Primary agent | pocket-tts | Integration, environment, source acquisition, GPU smoke, Nebius, review and personal-fork delivery |

## Integration contracts

- Training JSONL has `path`, `start`, `duration`, `transcript` with whole Chinese semantics. Optional `prompt` holds a different same-speaker audio interval.
- Sentence configs use `data.sampling_mode: sentence` and `data.precompute: false` for the initial raw-audio path. Existing aligned mode remains the default.
- Missing prompt means zero valid voice frames, not a sampled copy of the target. No fake `words` timestamps.
- Preparation distinguishes source-declared Taiwanese and ASR pseudo-labels from verified language or text. An explicit train-only smoke does not invent a validation split.
- The runner owns lifecycle and tracker integration outside the training loop. W&B receives metrics and sanitized identifiers, not corpus files or credentials.
- Use full revision and hashes for models, dataset manifests and configuration. Keep credentials outside Git.

## Sequence and gates

1. Capture existing loader/config tests and create separate worker branches.
2. Verify sentence loading and bounded launcher against offline fixtures.
3. Acquire a small real speech sample, transcribe with the pinned Breeze revision, and inspect generated manifests and exclusions automatically.
4. Train a Chinese tokenizer and generate a teacher smoke config. Run a few steps on the local RTX 3090 before cloud spend.
5. Inspect finite metrics, changed model state, complete model/optimizer checkpoint and restoration. Confirm W&B run and metrics remotely.
6. Inspect Nebius project and available capacity. Start a bounded cloud pilot only with a stop mechanism and estimated total within US$50. Preserve prior unrelated resources.
7. Run applicable tests, lint/type checks, independent review; commit and push to `MIBlue119/pocket-tts` on `feat/taigi-training-pipeline`.

## Resource limits

- Cloud total pilot ceiling: US$50, including pilot compute and storage allocation during the run. Record actual resource rates and a conservative deadline before provisioning.
- Local device: RTX 3090, 24 GiB, used sequentially for ASR and training.
- Initial acquisition is bounded to small samples, not entire channels/playlists.
- No production deployment or public model/dataset release is part of this execution.
- Existing Nebius CLI profile is named `newtaipei-ocr`; its project must be inspected before creating dedicated TTS resources. Do not alter existing OCR instances or disks.

`throughput checkpoint: three isolated implementation workers; primary integrates and verifies actual training, no duplicate GPU jobs.`

Decision trail: [execution-decisions.tsv](2026-10-25-taigi-execution-decisions.tsv).
