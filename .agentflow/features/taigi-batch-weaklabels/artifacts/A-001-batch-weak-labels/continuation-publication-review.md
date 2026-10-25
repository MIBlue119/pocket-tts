# Continuation and private publication review

Reviewed 2026-10-25, Asia/Taipei. Scope: conditional training within the approved USD 100 total budget, followed by private delivery; no remote actions or source changes performed by this reviewer.

- **No remaining code blocker found** in the final continuation, publisher, and ready-watcher sources. Two material findings were corrected: nonfinite fallback candidates are refused, and packaged inference defaults to zero voice prompt instead of silently requiring voice cloning.

- **Runtime containment conditions are satisfied by root's reported final readbacks:** the installed umbrella contains descendants, synchronously stops the original producer before final backup, and the backup watcher now follows the umbrella. This reviewer did not independently repeat remote checks.

- **Delivery remains experimental.** Exact raw checkpoint identity is preserved, but the packaged streaming example is not claimed to reproduce the offline evaluator or establish Taiwanese speech quality. No upload or complete production continuation was executed during this review.

## Identity and continuation

The controller verifies the parent configuration, dataset manifests, tokenizer, and recorded inputs before creating a fresh child directory. It copies and hashes the checkpoint and optimizer sidecar, preserves EMA in the checkpoint, and changes only run_dir and max_steps. The child gets its own identity and a lineage file; the parent identity is not rewritten. Data-loader randomness restarts, and this limitation is explicitly recorded.

Evaluation admission checks the raw checkpoint, tokenizer, configuration, fixed prompt order, generated audio hashes, and pinned Breeze model/revision/adapter. Continuation requires finite diagnostics and the specified validation/text improvement, with the bounded step-200 undercoverage exception. Selection now excludes all nonfinite candidates; an unhealthy but finite diagnostic may still be delivered with quality_verified=false.

The selected model is loaded from raw checkpoint state and exported with ema=None. Provenance binds the exact source checkpoint and exported safetensors hashes. Publisher rejects conflicting raw/EMA or checkpoint/tokenizer identities and incomplete evaluations. CPU metadata validation requires the complete runtime tensor names and shapes. It does not independently prove arbitrary caller-supplied export derivation; this workflow establishes derivation in the reviewed controller.

## Lifecycle and budget

The controller waits through polling failures and live parent states. Its imported completion predicate requires a loaded inactive/failed service, Restart=no, zero MainPID and ControlPID, and no remaining cgroup processes. The original producer lives in another cgroup: checking the continuation unit alone is insufficient without the integration below.

Root's proposed unit uses KillMode=control-group, Restart=no, and synchronous ExecStopPost=+/usr/bin/systemctl stop taigi-full-pipeline.service. Root reported a real remote disposable-service test: the dummy parent remained active during the initial stop-post window, then exited before ControlPID became zero. The real production PID 21133 remained active. This supports the unit pattern; it is not a readback of the final production unit.

Before arming, root must validate the required deployment receipt and remote Git commands, install/read back the stop-post containment, and retarget the local backup watcher to the continuation umbrella. A startup error in this fail-closed design deliberately stops the original producer. Casual umbrella restarts are therefore unsafe while the corpus is active.

The fixed deadline remains 2026-10-26 22:00 Taipei (14:00 UTC; epoch 1793023200), with the original accounting start unchanged. Training chunks are at most one hour, admission preserves a reserve, and external API/remote timer shutdown stays independent. An outer subprocess timeout intentionally fails the umbrella instead of exporting while nested descendants might remain alive; earlier candidates stay on disk for backup/manual recovery. A normal inner launcher return of 124 permits the documented prior-candidate fallback after supervised cleanup.

## Backup and private delivery

The local ready watcher waits for state=complete and backup=verified for the exact production directory, then verifies all five selected file paths remain inside it and match their recorded hashes. Publication runs locally, with a bounded deadline; the VM receives no HF credentials. A deadline without verified backup cannot authorize publication.

The publisher packages exactly model.safetensors, config.yaml, tokenizer.json, provenance.json, evaluation.json, README.md, inference.py, and manifest.json. It refuses extra entries/symlinks, checks metadata for credential fields/tokens, removes external model/tokenizer fallbacks, and resolves packaged files in the example. No optimizer, source audio, or training directory is uploaded. Existing public repositories are refused; new repositories are explicitly private in the authenticated user's personal namespace. Candidate paths cannot overwrite an existing version, upload uses the expected parent commit, and the returned immutable revision is checked for privacy, file sizes, and hashes.

The inference example now uses zero audio prompt by default. Optional --voice is explicit and described as experimental. Sampling settings and streaming-versus-offline differences are documented; portability requires the matching pocket-tts checkout and dependencies, whose source provenance is included by the controller. This is not a standalone dependency-free model application.

## Verification evidence

This reviewer read final source, relevant runtime/export/cgroup helpers, and the focused tests; no suite was redundantly rerun. Writer-reported results are distinguished from independent execution:

- Publisher author: 14 focused CPU tests pass, scoped Ruff/type checks pass, and an existing 1.3 GB export's complete tensor header matches the runtime architecture on CPU/meta. Tests include private/public handling, integrity failure, allowlist, zero/optional voice helper calls, and actual WAV creation using API spies.

- Continuation author: 35 tests pass (13 continuation, 8 corpus-job, 14 cloud-watch), with scoped formatting/lint checks. Tests use real CPU checkpoint/optimizer/EMA files and raw safetensors export, including raw value 2 versus EMA value 999, and verify the five-file selection hashes.

- Root: three ready-watcher tests pass; disposable remote stop-post containment test passes. Production package/runtime proof is separate work owned by root and is not asserted here.

## Reviewed source hashes

- `training/scripts/continue_taigi_training.py`: `7960ac5464bd9608364ffef8729d287b142b43b2dffefc6bc4258d28060544aa`

- `training/scripts/publish_taigi_model.py`: `c0d8646a8bd37237daccee75f73b69de50d8a4dac4bc188c8c193b3377f43afc`

- `training/scripts/publish_taigi_when_ready.py`: `516d2e89c51ad210af39213a3c9d72eb0ee4d53259f2d6ee5e7eef1ebea12f7b`

## Type-check follow-up

Root changed the freshly loaded configuration dict using config.update(run_dir=..., max_steps=...) instead of constructing an equivalent replacement, and annotated ready_selection as dict[str, Any] | None. Independent reread confirms the same two configuration values and no parent-file/identity mutation. Root reports 13 continuation and 17 publisher/ready-watcher tests passing after the changes, with type/lint checks passing. No redundant suite rerun was performed. Source hashes above include the subsequent smoke-helper review below.

## Final runtime-smoke and containment follow-up

**No remaining blocker found after independent reread.** Publication now requires an actual selected-package CPU inference run before upload. The helper validates the package, disables CUDA, runs the packaged zero-prompt example with a maximum 300-second timeout, and records output/log/manifest hashes, sample rate, duration, RMS, and warning excerpts outside the eight-file package. Empty output, nonfinite output, nonzero exit, and timeout fail publication. A finite WAV is an execution check only: silence or a maximum-generation warning is recorded, not treated as proof of speech quality.

Review reproduced a material PCM conversion pitfall using actual soundfile in memory: a default WAV write of [NaN, Inf, 0] decoded to finite [-1.0, 0.999969482421875, 0.0]. The author corrected the packaged example to reject empty/nonfinite generated tensors before writing WAV. Tests execute the packaged script with finite, empty, and NaN/Inf tensors for both zero-prompt and optional-voice paths and require invalid output to fail before a WAV exists.

Review also identified that killing only the publisher at the outer deadline could orphan its inference child. Root corrected the ready watcher to launch a new process session and terminate its whole process group in finally, using TERM followed by KILL and wait. Independent reread confirms the inference child inherits that group. Root reports a real descendant test passing: a 120-second sleep child is terminated after a 0.5-second publisher timeout. The operational service cgroup remains necessary for external service termination.

Root reports final checks: 25 publisher tests and four ready-watcher tests pass; all three scripts pass scoped type/lint checks. This reviewer did not repeat those suites. A fresh real-package CPU smoke was still executing at the time of this follow-up; its eventual success and waveform quality are not asserted here.

Root's final runtime readback reports the continuation umbrella active at PID 26787 and waiting for the parent, while original producer PID 21133 remains active (11 complete, one running, 278 pending). The installed unit has synchronous privileged stop-post with ignore_errors=no, KillMode=control-group, and Restart=no. Required deployment preflight passed; the controller source commit is 65490c420bbb9efdcf04436410b09ae32ac09660. Root reports that genuine Git metadata recovery preserved hashes/modes of all 231 tracked files. The local backup watcher was stopped with both PIDs zero, its old receipt preserved, then restarted against the umbrella/new job with deadline 1793023200. These root-reported integration readbacks satisfy the lifecycle conditions above; no additional remote inspection or mutation was performed by this reviewer.
