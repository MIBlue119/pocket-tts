* _2026-10-25 20:03:21 +0800 (gpt-6-astra/inherited)_

# Bounded continuation toward a usable Chinese-to-Taiwanese model

**Decision**

- Keep the running full-corpus job unchanged. Prepare a separate continuation controller that waits for its actual completion, evaluates the 200-update candidate, and admits at most three bounded continuations ending at cumulative updates **500, 1,000 and 2,000**.

- The first 200 updates can be an undercoverage diagnostic when new text embeddings and a much larger corpus are involved. Permit one controlled bridge to 500 when coverage and non-regression checks pass; do not require useful speech after only 200 updates, and do not run the remaining steps unchecked.

- Total authorization is now **US$100 including prior spending**, with the coordinator's operational interpretation of the owner's ‘tomorrow evening’, with hard cutoff **2026-10-26 22:00 Taipei / 14:00 UTC**. Reserve the final 90 minutes for evaluation, export, verified backup and private upload. A usable model by then is a goal, not a supported prediction: the corrected small pilot terminates normally but still does not follow Chinese reliably.

## 1. Preserve current work and transfer shutdown ownership safely

1. Leave the current corpus/training unit, frozen source and plan intact. Do not modify the source files included in its code identity while it is running. New controller code should live in a separate frozen deployment or otherwise avoid changing the active producer's imported implementation.

2. Before the current unit can finish, stop only the **local backup watcher**, verify it and any already-started stop helper have exited, and preserve its receipt. Keep the independent existing budget watchdog armed during this transition. Otherwise that watcher can stop the VM as soon as the 200-update unit ends, racing the continuation controller.

3. Install and start a **persistent, enabled umbrella service** with `Restart=no`, `KillMode=control-group`, bounded `TimeoutStopSec`, and a runtime derived from the absolute deadline. It initially waits without loading any model. The existing unit is a separate cgroup: the umbrella must explicitly verify its terminal state and empty cgroup before admitting continuation, not assume its own cgroup contains the producer.

4. Repoint the local mirror/backup watcher to the umbrella's unit and durable job JSON, covering the output root containing both the original job and all continuation jobs. Use a new receipt path or archive the old receipt. Keep the source output root free of tokens, cloud ledgers and model-cache secrets. Verify a real short lifecycle fixture for waiting, completion, failed/stale JSON, empty cgroups and exact-instance stopping before relying on it overnight.

5. Update both independent budget controls to the newly authorized cutoff only after recording the US$100 ledger and the handoff state; verify the remote timer and local guard agree. The billing clock still starts at original VM creation, not at this extension. Retain an earlier stop if measured costs plus storage/transfer reserve would reach US$100 first. The previously recorded hourly ceiling is not a new price check; use the actual resource receipts/rate. No second VM is required.

6. If the original unit fails, remains active, or lacks a complete corpus/checkpoint, the umbrella records a failed/incomplete outcome and does not launch training. It completes its report so the watcher can verify available artifacts and stop the VM. The separate hard deadline remains effective even if backup or upload fails.

## 2. Immutable child-run continuation, not an edited pilot

The inspected `run_experiment` launcher hashes the entire YAML and input/provenance files. Changing `max_steps` in the existing run is correctly rejected. `run_taigi_corpus_job` also refuses an existing job directory. Do not delete or rewrite the pilot's identity to evade either check.

- Add one small explicit **fork-from-checkpoint preparation operation** for a new child directory. It must require a stopped parent, acquire the parent's launcher lock, use the existing checkpoint-pair/snapshot checks, and copy the latest model checkpoint and exactly matching optimizer sidecar. Checkpoint payloads include step, raw model state and EMA shadow. Record parent run identity, config/tokenizer/data hashes, checkpoint/optimizer hashes and the exact inherited step in a child lineage receipt.

- Generate a new child config and launcher manifest using the normal identity function. Permit only the new `run_dir` and increased `max_steps` for this continuation; keep tokenizer bytes and size, train/valid manifests, model config, all model/flow settings, optimizer type/rate, batch/accumulation, warmup and validation settings identical. Include the immutable lineage receipt as provenance. This is a new authorized identity derived from a verified parent, not an alteration of the parent's evidence.

- Start the ordinary bounded launcher against that prepared child and prove it restores the expected step, optimizer state and EMA. The present constant learning-rate schedule is computed from cumulative step; keeping warmup 100 avoids restarting warmup at each child. A changed tokenizer or dataset requires a fresh adaptation experiment, not this continuation path.

- Keep `3e-5`, microbatch 1, accumulation 8, EOS fix, LSD and the same architecture. Do not switch to `init_from` as a shortcut: loading model weights alone is not restoration of optimizer/EMA training state. The trainer reseeds its data loader using the resume step and does not store the precise loader/RNG position; claim stateful continuation, not bit-identical continuation of an uninterrupted run.

- Only the newest checkpoint has an optimizer sidecar. Older retained raw checkpoints can be evaluated, but cannot be selected for exact optimizer continuation unless their matching sidecars were separately preserved. Continue from the verified latest complete pair; preserve best raw candidates separately for delivery/evaluation.

## 3. Executable gate and receipt

At each boundary, write an atomic JSON receipt with these fields:

- `stage`, `parent_step`, `candidate_step`, `decision` (`bridge`, `continue`, `stop`), `reasons`, `deadline_utc`, and measured/reserved budget.

- `identity`: config, source commit, release, train/valid/tokenizer hashes; checkpoint and optimizer hashes; weight variant (`raw` or `ema`).

- `coverage`: training row count and accepted seconds; unique source/group counts; batch size; accumulation; cumulative example slots `step * batch_size * accumulation`; approximate passes `example_slots / train_rows`. Counts are exposure estimates, not proof every row appeared equally often.

- `validation`: fixed subset identity, row/group count, step-zero loss, every checkpoint loss, finite-value status and candidate-minus-reference change. The existing capped first-100-row diagnostic is not the entire validation corpus; record its actual composition and keep it fixed across these comparisons.

- `generation`: fixed prompt-file hash and seed/settings, total prompts, healthy decoded audio count, nonfinite/empty/silent counts, frame-limit count, durations, per-prompt ASR text and character similarity, plus recognized repetitive-text flags. Keep raw ASR text; it is evidence, not a truth label.

Use the same 12 unseen prompts at every boundary. A separate fixed small set of training sentences can diagnose memorization; do not mix it into the unseen-prompt gate. A healthy output is finite, non-silent audio with at least eight generated frames. Additional maximum-duration or repetition flags remain visible.

**First bridge, 200 to 500:** admit only when all identity/restore checks pass, all observed training/validation losses and gradients are finite, approximate passes at step 200 are below one, latest fixed validation loss is no greater than `1.10 * baseline_loss`, healthy output count has not fallen, and frame-limit count has not increased relative to the same baseline prompts. This permits one limited undercoverage bridge even when semantics are not yet useful. The 10% guard is an explicit conservative operating rule, not a confidence interval or evidence of acceptable quality.

**Further boundaries, 500 to 1,000 and 1,000 to 2,000:** require finite/stable execution, no worsening generation-degeneracy counts, and both:

- Latest fixed validation loss is at least 1% below the previous accepted boundary's loss; retain the step-zero comparison separately. This directional comparison must not be described as beating initialization when it has not.

- On the same unseen prompts, median character similarity improves by at least 0.05, or at least three of twelve prompts gain at least 0.10 relative to that boundary. Reject a claimed gain driven only by repeated ASR text. These thresholds are admission heuristics for this bounded run, not validated Taiwanese-quality criteria.

If a gate fails, stop additional optimization, preserve the evidence and candidates, produce the private diagnostic package, and allow verified backup/stop. Do not silently substitute a different learning rate, tokenizer, architecture or more steps. If fewer than 12 comparable prompts are available or a metric is missing, record the gate as unresolved and stop automatic continuation rather than invent a score. At most three continuation chunks are admitted; each has its own time limit and checkpoint grace inside the single absolute deadline.

## 4. Measure affordability and evaluate the weights actually delivered

- Use observed seconds per update from the full-data run, including validation/checkpoint overhead, to bound each next chunk. Admit it only if it fits before **20:30 Taipei**, leaving the final 90 minutes before the hard VM stop for diagnostics and artifact handling. Stop earlier when the cumulative budget requires it. The US$100 limit permits considering continuation; it does not require spending the balance.

- Evaluate raw weights at every boundary. With EMA decay 0.999, the EMA after 200 updates still retains roughly 82% of the initialization contribution, so it can lag short adaptation. If time permits at 500 or later, evaluate one separately labeled EMA candidate on exactly the same prompts and validation inputs. Do not mix raw and EMA metrics or overwrite training weights to perform that comparison.

- The existing `model.safetensors` export normally uses EMA, whereas the current diagnostics load raw checkpoints. Delivering that default file after selecting raw weights would silently change the candidate. Export the selected variant explicitly, record its hash, and rerun generated-audio and actual Pocket-TTS loading/inference checks against the exported file with the exact tokenizer and model config.

- Verify the intended no-voice inference wrapper and text preprocessing. The current training/evaluation condition is no voice prompt; introducing a default public-API voice prompt at delivery is a different condition and must be tested rather than assumed compatible. Do not promise the 24-layer teacher has the small model's CPU latency; record real inference time on the intended runtime or state that it was not measured.

## 5. Private publication and honest usability gate

- The coordinator has created and verified `weiren119/pocket-tts-taigi` as private. Upload only the selected model package, tokenizer, matching relative-path configuration, inference instructions, hashes, source/parent provenance and evaluation report needed to reproduce it. Keep credentials and unrelated cloud state out; retain raw training media separately unless specifically included in an authorized artifact plan.

- Verify repository privacy and the exact uploaded revision, then download or otherwise hash-check the uploaded artifacts and load that revision. A successful upload is not successful inference. Prefer upload from the local verified backup so a token need not be placed in the GPU job environment. Failure to upload must not extend the spending deadline; preserve the local copy and report it.

- Mark candidates as experimental when Chinese prompt following or Taiwanese language/pronunciation is unverified. Usability requires the intended Chinese meaning to be conveyed in Taiwanese on unseen prompts, including negation and numbers, plus stable termination and reproducible inference. Same-labeler Breeze agreement, a lower teacher-forced loss or normal waveform statistics cannot establish those properties alone. A small independent Taiwanese listening check is the strongest final gate if available; absence of it must remain explicit.

## Main feasibility limit

The greatest demonstrated bottleneck is now **learning reliable semantic conditioning from weak, mixed-source data**, not EOS plumbing or the ability to run the GPU. The corrected six-minute pilot stops normally but does not follow its six Chinese inputs. The long-source trial yielded 38.645 accepted minutes from one 52.501-minute episode; it does not establish that all 26.1 source hours are usable Taiwanese pairs. The full corpus may contain Mandarin, music, scene changes, weak semantic ASR labels or repeated material despite exact-byte grouping. More updates cannot by themselves repair those labels.

Two hundred updates expose 1,600 example slots; whether that is several passes or a small fraction of one depends on the final accepted row count. Even 2,000 updates expose only 16,000 slots. Record that denominator before interpreting failure as insufficient steps or success as generalization. The proposed staged plan can discover a useful signal by tomorrow evening while bounding failed work, but cannot guarantee a usable cross-language model on this deadline. If no useful signal appears, the correct delivery is the verified private experimental artifact and a clear failed-quality result, not a claim that spending more would necessarily solve it.

Self-check: This is a concrete implementation/operations plan, not an executed controller. It uses the newly authorized US$100 total and fixed cutoff without resetting prior spending, preserves current corpus work, handles the old watcher's shutdown race, and respects launcher/tokenizer/optimizer identity. No source, runtime, cloud resource or Git state was modified; only this plan was written. No GPU work, delegation or Agentflow startup occurred.

## Implemented bounded controller

* _2026-10-25 20:10:03 +0800 (gpt-6-astra/inherited)_

- Implemented only `training/scripts/continue_taigi_training.py` and `tests/test_taigi_continuation.py`. The running corpus modules are unchanged. The controller requires a persistent systemd invocation and a new output directory; it waits for the existing parent unit to end with an empty cgroup and successful job result. Transient parent polling failures keep it alive. The operator's synchronous `ExecStopPost` parent-stop action remains necessary for unhandled umbrella failure while the parent is in a separate cgroup.

- Default absolute deadline is `1793023200`, with a **5,400-second reserve**: training admission ends by **20:30 Taipei**, leaving 90 minutes before the coordinator-selected 22:00 interpretation of “tomorrow evening.” The minimum accepted reserve is 1,800 seconds; individual training chunks cannot exceed 3,600 seconds. A normal bounded launcher return of 124 preserves prior evaluated candidates for selection. An outer command timeout fails closed, because nested-session termination has not been established; systemd then cleans the cgroup and existing artifacts remain available for backup.

- The controller reuses the parent's actual completed `--asr` baseline/final reports; it verifies their raw/baseline weight variant, checkpoint/config/tokenizer hashes, ordered fixed prompts, ASR identity and emitted audio hashes. Finite trained candidates only are eligible for export. Gate halt is operational completion with `quality_verified=false`; training/integrity failure remains a failed operation. Raw candidate export is explicit (`ema=None`); no automatic EMA trial is added.

- Child directories contain copied verified model/optimizer/EMA state, a lineage receipt and a fresh exact launcher identity. Only `run_dir` and `max_steps` change. Final `selected.json` has `state=complete`, `quality_verified=false`, `candidate_id=step-N-raw`, and exactly five `files` entries: `weights`, `config`, `tokenizer`, `evaluation`, `provenance`, each with absolute `path` and `sha256`. Publication remains local after verified backup.

- Before starting, root writes `PRODUCTION/continuation-deployment.json` with required `source_commit` (the full 40-character extension commit) and `controller_sha256` (the exact deployed script's SHA256). Startup verifies that receipt. Final provenance separately records actual runtime Git HEAD/dirty state, controller hash, Pocket-TTS inference API source hash, and the deployment receipt contents/hash; it does not pretend the frozen runtime checkout has the extension's commit.

```bash
/path/to/taigi-pipeline/bin/python -m training.scripts.continue_taigi_training \
  /absolute/PRODUCTION/job-200-tw \
  /absolute/PRODUCTION/continuation \
  --parent-unit taigi-full-pipeline.service \
  --absolute-deadline-epoch 1793023200 \
  --reserve-seconds 5400 \
  --chunk-seconds 3600 \
  --diagnostic-seconds 600
```

- Initial tests failed because the new controller module did not yet exist. Final focused evidence: **35 tests passed in 2.71 seconds** across continuation, corpus-job and cloud-watcher suites; Ruff check and formatting pass for both new files. Tests include actual CPU checkpoint/optimizer/EMA copying with fresh identities, nonfinite refusal, immutable parent state, raw safetensors export distinct from an EMA shadow, every selected-file hash, generation identity checks, poll failure/live-parent waiting and successful operation completion after a quality gate halts further training. No GPU training or cloud service was run by this worker.

Self-check: Implementation follows the approved bounded controller scope and exact publisher contract. This worker changed no existing training/corpus module, runtime, cloud resource, Git state or notebook. The deployment and end-to-end overnight operation remain root-owned and require the independent review and actual service lifecycle checks.
