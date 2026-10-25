* _2026-10-25 19:39:57 +0800 (gpt-6-astra/inherited)_

Reviewed commit: 93e69672c6b8d00db68555fc64a3d85c50f43794

# Focused resident ASR batching review

- No actionable blocker was found in the six-file change from `78bc38d` to this commit. Batch size one remains the default; the new option admits one to four segments in one resident Breeze process.

- The coordinator's actual GPU acceptance covers 11 unequal segments from 0.8 to 17.34 seconds, grouped 4/4/3: all 11 texts matched single inference, the initial cache pass loaded the model once, and resume performed zero model loads and zero inference calls.

- No end-to-end speedup is established by that check. Raw single inference took 5.337 seconds, while batched work including hashes/cache writes took 5.446 seconds; those timings do not compare equivalent work. Duration-sorted prefill and a fair comparison are proposed follow-ups, not part of this reviewed commit.

Outcome: PASS

Minimality: PASS

Conformance: PASS

## Reviewed behavior

- Single and batched adapters share the same interval reading, mono conversion and explicit resampling to the model's 16 kHz feature-extractor rate. Each call builds separate input dictionaries because the Hugging Face pipeline mutates them. The pinned model, Chinese transcription settings and corrected adapter remain unchanged.

- Batch output count is checked before storing results. Ordered results are paired with their original interval identities, atomically cached one at a time, and subsequently read through the unchanged single-row publisher. A failed later batch preserves earlier completed caches; a failure partway through cache writing preserves completed rows and retries missing rows. Exact duplicate cache keys are skipped, including duplicates within the pending list.

- Candidate audio hashes are checked before batch inference. Cache identity contains audio hash, interval, model/revision, adapter, stage-code identity and configured batch size. Batch size is also frozen into the corpus plan, worker specification and stage reports; changing it cannot silently resume the same corpus. Existing cached text integrity checks and release validation remain active.

- Only one model stage/process is introduced, preserving the serial enhancement/transcription lifecycle. The current stage deadline, token cleanup, retry bounds and disk admission remain in force. No additional cloud resource or model is required.

## Evidence and limits

- Reused the worker's reported 38 passing focused tests and inspected the added tests for mixed-rate/mixed-length resampling, fresh input dictionaries, order, duplicate skipping, short final batches, partial failure, cache corruption, resume and frozen batch configuration. The coordinator was also rerunning the focused set. No redundant suite or GPU work was launched by this reviewer.

- The coordinator reports the separate old-code whole-video trial completed both sources, including a 52.501-minute episode with exact decoded-frame coverage and 760 weak labels covering 38.645 accepted minutes. Its actual resume completed with no repeated inference. That establishes the earlier serial integration; the new actual batching acceptance above checks the added path.

- Exact output equality on these 11 samples does not guarantee identical numerical decoding for every future batch composition. Keep labels unverified and preserve the configuration/code identity. No Taiwanese speech-quality or universal batching-speed claim follows from this acceptance.

## Minimality and proposed follow-up

- Reusing the pipeline's list input with per-segment caches is smaller than multiple ASR processes, a queue service or a different model. Prefilling existing caches leaves publication, quarantine and original manifest order intact. All added mechanisms directly support the owner's requested acceleration while preserving resumed work.

- A stable duration sort of only the inference work list is reasonable to reduce padding and long-segment delays. Keep original publication order and cache keys unchanged, then repeat exact per-row text and zero-work-resume checks on that new commit. Compare `CachedTranscriber` batch sizes one and four using separate empty cache directories and equivalent hashing/cache work, with model-load and steady-state timings distinguished. Do not compare the earlier raw-pipeline timing against a cache-writing batch run.

Self-check: The source verdict is anchored to the exact commit and separates existing unit evidence, coordinator-observed GPU acceptance and proposed future sorting. No source file, GPU process, cloud resource or Git state was changed; only this report was written. No Agentflow invocation or delegation occurred.

## Focused follow-up — duration-sorted prefill

* _2026-10-25 19:41:30 +0800 (gpt-6-astra/inherited)_

Reviewed follow-up commit: 43735eba6ab9fd190d99607c909f065fc51afb9f

- **Source assessment: PASS.** The only implementation change sorts a new inference work list by numeric duration. Python's stable sort preserves supplied order for equal durations and does not mutate the original candidate list. Cache identities remain attached to each original interval, and publication still reads candidates in their original order through those caches.

- The added focused test verifies duration grouping, equal-duration stability, unchanged input rows, original-order text lookup and no repeated inference on a second prefill. Reused the coordinator's 14 passing worker tests; no additional source risk required another suite.

- The existing implementation hash changes with this edit, so frozen plans/caches cannot silently reuse the earlier grouping implementation. The one-line sort is sufficient for the intended padding reduction; no process concurrency, scheduler or model change is needed.

- **Runtime acceptance remains pending for this sorted commit.** The coordinator has prepared a fair 39-row comparison of `CachedTranscriber` batch sizes one and four, with cold-cache hashing and cache writes included on both sides, nine full batches and a final batch of three, original-order text comparison and zero-work resume. The earlier 11-of-11 result remains evidence for the preceding unsorted implementation, not a substitute for this new measurement. Do not claim a speedup before the fair result is available.

Self-check: This appendix covers only the one-line sort and its focused test at the named commit. Source acceptance is separated from pending real-model parity/performance evidence. No source, GPU, cloud, Git or live-process action occurred; only this report was appended.

## Actual sorted-path acceptance

- The coordinator reports acceptance at `43735eba6ab9fd190d99607c909f065fc51afb9f`: **39 variable-length rows**, totaling **127.54 source seconds**, produced **39 of 39 exact text matches** between batch sizes one and four. The sorted groups were nine batches of four and one final batch of three. Resumed cache access performed zero model loads and zero inference calls.

- Equivalent `CachedTranscriber` work, including input hashing and cache writes, took **15.3596 seconds** at batch size one and **11.4703 seconds** at batch size four: **1.339× measured throughput**, or approximately **25.3% less wall time**, on this workload. The same 3.778-second warm/load cost was excluded from both. Peak GPU memory increased from **8.357 to 15.540 GiB**. This establishes the proposed optimization on the measured sample, not a guaranteed whole-corpus speedup.

Self-check: The previously pending sorted-path parity/cache acceptance is now supported by coordinator-reported actual measurements; no new source change or independent GPU run was made by this reviewer.
