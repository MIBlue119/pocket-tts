* _2026-10-25 19:19:19 +0800 (gpt-6-astra/inherited)_

Reviewed commit: 6c3e6ecdbee817d19686be6eaae365efe852dd26

# Full-corpus source review before production

**Summary**

- No actionable source-code blocker was found in the serial corpus worker, resumable queue or training orchestrator at this commit.

- Full production launch remains conditional on completing the actual whole-video acceptance: correct the service's Hugging Face cache environment, finish successful transcription/publication, and verify complete coverage of the long episode. The short 71-second source has completed enhancement; that alone is not end-to-end acceptance.

- This report covers the serial design. The owner's newer request to accelerate processing and any resulting concurrency changes require a focused review of their own before relying on this acceptance.

Outcome: PASS

Minimality: PASS

Conformance: PASS

These verdicts concern the reviewed code. They do not certify a completed corpus, trained model, cloud production launch or useful Taiwanese speech.

## Remaining launch blockers

- **Integration environment:** The coordinator's actual trial found that Breeze uses the default Hugging Face cache while RE-USE receives an explicit cache directory. The service lacked `HF_HUB_CACHE`; with offline loading, the correct files were therefore not found for Breeze. Set the actual service environment to the verified shared cache and demonstrate successful transcription/publication. This is an observed launch configuration problem, not a request to refactor the model loaders.

- **Whole long-video acceptance:** The design requires one complete long episode before scaling. At this report's cutoff, long-video acquisition was underway and only the short video's enhancement had completed. Inspect its final source frame count, contiguous chunk intervals, covered frame total, hashes, accepted/quarantined counts and final release. A successful 20-second compatibility test or the short enhancement stage does not replace this check.

## What was reviewed

- Read `full-corpus-design.md`, all three new implementation modules, their corresponding tests, the frozen inventory and the existing acquisition, enhancement, candidate preparation, publication, merge, tokenizer conversion and bounded training helpers they call. Inspected the final seven-file commit after initially reading the pending files. Subsequent uncommitted documentation and cloud-controller work are outside this source verdict.

- Independently normalized the actual inventory using the reviewed code with GPU visibility disabled and bytecode writes disabled: exactly **290 video IDs**, no missing IDs, and six groups containing **91, 88, 36, 27, 26 and 22** videos. All memberships remain represented; duplicate playlist memberships unite their groups before scheduling.

- Reused coordinator test evidence: **22 corpus/worker tests** and **7 orchestrator tests** passed. These cover bounded outcomes, queue identity, corruption refusal, completed-stage reuse, exact chunk coverage, resident loading, interrupted text caching, train-only tokenizer input and partial-corpus refusal. I did not repeat those suites or run GPU inference.

## Correctness and boundary assessment

- **Resume and identity:** The frozen queue includes inventory, configuration and implementation hashes. Completed stages are verified before new work; stages published immediately before interruption can be recovered. Acquisition promotion can recover already-verified raw audio. Every source retains a queue outcome, and retry caps do not silently drop unavailable videos. Code changes refuse reuse rather than mix implementations within one corpus.

- **Source and label integrity:** Enhancement tiles the actual decoded frames in bounded chunks, preserves the original video hash and sample offsets, and publishes completed chunks by directory rename. One enhancement model is loaded lazily per video; completed chunks are reused. Breeze loads lazily once per transcription stage, and cached text keys include audio, interval, model, adapter and implementation identity. Candidate rejection remains explicit. Nonempty ASR output remains an unverified weak label.

- **Frozen training data:** The orchestrator requires a terminal outcome for every frozen queue item and rejects active, pending or interrupted work before merging. It verifies successful worker releases and reuses the established merger's provenance checks, duplicate handling and source-group separation. Only the training manifest trains the tokenizer; validation text is used for conversion parity checking. The vocabulary size is checked against the actual converted tokenizer rather than assumed.

- **Training and diagnostics:** The fresh recipe preserves the pretrained 24-layer model, reset text embeddings, sentence mode, frozen Mimi, LSD, no target voice prompt and no precompute/compilation. It bounds training time and updates, logs a seeded validation baseline, evaluates fixed prompts before and after raw-checkpoint training, and requires a checkpoint/optimizer pair. The validation cap is recorded; it is a bounded development diagnostic, not complete corpus evaluation. Fixed generated prompts and teacher-forced losses do not establish Taiwanese pronunciation or translation quality.

- **Cancellation under systemd:** Corpus stages reuse the reviewed durable token cleanup. The outer job explicitly relies on systemd to contain nested sessions; its process-group signals alone are not a containment guarantee. The coordinator reports verifying the actual trial unit with `KillMode=control-group`, `TimeoutStopSec=60s` and `RuntimeMaxSec=1800s`. Preserve that cgroup ownership and derive the production runtime from the original budget deadline, retaining backup time. The presence of `INVOCATION_ID` alone does not verify those unit properties.

- **Budget and disk:** Source duration/file-size limits, stage deadlines, attempt caps and disk admission failures remain explicit. Disk admission is a reserve check, not a hard quota. The job reserves time for training, diagnostics and backup, while the independent VM deadline must bound billable runtime including setup. Read the recorded absolute-stop timer evidence: `2026-10-26 07:06:12 UTC`, persistent across reboot. Its existence does not certify final charges or storage cleanup; those remain operational responsibilities. This review did not invoke cloud APIs or change resources.

## Scope and minimality

- The owner's correction explicitly requires processing all supplied playlist videos rather than the earlier short-video sample. The queue, full-source acquisition, exact chunk coverage, resident stages and partial caches directly implement that change. The training wrapper connects a completed frozen corpus to the already requested adaptation and diagnostics; it does not introduce a new model or service API.

- I considered raising the old single-video limits and looping that command. It would repeatedly acquire the same source and reload models for short intervals, and would not preserve verified resident-stage progress. The two corpus modules instead reuse the existing enhancer, segmenter, transcriber, release publisher and lifecycle helper. Reusing those helpers also keeps the training-data contract unchanged. No plausible deletion or combination was found that preserves the accepted full-source and restart guarantees with less machinery.

- No source repair is requested by this report. Close the two observed integration/acceptance items above, and review any newly implemented acceleration separately. Actual corpus completion, training outcomes, checkpoint restoration, backups, final spending and Git delivery remain to be demonstrated by the coordinator.

Self-check: The reviewed commit is exact; code acceptance and pending launch acceptance are separated. Only observed launch blockers are listed. Inherited tests, independent inventory inspection and coordinator-reported runtime evidence are identified distinctly. No product files, live processes, GPU workloads, cloud resources or previous review reports were modified; only this report was written. No Agentflow execution or delegation occurred.

## Integrated follow-up — tokenizer, repetition quarantine and service lifetime

* _2026-10-25 19:26:39 +0800 (gpt-6-astra/inherited)_

Reviewed follow-up commit: 0647ed10663eea267bf4d4fa08d6a7208df0070e

- **Code assessment:** No actionable blocker in the four-file diff from `d4c109a8a4b9c0e55b4920d13d69ff2bc3e39d8f` to this commit. Reused the coordinator's 35 passing focused tests. The earlier full-corpus production acceptance conditions still apply; the old-code long-video trial remains separate from this new frozen implementation.

- **Tokenizer diagnosis and fix:** Independently inspected the actual failing unigram model and text. SentencePiece and the JSON tokenizer chose the same counts of six-character `吟` pieces and single `吟` pieces, but placed the five singles at opposite ends of a repeated run. This is an equal-score segmentation difference, not missing vocabulary. The orchestrator now selects the existing BPE trainer mode and records that choice; the converter and its exact parity requirement are unchanged. The coordinator's real BPE conversion passed all 195 original corpus texts plus nine built-in checks, including the pathological text. The original failing unigram artifacts remain preserved. Future full-corpus BPE conversion must still pass its own parity check; this pilot result is not a guarantee for every future vocabulary.

- **Narrow label gate:** The new helper flags at least 12 identical consecutive alphabetic/CJK characters after removing whitespace. Punctuation breaks the run and cannot itself trigger rejection. `transcribe_release` preserves the original supplied text, source reference and rejection reason, excludes the row from accepted manifests, and counts quarantine reasons. Tests retain the actual chanting and laughter labels, check the 11/12-character boundary and punctuation behavior, and verify preserved evidence. This is a conservative automatic exclusion rule, not an assessment of Taiwanese linguistic truth or a general hallucination detector.

- **Actual data evidence:** The independent earlier audit found one 2.04-second training row with 53 repeated `吟` and 41 repeated `吿`, and one 3.22-second validation row with 81 whitespace-separated `笑`. The coordinator reports a new immutable filtered release `c7977dd525b91eea` containing 193 rows: 165 training, 28 validation and two quarantined rows; original releases remain unchanged. The local 100-step diagnostic is running, not completed, and this report makes no speech-quality claim.

- **Service lifetime:** A real transient service was garbage-collected before a later watcher poll, so the production contract now uses a persistent enabled unit with `Restart=no`. The coordinator's actual tiny-unit fixture remained loaded and inactive with start/exit history and invocation identity after completion. That evidence supports retaining the current watcher logic; adding `RemainAfterExit=yes` or accepting `active/exited` is unnecessary. Preserve the persistent unit until final backup verification and stopping are complete. The real SSH/rsync/hash watcher fixture is still being exercised at this report's cutoff.

Self-check: This appendix reviews only the named four-file change and the supplied service-lifetime evidence. It preserves the original report and unresolved operational acceptance boundaries, distinguishes actual pilot diagnostics from general claims, and introduces no new model, converter rewrite or source edits. No GPU, cloud operation, Git mutation, Agentflow invocation or delegation occurred.
