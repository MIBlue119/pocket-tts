# Tracker

## Identity

- **Work key:** A-001-batch-weak-labels.

- **Active Ask:** A-002.

- **Goal:** Expand resumable weak labels and prove fixed-release training.

- **Last update:** 2026-10-25 19:17:22 +0800.

- **Evidence commit:** 43735eb.

## Overall state

- **State:** active.

- **Reason:** Work remains.

- **Total:** 10.

- **Completed:** 6.

- **Remaining:** 4.

## Accepted task checklist

- [x] **T-1:** Build resumable bounded batch processing in training/taigi/batch.py and tests/test_taigi_batch.py; obey design INV-1/2, preserve failures, verify resume and cancellation with focused tests. Source: A-002, recovered A-001 and prior weak-label automation request. Proof: pytest tests/test_taigi_batch.py,9 passed4.50s; commit4532e87.

- [x] **T-2:** Build immutable grouped release merging in training/taigi/merge.py and tests/test_taigi_merge.py; obey INV-3/4/5, verify duplicate handling, split isolation and tamper rejection. Source: A-002, recovered A-001 and prior ML pipeline/data expansion request. Proof: pytest tests/test_taigi_merge.py,11 passed0.68s; commit4532e87.

- [x] **T-3:** Run actual batch and bounded training/validation, add training/taigi/evaluate.py, tests/test_taigi_evaluate.py and deterministic validation in training/train.py / training/tests/test_validation_rng.py; save ignored artifacts and results in docs/plans/2026-10-25-batch-weak-label-results.md; failed jobs stay recorded, no false quality claim. Source: A-002, recovered A-001 and prior training authorization. Proof: actual54/56batch, filtered193-row release, fresh100-step corrected-EOS run and six generated/ASR diagnostics; docs/plans/2026-10-25-batch-weak-label-results.md. Termination improved; text-control quality remains insufficient. Additional necessary paths: training/modules/model.py, training/tests/test_terminal_eos.py, training/taigi/pipeline.py and focused tests.

- [ ] **T-4:** Review final commit, document CLI in docs/taigi-batch-weak-labels.md, deliver Git branch and notebook; resolve relevant review findings before delivery. Source: A-002, recovered A-001 and prior personal-repo request.

- [ ] **T-5:** Process every video from all six playlists with full-video acquisition, resident enhancement/ASR and verified resume; no silent sampling limit. Source: A-002, owner correction about all videos. Paths: training/taigi/corpus.py, corpus_worker.py, training/configs/taigi_youtube_full_inventory.json and their tests.

- [ ] **T-6:** Execute a durable corpus-to-training job with grouped validation, train-only tokenizer, offline W&B and fixed generated diagnostics. Source: A-002 continuation of original training authorization. Paths: training/scripts/run_taigi_corpus_job.py and tests/test_taigi_corpus_job.py.

- [ ] **T-7:** Use dedicated Nebius compute within the existing USD50 total, preserve OCR, establish fixed stop deadline, verify model compatibility and backup outputs. Source: A-002 continuation and prior explicit cloud/budget authorization. Paths: corpus-cloud-operations.md, corpus-compute-plan.md, training/scripts/watch_taigi_cloud_job.py, tests/test_taigi_cloud_watch.py and ignored data/taigi-full-corpus/cloud/.

- [x] **T-8:** Independently review full-corpus behavior and verify actual short/long video execution before production launch. Source: A-002 Astra preference and all-video scope. Paths: full-corpus-review.md and existing notebook/review records. Proof: source reviews, actual2/2whole-video trial with52.501-minute complete frame coverage, cloud5-update teacher tensor verification, persistent-unit and real SSH/rsync/SHA256 acceptance.

- [x] **T-9:** Send processed audio previews to phone. Source: A-002 latest exact user request. Proof: Telegram send_document confirmed three MP4s delivered; ffprobe verified audio streams and20/20/10-second durations. Paths: /home/weiren/tts-deliverables/2026-10-25-audio-preview/.

- [x] **T-10:** Measure rented GPU utilization and matched throughput, adopt acceleration only when measured useful within the existing budget and source/label invariants. Source: A-002, owner asks to maximize allocated compute. Paths: full-corpus-throughput-plan.md, ignored cloud benchmark receipts, and any justified focused implementation/tests. Proof:39variable clips exacttext parity; fair cached batch4 throughput1.339x; no-work resume; duplicate GPU workers did not help; production opts into batch4 at43735eb.

## Accepted scope changes

- Owner explicitly requested Astra for good training; add deterministic baseline/validation and bounded generated-audio diagnostics to T-3. Source: A-002. Effect: T-3 includes deterministic validation and generated-audio checks.

- Owner requested every video; remove sample-only scope and add T5–T8. Source: A-002 all-video correction. Effect: full290-video queue and durable cloud processing/training/backup are required; original56-window pilot remains evidence only.

## Current recovery

- **Current item:** T-5 through T-8.

- **Last proven result:** Corpus22, orchestration7, watcher14, repeat-filter/orchestration35 focused tests passed; actual short video17pairs; exact BPE parity; actual SSH/rsync checksum fixture passed.

- **Active blocker or running process:** Full290-video production resumed at19:50Taipei under persistent taigi-full-pipeline.service;5complete,1running,284pending at19:52. Taiwan-only retrieval repaired through loopback SSH SOCKS; backup watcher and absolute stop timer active.

- **Next safe action:** Observe full corpus and200-update candidate, verify backup receipts, sync cloud W&B locally, and clean up dedicated cloud resources after verified preservation; original deadline remains2026-10-26 15:06Taipei.

- **Expected changed files:** training/taigi/batch.py, training/taigi/merge.py, training/taigi/evaluate.py, training/train.py, training/tests/test_validation_rng.py, tests/test_taigi_evaluate.py, tests/test_taigi_batch.py, tests/test_taigi_merge.py, docs/taigi-batch-weak-labels.md, docs/plans/2026-10-25-batch-weak-label-results.md, .agentflow/, ag.json, .gitignore.

## Completion proof

- **All accepted tasks checked:** no.

- **Blocking accepted decision:** none.

- **Operation running:** yes.

- **Next action remaining:** T-3 through T-8 and T-10.

- **Evidence status:** current.

- **Judgment:** active.

## Update meaning

- Saving this tracker is a recovery checkpoint, not a stop signal.

- For completed work, Evidence commit names the Git evidence commit, or is not applicable in a plain folder. Local file and test proof is still required.

- Work continues with the next unfinished item unless an independent stop condition applies.
