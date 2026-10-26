# Tracker

## Identity

- **Work key:** A-001-batch-weak-labels.

- **Active Ask:** A-002.

- **Goal:** Expand resumable weak labels and prove fixed-release training.

- **Last update:** 2026-10-26 07:19:00 +0800.

- **Evidence commit:** 43735eb.

## Overall state

- **State:** active.

- **Reason:** Work remains.

- **Total:** 13.

- **Completed:** 10.

- **Remaining:** 3.

## Accepted task checklist

- [x] **T-1:** Build resumable bounded batch processing in training/taigi/batch.py and tests/test_taigi_batch.py; obey design INV-1/2, preserve failures, verify resume and cancellation with focused tests. Source: A-002, recovered A-001 and prior weak-label automation request. Proof: pytest tests/test_taigi_batch.py,9 passed4.50s; commit4532e87.

- [x] **T-2:** Build immutable grouped release merging in training/taigi/merge.py and tests/test_taigi_merge.py; obey INV-3/4/5, verify duplicate handling, split isolation and tamper rejection. Source: A-002, recovered A-001 and prior ML pipeline/data expansion request. Proof: pytest tests/test_taigi_merge.py,11 passed0.68s; commit4532e87.

- [x] **T-3:** Run actual batch and bounded training/validation, add training/taigi/evaluate.py, tests/test_taigi_evaluate.py and deterministic validation in training/train.py / training/tests/test_validation_rng.py; save ignored artifacts and results in docs/plans/2026-10-25-batch-weak-label-results.md; failed jobs stay recorded, no false quality claim. Source: A-002, recovered A-001 and prior training authorization. Proof: actual54/56batch, filtered193-row release, fresh100-step corrected-EOS run and six generated/ASR diagnostics; docs/plans/2026-10-25-batch-weak-label-results.md. Termination improved; text-control quality remains insufficient. Additional necessary paths: training/modules/model.py, training/tests/test_terminal_eos.py, training/taigi/pipeline.py and focused tests.

- [x] **T-4:** Review final commit, document CLI in docs/taigi-batch-weak-labels.md, deliver Git branch and notebook; resolve relevant review findings before delivery. Source: A-002, recovered A-001 and prior personal-repo request. Proof: code and evidence push8887aac73b60990df94d3e64d49fd769b9b77d23 verified identical to origin/taigi-batch-weaklabels; source reviews and operating report included.

- [x] **T-5:** Process every video from all six playlists with full-video acquisition, resident enhancement/ASR and verified resume; no silent sampling limit. Source: A-002, owner correction about all videos. Paths: training/taigi/corpus.py, corpus_worker.py, training/configs/taigi_youtube_full_inventory.json and their tests. Proof: all290terminal outcomes verified on2026-10-26;288complete,2explicit regional-access failures;22754weak pairs/17.9897hours. Failed sources remain recorded.

- [x] **T-6:** Execute a durable corpus-to-training job with grouped validation, train-only tokenizer, offline W&B and fixed generated diagnostics. Source: A-002 continuation of original training authorization. Paths: training/scripts/run_taigi_corpus_job.py and tests/test_taigi_corpus_job.py. Proof: r2 completed200updates with22606train/148valid rows, offline logs and12fixed generated/ASR diagnostics; validation.5278603→.5000720. Generation remains unsuitable,7/12caps and unrelated transcripts.

- [ ] **T-7:** Use dedicated Nebius compute within the owner-updated USD150 total, preserve OCR, establish fixed stop deadline, verify model compatibility and backup outputs. Source: A-002 continuation and prior explicit cloud/budget authorization. Paths: corpus-cloud-operations.md, corpus-compute-plan.md, training/scripts/watch_taigi_cloud_job.py, tests/test_taigi_cloud_watch.py and ignored data/taigi-full-corpus/cloud/.

- [x] **T-8:** Independently review full-corpus behavior and verify actual short/long video execution before production launch. Source: A-002 Astra preference and all-video scope. Paths: full-corpus-review.md and existing notebook/review records. Proof: source reviews, actual2/2whole-video trial with52.501-minute complete frame coverage, cloud5-update teacher tensor verification, persistent-unit and real SSH/rsync/SHA256 acceptance.

- [x] **T-9:** Send processed audio previews to phone. Source: A-002 latest exact user request. Proof: Telegram send_document confirmed three MP4s delivered; ffprobe verified audio streams and20/20/10-second durations. Paths: /home/weiren/tts-deliverables/2026-10-25-audio-preview/.

- [x] **T-10:** Measure rented GPU utilization and matched throughput, adopt acceleration only when measured useful within the existing budget and source/label invariants. Source: A-002, owner asks to maximize allocated compute. Paths: full-corpus-throughput-plan.md, ignored cloud benchmark receipts, and training/taigi/{corpus.py,corpus_worker.py,transcriber.py} with focused tests. Proof:39variable clips exacttext parity; fair cached batch4 throughput1.339x; no-work resume; duplicate GPU workers did not help; production opts into batch4 at43735eb.

- [ ] **T-11:** Gate bounded continuation on full-data evidence and deliver a reproducible candidate to verified-private weiren119/pocket-tts-taigi. Source: A-002 owner Oct25 evening deadline, USD100 and private Hugging Face instructions. Paths: continuation-plan.md, training/scripts/{continue_taigi_training,extend_taigi_training,publish_taigi_model,publish_taigi_when_ready}.py and their focused tests. Private step-200 publication is verified at bec7e32ccc5dbd6d3330954a531adb8999d4a698; usable-model quality remains unproven. Longer continuation is active.

- [ ] **T-12:** Incorporate ADI Hokkien through an audited full eligible source release, corrected Breeze weak labels and a fresh training identity. Source: A-002 owner says ADI台語資料也應該用上. Paths: training/scripts/prepare_adi_taigi.py, tests/test_taigi_adi_prepare.py, training/scripts/build_adi_semantic_release.py, tests/test_adi_semantic_release.py, training/scripts/fork_taigi_adaptation.py, tests/test_taigi_adaptation.py, ADI-integration-plan.md, ignored data/taigi-adi-20261026 and explicitly documented adaptation configuration. Preserve official test, speaker/sentence/audio isolation, original transcripts and TRAIL notices; exclude Mandarin audio. Proof: actual archive hashes, manifest exclusion counts, focused tests, weak-label receipts and training run manifest naming the ADI release. Failures stay recorded; no silent sample cap or mutation of running YouTube inputs.

- [x] **T-13:** Use pstack Investigation/how with independent recipe and language explorers and a fresh explainer to recommend non-English fine-tuning. Source: A-002 latest explicit pstack/subagent request for 台語、繁體中文、日文. Paths: docs/research/2026-10-26-nonenglish-*.md. Proof: primary-source citations, pinned local architecture evidence, worker reports and consolidated recommendation with uncertainties. Investigation changes no live training configuration. Completed: two independent explorer reports and fresh Astra synthesis saved, actual local config/embedding/evaluator and primary sources checked; recommendation report distinguishes evidence from unexecuted experiments.

## Accepted scope changes

- Owner reauthenticated Nebius, authorized an H100 if useful, and raised the total budget to USD150. Source: A-002 latest owner messages on Oct26. Effect: H100 instance computeinstance-e00qm3e1gfftr7ey2y admitted with conservative total estimate USD91.557 and independent setup/final shutdown controls; old task VM verified STOPPED. Absolute22:00 deadline unchanged. T12 also includes source-aware semantic admission because actual short-clip ASR produced meaning errors.


- Owner explicitly asks「還是要訓練久一點」onOct26. Source: A-002 latest owner input. Effect: T11 now includes a separate bounded exploratory continuation200→1000→3000→6000 using local RTX3090 while cloud API authentication is unavailable. Early pilot quality admission is superseded for this exploration; finite/divergence/time limits and honest quality reporting remain. No change toUSD100total or22:00cutoff.

- Owner raises total cap to USD100, requests usable model by Oct26 evening and private Hugging Face publication. Source: A-002 exact latest owner inputs. Effect: T7 deadline controls update preserving original start; T11 adds conditional continuation and private model delivery.

- Owner explicitly requested Astra for good training; add deterministic baseline/validation and bounded generated-audio diagnostics to T-3. Source: A-002. Effect: T-3 includes deterministic validation and generated-audio checks.

- Owner requested every video; remove sample-only scope and add T5–T8. Source: A-002 all-video correction. Effect: full290-video queue and durable cloud processing/training/backup are required; original56-window pilot remains evidence only.

## Current recovery

- **Current item:** T-7, T-11, T-12 and T-13.

- **Last proven result:** Original1200 semantic control failed its generation gate. Full prepared ADI audio22745 files is verified;15924 original ASR caches plus212 cloud prefix retained. Independent content-inventory proof reproduces all22754 accepted YouTube rows from288 parents;2 unavailable sources remain explicitly failed. Shard/cache identity and repetition-filter proofs pass.

- **Active blocker or running process:** Three disjoint ASR workers: local1705 rows, RTX2452 rows, H1002452 rows; all pinned batch4. H100 also receives frozen YouTube release directly fromRTX. Full accepted+quarantined ADI accounting is unfinished. No new training started. Independent22:00 provider watchdogs and cumulativeUSD150 cap remain binding.

- **Next safe action:** Finish all shards, verify exact cache union, finalize ADI, build filtered ADI+YouTube release, validate source hashes/splits/token coverage/real loader and exact1000 restart pair. Then run clean-history LR5e-5 data-only1200 first stage, evaluate fixed unseen demo/heldout panels, and extend only with actual generation evidence. User's one-hour start target is subordinate to complete inputs; goal is honest social-media demo quality. Private experimental Hugging Face backup remains authorized.

- **Expected changed files:** ASR and trainer implementations remain unchanged this turn. Delivery fix touches training/scripts/publish_taigi_model.py and tests/test_taigi_publish_model.py to preserve selected decoding and punctuation settings. Operational helpers/configs/receipts remain under ignored data/taigi-adi-20261026/ and data/taigi-full-corpus/cloud/. Other tracked work is this tracker, stream devlog and experiment report. History-noise implementation stays committed but disabled.

## Completion proof

- **All accepted tasks checked:** no.

- **Blocking accepted decision:** none.

- **Operation running:** yes.

- **Next action remaining:** T-7, T-11, T-12 and T-13; generation-quality recovery remains necessary for the owner’s usable-model objective.

- **Evidence status:** current.

- **Judgment:** active.

## Update meaning

- Saving this tracker is a recovery checkpoint, not a stop signal.

- For completed work, Evidence commit names the Git evidence commit, or is not applicable in a plain folder. Local file and test proof is still required.

- Work continues with the next unfinished item unless an independent stop condition applies.
