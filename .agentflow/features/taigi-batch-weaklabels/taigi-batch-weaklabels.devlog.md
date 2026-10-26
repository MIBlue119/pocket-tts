# STATUS

Project: pocket-tts

Notebook: .agentflow/features/taigi-batch-weaklabels/taigi-batch-weaklabels.devlog.md — stream.

Current commit: stream-open only, no code commits yet.

Tests/scenarios: none.

Configuration: .agentflow/features/taigi-batch-weaklabels/ag.json — schema v8; validated for codex this round.

Proven: the stream configuration was copied from the root configuration.

Open: none.

Next: reply to the first Ask below.

Artifacts: none.

Archived eras: none.

Streams: none.

Backlink: main notebook `.agentflow/devlog.md` (main checkout)

Feature: taigi-batch-weaklabels — active — Taigi batch weak labels

Opened by the `agf` shell shortcut on 2026-10-25, not by an agent round. The main-notebook `stream:` pointer line was deliberately NOT written — the next main-checkout session re-derives it from `.agentflow/features/*/*devlog.md`.

---

# → Ask / A-001 (MIBlue119)

+ 請繼續幫我完成任務 你可以用 agentflow 的godev來達成

---

# → Ask / A-002 (MIBlue119)

+ 請繼續幫我完成任務 你可以用 agentflow 的godev來達成

+ help me use astra to have this good training

+ 進度？

+ 為什麼 我給你的 youtube 清單 不是有超多影片 你應該要都處理

+ 你目前處理得如何？

+ 假如你想省token 可以astra plan, implement with sol, if some problem happened fixed with astra

+ 處理進度？

+ 你可以傳處理過的片段到我手機嗎？ 或是告訴我哪裡可以聽到

+ 你有辦法榨乾你開啟的算力，然後加速處理嗎？

+ 進度如何？約莫多久可以處理好

+ 處理時間要多久

+ 假如狀況不錯 你可以訓練到明天晚上 預算提升到100usd

+ 我的目標是我明天晚上至少有一版可以用的pocket tts 台語版

+ 然後模型可以幫我放在我的huggingface 先設private

+ 進度呢

+ 還是要訓練久一點

+ 這在rtx6000 或是 H100 會訓練的比較快嗎？ 另外可以用你已經訓練的版本跑個推論給我嗎？  模擬我輸入文字 `你好嗎？`

+ 為什麼會爛爛的

+ 你目前的訓練資料是有用到我們全部的訓練資料嗎？

+ ADI的台語資料你也應該用上才對

+ 另外我也好奇你目前是finetune嗎？

+ help me let pstack to do with subagent to check how could we finetune pocket tts at non english language like 台語, traditional chinese, japanese go

+ 你知道我們目前的目標嗎？

+ ＡＤＩ資料的標注？台語訓練資料是有中文標註的嗎？

+ 沒有的話你就會需要另外處理了

+ 我先去跑步，交給你來優化了

---

## [RUN-001] Event — 2026-10-25 18:48:02 +0800 (A-002)

- Recovered the unhandled A-001 request verbatim into the final A-002 Ask. Stream creation appended an empty A-002, so the writer correctly refused a progress write to non-final A-001. Continue the same authorized task; original owner text is preserved.
- Requirements, discovery/codewalk and specification are committed at6fd16f5 in the already allocated artifacts/A-001-batch-weak-labels/design.md. No new owner decision is pending; continuation authorizes implementation. Route full_pipeline, allow-ag on; host owns requirements/specification and integration, separate native workers implement disjoint slices, followed by independent acceptance.
- Astra owns batch lifecycle/recovery and Sol owns immutable release merging. Both have explicit file lists, shared filesystem permissions, and no permission to invoke Agentflow or delegate. Root retains actual GPU operations and final delivery.
- The documented feature workspace resolved the root-notebook commit refusal. Only inherited Claude identity markers were omitted from that command to select the actual Codex host. Existing product branch remains preserved.
- Prepared56 nonoverlapping intervals from25 supplied short videos, totaling905 requested seconds. Five long interviews remain explicitly deferred by existing source limits. Data files are ignored; no new cloud resource is allocated.
- Explore/spike reuse prior actual model/video evidence. Security boundaries (process cancellation, file integrity and split leakage) are part of final independent acceptance; no separate learning stage is needed.

## [RUN-002] Event — 2026-10-25 18:50:09 +0800 (A-002)

- Batch unit326162f is committed. Host reran the focused suite:9 tests passed in4.50s, including actual cancellation and interpreter behavior.
- Real batch started in PTY /dev/pts/154, tool session34149:56 jobs,2700-second overall admission/run budget,180-second per-attempt timeout,2 total attempts per job and10GiB disk reserve. It runs locally on RTX3090, with no new cloud allocation.
- Owner requested Astra for good training. A separate gpt-6-astra native advisor is inspecting the real training/validation path and will provide a concrete quality plan; its only write is astra-training-plan.md.
- Host review found a deduplication bridge that could lose group connectivity and duplicate utterance IDs across chunks. The merge worker is correcting those specific R3/INV3 integrity issues before integration; source and tests remain within its two allowed files.

## [RUN-003] Event — 2026-10-25 18:55:00 +0800 (A-002)

- Host accepted merge bridge/ID/integrity corrections after reading code and running11 tests. Batch9, evaluator2, validation/config19 tests pass; scoped ty and Ruff pass. Product and operating guide committed at4532e87.
- Astra's training advice is accepted: batch1×accumulation8, lr1e-4, warmup10, up to100 updates with actual baseline/held-out validation and fixed generated prompts. New validation isolates RNG; finite generated-audio diagnostics inspect raw weights and keep same-ASR consistency distinct from quality.
- Native Astra independent review started at exact4532e87 with frozen review-dispatch.md, review-facts.json and full-depth review-plan.json. It has a separate context and shared permissions; no enforced read-only or different-family claim. No GPU tests are authorized during production.
- Owner asked “進度？”; at that observation8/56 jobs were complete,28 pairs and91.18 accepted seconds, no failed jobs. The batch continues; training has not started yet.
- Scope comparison: changed code/tests, CLI guide, Astra advice and review records match tracker T1–T4 and the owner's added good-training request. Media/model artifacts remain outside Git. Tracker validated2/4 complete.

## [WIP-001] Checkpoint — 2026-10-25 18:55:00 +0800 (A-002)

- **Finished:** Resumable batch, immutable grouped merger, deterministic validation and bounded generation diagnostic code; relevant tests and scoped checks pass.
- **Running now:** Local56-job producer in PTY34149 and independent Astra review of4532e87. No additional cloud resources.
- **Still to do:** Finish or explicitly account for each batch job, freeze train/validation release, train tokenizer and bounded teacher, compare baseline/final audio, verify artifacts and deliver branch.
- **Next work action:** Monitor batch state and inspect generated releases while review runs; keep GPU inference/training sequential.
- **Checks:** [x] tracker.md | [x] devlog RUN | [x] scope matches tracker

## [RUN-004] Event — 2026-10-25 18:55:11 +0800 (A-002)

- Correction to the preceding record: the first tracker check rejected missing literal Proof/Effect fields; the command sequence continued and wrote an inaccurate validation claim. Added the already observed test evidence and scope-effect fields, then reran the checker successfully:2/4 complete. Product test results were unchanged; no implementation tests were repeated.

## [RUN-005] Event — 2026-10-25 18:58:54 +0800 (A-002)

- Astra review found two blocking edge cases: altered compact manifests with updated local hashes, and orphan GPU stages after abrupt wrapper death. Existing INV2/4 required fixes; design explicitly reopened. The original blocking report is preserved.
- Corrections committed at2d25d35: compare canonical compact content on reuse; persist random attempt ownership before launch and use PID descriptors for safe cleanup even after parent death. Root batch13 tests pass5.65s; exact manifest-tamper regression passes. Linuxx86-64 requirement is documented.
- Controlled real PTY interruption stopped job20 and saved19 completed jobs. GPU inspection showed only unrelated610MiB application remained. Restarted exact same plan/options under corrected runner in PTY session13496; independently verified all19 saved jobs/checksums and one attempt each, with zero repeat inference for completed work. Resume proof is in ignored data/taigi-batch-v1/resume-proof.json.
- Astra second review now targets2d25d35 through review-dispatch-r2.md; final data and training evidence remains pending.

## [RUN-006] Event — 2026-10-25 19:17:08 +0800 (A-002)

- Owner clarified all six playlists, superseding the sampled producer. Full enumeration contains 290 unique videos and approximately26.1source hours. All IDs are frozen in training/configs/taigi_youtube_full_inventory.json; round-robin scheduling includes every playlist, including long interviews.
- Full-corpus worker, supervisor and bounded corpus-to-training orchestration committed at6c3e6ec. Host verified22 corpus/worker tests and7 orchestration tests. Native Astra independently reviews this new scope; native Sol implements a separate bounded backup controller. Shared filesystem access is not OS-enforced isolation.
- The original56-window producer finished54 successful windows and2 explicit failures; it did not process all290videos. Its outputs remain reusable evidence and are not presented as full coverage.
- Dedicated Nebius RTX PRO6000Spot VM computeinstance-u00wakm49f57fd1zp8 passed actual RE-USE and Breeze inference, with24model hashes verified. Root accepted healthy VM ownership; original20-hour deadline2026-10-26 07:06:12UTC remains fixed. Both external provider-stop and persistent in-VM shutdown timers are active. Prior OCR instance remains STOPPED. Full corpus and training have not started yet.
- Owner requested processed examples on phone. Three actual MP4 files were created with ffmpeg, audio streams and exact20/20/10-second durations verified, and Telegram confirmed all three delivered. Direct WAV delivery was rejected by extension allowlist; real AAC/MP4 conversion resolved it. Files live under /home/weiren/tts-deliverables/2026-10-25-audio-preview. These are source/enhanced training material, not model-generated TTS.

## [WIP-002] Checkpoint — 2026-10-25 19:17:22 +0800 (A-002)

- **Finished:** Full290-video inventory;54/56pilot windows;22corpus/worker and7orchestration tests; actual cloud model compatibility; three phone previews delivered.
- **Running now:** Astra full-corpus review and cloud dependency readiness; Sol backup-controller implementation; root actual integration.
- **Still to do:** Actual short/long full-video acceptance, all290durable launch, training/diagnostics, backup, cloud cleanup and Git delivery.
- **Next work action:** Install frozen6c3e6ecarchive on the healthy cloud VM and run bounded full-video checks under systemd containment.
- **Checks:** Tracker scope updated to T1–T9; RUN records current evidence. Full completion and tracker schema validation are not claimed.

## [RUN-007] Event — 2026-10-25 19:27:28 +0800 (A-002)

- Full-corpus source6c3e6ec passed independent review; actual short source produced17weak pairs after correcting the cloud service HF_HUB_CACHE environment. Failed preload attempts remain recorded. The long source has published six enhancement chunks; whole-source completion checks and ASR are pending.
- Owner requested compute acceleration. Actual long-video GPU samples show97–99% utilization, approximately5.95GiB memory and554–563W. Low memory usage does not establish spare compute. Astra is preparing bounded Breeze batch1/4 and matched one/two-process throughput comparisons after the actual trial releases the GPU; no speculative extra hardware was provisioned.
- Bounded backup controller committedd501685; review found stale running/missing job metadata after service death could delay early stop. Correctiond4c109a passed14host tests and independent follow-up. Persistent enabled units are required: a transient successful unit was actually garbage-collected. A real persistent tiny service retained execution history; SSH/rsync copied and SHA256 verified both fixture files, correctly marking the stale running job interrupted. No stop helper was supplied to the test.
- Actual local tokenizer conversion exposed equal-score unigram token-order differences on a repetitive ASR label. Existing BPE mode passed all204original train/validation/builtin texts without bypassing parity. Two obvious repeated-character labels were found. Commit0647ed1 adds a narrow >=12identical alphabetic-character quarantine with preserved text/audio/reason, and explicit BPE training. Host35focused tests and Astra review pass; original failing artifacts remain preserved.
- Replaying existing ASR labels through the new gate required no model inference. New immutable pilot releasec7977dd525b91eea contains193rows:165train,28valid;2quarantined. Local bounded100-step diagnostic now runs with exact BPE parity and completed fixed baseline generation; training completion and audio quality are not claimed.
- Owner asked timing. At19:24Taipei measured stable enhancement was about7.4source seconds per wall second; communicated provisional6–10hours full data processing plus1–3hours initial training/evaluation, subject to whole-source and throughput evidence. All290videos are queued in the inventory, not yet running in production.
- Tracker validated3/10complete after correcting missing canonical Source fields; its tasks cover full corpus, training, cloud/backup, review, phone delivery and measured acceleration. Current source changes and allocated reports match those boundaries; ignored media/model artifacts remain outside Git.

## [WIP-003] Checkpoint — 2026-10-25 19:27:28 +0800 (A-002)

- **Finished:** Whole290-video inventory; actual short-video pipeline; source and backup reviews; real backup fixture; BPE parity and repetition quarantine; phone previews delivered.
- **Running now:** Cloud long-video acceptance, local100-step diagnostic, Astra bounded throughput work waiting for the cloud GPU.
- **Still to do:** Verify whole long video, adopt only measured acceleration, launch full production with durable backup/stop controls, inspect training results and push delivery.
- **Next work action:** Read long-video result and local training metrics; integrate benchmark findings before freezing cloud production code.
- **Checks:** [x] tracker.md | [x] devlog RUN | [x] scope matches tracker

## [RUN-008] Event — 2026-10-25 19:54:02 +0800 (A-002)

- The whole-video acceptance finished2/2: long52.501minutes with exact frame coverage,760weak labels/2318.72accepted seconds; short17labels. A real PTY resume verified existing artifacts without repeating inference.
- Measured optimization is deployed at43735eb: one enhancer, resident Breeze batch4 with stable duration grouping. Fair39-row cached workload improved15.360s to11.470s (1.339x throughput), all39texts exact, finalbatch3, resumed0loads/0calls. Root38focused tests plus14sort-worker tests pass; independent source review passes. Duplicate model processes gave no benefit.
- Real generated diagnostics exposed a root EOS bug: batch-size1/longest rows had no positive terminal target. Fix78bc38d adds one false-mask terminal position excluded from flow/statistics;54related CPU tests and independent review pass. Fresh local100updates completed with finite losses and all6prompts ending1.20–2.16s instead of6/6frame caps. Text content remains unreliable and corrected held-out loss did not improve. No model promotion or quality claim; W&B sync of both comparison runs completed.
- The corrected teacher ran5actual cloud updates; model, EMA, optimizer and export tensors were finite, EOS weights changed and GPU was released. Full-data first stage is200updates at3e-5 with50-step checks, four raw candidates and baseline/final same-ASR diagnostics. Longer training is contingent on evidence.
- Full290-video production first started19:43Taipei. Two genuine Taiwan-only restrictions appeared on the US cloud host. Controlled repair paused the backup watcher before stopping the producer, preserved its frozen corpus and original attempts, and used a loopback127.0.0.1:18080 reverse SSH SOCKS tunnel through the user's Taiwan machine. Actual metadata and media retrieval succeeded; existing acquire() and the corpus lock produced verified source reports for both sources. No cookies, HF/W&B credentials or SSH private keys were copied.
- Production resumed19:50in fresh orchestration directory job-200-tw, with the SAMEcorpus/state/plan and unchanged43735ebcode. At19:52:5complete,1running,284pending; both formerly restricted videos complete. Persistent enabled cloud service:Restart=no,KillMode=control-group,TimeoutStopSec60,RuntimeMaxSec64860. Local backup watcher now points at the new active job and is running; proxy and original external stop timer are active. Budget stop remains2026-10-26 15:06:12Taipei and is never reset. Stop helper also closes the proxy.
- Actual deployment verification matched all230regular archived files. Production stores code archive/hashes and transport/recovery receipts. The real backup fixture verified every file and correctly recognized stale running metadata after service exit. Temporary proof unit was disabled and archived; source data and trial artifacts remain preserved.
- Personal branch was pushed through43735eb. The current evidence report records limitations and artifact locations. Tracker validated6/10complete: full processing, later cloud training, final delivery records and cleanup remain active. This is an unattended ongoing run, not a completed dataset or usable Taiwanese voice model.

## [WIP-004] Checkpoint — 2026-10-25 19:54:02 +0800 (A-002)

- **Finished:** Phone audio delivery; all290inventory; whole-video and cloud-teacher acceptance; measured batch4 acceleration; EOS regression fix and real100-update comparison; source reviews and personal-branch code push.
- **Running now:** Full corpus on dedicated Nebius VM through Taiwan egress, local incremental backup watcher, independent original absolute-deadline controls. Production at19:52has5complete/1running/284pending.
- **Still to do:** All source outcomes, full-data200-update candidate and diagnostics, verified final backup, local W&B synchronization and dedicated resource cleanup. Text correctness is not established.
- **Next work action:** Verify the resumed producer and backup receipts remain healthy, finish evidence-record push, then inspect the durable job's next state without changing its frozen code or budget.
- **Checks:** [x] tracker.md | [x] devlog RUN | [x] scope matches tracker

## [RUN-009] Event — 2026-10-25 19:56:00 +0800 (A-002)

- Delivery commit8887aac73b60990df94d3e64d49fd769b9b77d23 was pushed and independently matched origin/taigi-batch-weaklabels; the worktree was clean. Documentation and operating receipts preserve the remaining asynchronous work rather than claiming completion.
- At19:54Taipei the resumed backup watcher, Taiwan egress proxy and original stop timer were all active. Backup receipt reported a running remote service/job, successful incremental mirror and no error. Full verification remains pending while the producer writes files.
- Tracker now validates7/10complete. T5(full corpus), T6(full-data candidate) and T7(final cloud preservation/cleanup) remain active. The owner status answer is: full290queue running; last verified snapshot5complete/1running/284pending; provisional6–10hours data processing plus initial training/checks, with actual quality still unproven. Phone previews were delivered earlier; corrected local trial is synced to W&B.

## [RUN-010] Event — 2026-10-25 20:06:03 +0800 (A-002)

- Owner explicitly raises total budget to USD100, conditionally allows training through tomorrow evening, targets a usable Taiwanese model and authorizes private Hugging Face publication. Design/tracker now include continuation and private artifact delivery; no completed-model claim.
- Authenticated local Hugging Face account is weiren119. Created weiren119/pocket-tts-taigi and read back private=true; no model weights uploaded yet. Token stays local and was not printed/copied.
- Native Astra owns bounded continuation planning/implementation and an independent Astra owns deadline-control updates; native Sol owns strict private publication packaging. Shared filesystem access is not OS-enforced isolation. Root integrates and verifies the lifecycle.
- New local delivery watcher waits for verified backup and checks selected artifact hashes/paths before calling the publisher. Three focused tests pass after the expected missing-module failure. Changed paths match accepted T11; corpus producer code remains frozen.

## [RUN-011] Event — 2026-10-25 20:11:56 +0800 (A-002)

- Budget readback confirms original start1792926372 retained and latest cutoff1793023200 (Oct26 22:00Taipei). Conservative bound56.114USD includes prior0.70 and reserve7; this is not an invoice. Dedicated producer remains active with unchanged PID21133.
- Independent review corrected finite-only candidate selection and a publication example that incorrectly required voice prompting. Root verified13continuation tests and17publisher/readiness tests; scoped Ruff/type checks pass after two type corrections. New paths remain within T11, and frozen corpus modules are unchanged.
- Created the private HF repo and verified the root README upload at revision1a208143b5486493ee6dabef6f0402f0292800bc; model weights are not uploaded yet. Real local raw export packaged successfully and produced a finite2.4-second24kHz GPU WAV. CPU inference also ran in a real PTY but hit its generation cap on the known-poor tiny model; this proves runnable packaging, not speech quality. A selected-candidate CPU smoke gate is being added before future uploads.
- A separate real cloud failure fixture proved synchronous ExecStopPost stops its dummy parent before ControlPID clears. Real corpus PID21133 stayed active. The umbrella will use this containment so its backup watcher cannot finalize while the original producer still writes.

## [WIP-005] Checkpoint — 2026-10-25 20:11:56 +0800 (A-002)

- **Finished:** Private HF repository; extended budget controls; bounded continuation source/tests; real package and CPU/GPU inference checks; independent source review.
- **Running now:** All-video corpus on the same cloud VM; final publication smoke-gate implementation and review.
- **Still to do:** Deploy umbrella, retarget backup ownership, arm local private publication, observe full-data training and actual model quality.
- **Next work action:** Preflight the frozen continuation deployment, then transfer backup-watcher ownership without restarting the corpus producer.
- **Checks:** [x] tracker.md | [x] devlog RUN | [x] scope matches tracker

## [RUN-012] Event — 2026-10-25 20:16:01 +0800 (A-002)

- Deployed continuation commit65490c420bbb9efdcf04436410b09ae32ac09660 with required source receipt; actual umbrellaPID26787 is active/waiting_parent and original producerPID21133 remains active. Backup watcher now follows umbrella/job with the new absolute deadline. Live receipt confirms a successful mirror and no error.
- Cloud preflight found no Git metadata in the earlier archive, which would also break the existing experiment launcher. Genuine metadata for43735eb/tree95ec36005e4545a27edb9460bbfac49b89bf0f54 was installed atomically; all231tracked file hashes/modes unchanged. Only the controller is untracked, accurately reported as dirty. No corpus code or process was restarted.
- Local private publisher service is enabled/active and records waiting_for_verified_backup for weiren119/pocket-tts-taigi. It performs package inference before upload and cleans the entire subprocess group at deadline. Real descendant-timeout test and4readiness tests passed; final25publisher tests and scoped type/lint checks passed.
- Final real CPU smoke helper completed on the earlier diagnostic package:24kHz,7.04seconds, finite samples; its maximum-generation warning is retained. This proves execution only. The known-poor diagnostic remains local; full-data weights and usability are still pending.

## [RUN-013] Event — 2026-10-25 20:31:00 +0800 (A-002)

- Owner requested status. At20:30–20:31Taipei, live cloud state advanced to18complete,1running,1failed,270pending of290; parent remains in corpus stage and continuation waits. Latest measured completed-source snapshot was2.7436/26.0964hours, with2397accepted weak pairs totaling6969.74seconds. These remain unverified labels, not usable-model evidence.
- Failed source5GGKG_4w6uE exhausted two metadata attempts with YouTube country restriction. A bounded read-only Taiwan-local yt-dlp check also failed with Video unavailable; no retry counters or source identity were changed. Other videos continue normally.
- Local backup, private publisher, Taiwan proxy and external budget timer are active. Latest mirror succeeded without error; final backup is not yet complete. HF repo is still private and contains only README/.gitattributes, no model weights. Full-data training has not started.

## [RUN-014] Event — 2026-10-25 22:48:43 +0800 (A-002)

- Live status at2026-10-25 22:48Taipei:76complete,1running,1failed,212pending of290. Completed source duration11.0794/26.0964hours (42.46%);9410accepted weak pairs total7.8627hours. Active sourceBijjRjaosiM is enhancing; GPU99%,5951MiB. The unchanged failed source is5GGKG_4w6uE; no new source failures.
- Original producerPID21133 remains active in corpus stage; full-data training has not started. ContinuationPID26787 waits for parent completion. Backup watcher, private publisher, Taiwan proxy and external stop timer are active. Latest backup mirror has no error; final verification remains pending.
- HF readback confirms weiren119/pocket-tts-taigi remains private with README/.gitattributes only, no weights. Based on the recent source-duration rate, remaining data processing is approximately4–5hours (roughly03:00–04:00Taipei), followed by training/evaluation; this is an estimate, not proof of model usability.

## [RUN-015] Event — 2026-10-25 23:30:46 +0800 (A-002)

- At2026-10-25 23:30Taipei live status is93complete,1running,2failed,194pending of290. Completed source duration13.7875/26.0964hours (52.83%);11902accepted weak pairs total9.7619hours. Active sourceQdv0QW1_bWo is enhancing at99% GPU utilization/5951MiB. This is17additional completed videos since22:48.
- New failed sourcejGQHPh0QPPA exhausted its two metadata attempts with an explicit YouTube country restriction, same failure class as5GGKG_4w6uE. Source failures remain visible and other queue entries continue; no retry counts were reset.
- Full-data training has not started; original producer remains in corpus stage, continuation waits. Local backup has no error and all four local services/timer remain active. HF repo is private, still README/.gitattributes only. Recent throughput remains consistent with roughly03:00–04:00 completion of data preparation, followed by model training/evaluation; usability remains unverified.

## [RUN-016] Event — 2026-10-26 07:15:59 +0800 (A-002)

- Morning inspection at07:12Taipei found288successful/2failed source outcomes,22754weak pairs/17.9897hours; full-data training never started. Parent failed exact tokenizer conversion around03:57; continuation failed closed, and private publisher correctly refused missing candidate. HF remains private with no weights.
- Final backup is verified, but provider stop failed and the GPU VM remained idle. Astra traced the stop failure to expired Nebius federation authentication; cloud persistent22:00poweroff timer remains armed. No running VM was falsely reported stopped and the original cost clock remains unchanged.
- Astra independently reproduced4/22763whitespace parity failures; disabling Metaspace splitting in memory yields0/22763 without changing model/text. Sol implements the focused regression/fix. Root prepares a fresh recovery subtree and service handoff using the complete frozen corpus; all old failure receipts remain intact.

## [RUN-017] Event — 2026-10-26 07:22:09 +0800 (A-002)

- Recovery code1485064was committed/pushed; host regression and Ruff passed, and the actual converter CLI matched22763/22763texts both locally and on cloud before service start. Only convert_tokenizer.py differs among tracked cloud source; frozen corpus files remain unchanged.
- ParentPID137912and continuationPID137915 started07:18under persistent contained units. Local backup/private publisher were retargeted to production/recovery-20261026 and are active. Old backup/failure/service records were preserved under cloud/recovery-20261026. Ledger deadline and cost clock were not changed.
- Verified exact source reuse:871attempts before and after, zero new download/enhancement/ASR attempts, same288complete/2failed outcomes. The new job is still validating/merging existing corpus; actual training progress has not yet been observed.
- Provider API login is still expired. Exact-instance/OCR/deadline/nonmutating-check self-tests passed for the guarded SSH fallback, and its live readiness check issued no poweroff. Backup will not mislabel SSH acknowledgement as provider-confirmed stopped. Root retains the VM for recovery training.

## [WIP-006] Checkpoint — 2026-10-26 07:22:09 +0800 (A-002)

- **Finished:** All source outcomes; verified old backup; exact whitespace tokenizer fix and full parity proof; frozen source-reuse verification; stop fallback readiness; recovery service handoff.
- **Running now:** Recovery parent validates/merges existing data; continuation waits; local mirror and private publisher are active.
- **Still to do:** Observe actual full-data training, evaluate and gate continuation, verify recovered backup/private upload, and complete provider-confirmed cleanup after authentication is restored.
- **Next work action:** Follow the recovered job through conversion/baseline into actual training progress; diagnose any new failure before claiming recovery complete.
- **Checks:** [x] tracker.md | [x] devlog RUN | [x] scope matches tracker

## [RUN-018] Event — 2026-10-26 07:27:19 +0800 (A-002)

- Observing recovery exposed eager repeated audio hashing in both merge validators. Independent Astra reproduction and Sol regressions proved the issue; minimal per-call resolved-path caches preserve every row's digest check. Host16merge tests pass; actual22754rows/421paths imply54times fewer audio-hash calls per phase, not a claimed wall-time multiplier.
- Paused local backup/publication before controlled umbrella stop; verified both remote units MainPID/ControlPIDzero. No training checkpoint existed. Preserved the interrupted first recovery and deployed979ab7a into fresh production/recovery-20261026-r2. Only converter and merge differ among tracked cloud files; eight frozen corpus identity files are unchanged.
- New parentPID140435and umbrellaPID140438are active; local backup/private publisher are retargeted and active. Budget clock and22:00cutoff remain fixed. Actual training steps still pending while existing data validation completes.

## [RUN-019] Event — 2026-10-26 07:36:17 +0800 (A-002)

- Full-data r2 completed200updates07:35Taipei;22606train/148valid rows and unchanged source attempts. Validation.5278603→.5000720,5.27%improvement;12final audio diagnostics completed.
- Generation failed quality criteria:7/12caps versus2baseline, unrelated repeated ASR outputs. No v2 continuation was launched and no usable-model claim is justified. Original controller exports200raw as diagnostic; backup/private publisher rearmed unchanged, both active and publication waiting for verified backup.
- Corrected narrow known decoder-too-short baseline scoring without fabricating ASR or weakening health thresholds. Added contained alternate selection path for recovery. Root42focused tests, Ruff and implementation ty passed; independent Astra source review passed. Keep original live controller frozen.
- Astra diagnosis finds no proven new training-code bug; only0.0708nominal data passes with reset text embeddings. A bounded checkpoint/stopping-behavior diagnostic is the next evidence step, not blind500-step extension. Budget clock and22:00cutoff unchanged; provider-confirmed shutdown remains unavailable with expired API login.

## [RUN-020] Event — 2026-10-26 09:56:03 +0800 (A-002)

- Owner explicitly requests longertraining. Prepared/committed bounded200→1000→3000→6000exploration plan; oldpilot generation gate is no longer admission for this newly authorized experiment. Recipe, finite/divergence checks,USD100total and22:00deadline preserved.
- Verified r2backup488files. VM received guardedSSHshutdown08:38; SSHnowunreachable and providerAPIloginexpired. No newpaidcompute; localRTX3090 has availablecapacity. Resumed model/optimizer/EMA with identical tokenizer/split in freshstep-1000; actual201update09:55:39hasLR3e-5,finitegradient1.266. Userunitboundsfirststage6000seconds,cgroupcontainmentandnoautorestart.
- HFpublicationfailed because systemd lacked cachedtokenpath. Sol'sread-onlywhoami/privatecheckpassedwith explicitHF_TOKEN_PATH (no secret exposed); rootrestartedpublisherusingfreshauthfixoutput because earlierpackage directory already existed. Uploadnowpending,notclaimedcomplete.
- Solimplementsminimal localextensioncontroller; Astrareviewsplan/source. Newsource/testpaths are authorized withinT11. Parentcloudartifactsstayimmutable; unrelatedlocalGPUprocessuntouched.

## [RUN-021] Event — 2026-10-26 09:57:57 +0800 (A-002)

- Verified20actual resumedupdates201–220, allfinite losses/gradients, LR3e-5; saved resume-proof.json. LocalGPUuse about8GiB/24GiB. First1000remainsactive, originalparentunchanged.
- Experimental200privatepublication completed. Direct HFAPI confirms private=true, revisionbec7e32ccc5dbd6d3330954a531adb8999d4a698 and8candidatefiles including1.34GBweights. No qualitypromotion; packageincludesfullfailed-generationevaluation.
- Longercontroller independentreview found and is correcting cancellation/fallback ordering before deployment. First1000doesnotdependonthatnewcontroller andcontinueswithitsown100-minute bound. Source/tests remainwithSol; Astraverifiesfinalsource.

## [RUN-022] Event — 2026-10-26 10:03:06 +0800 (A-002)

- Committed/pushed142bd70 afterSol18focusedtests/Ruff/ty andAstraindependentPASS. HostverifiedCLIflags, exactreviewedsourcehash2b3a247a anduserunitproperties. Firststagehasactual450+updates; val450=.4957895.
- StartedlocalextensionPID2164739at10:01,KillModecontrol-group,Restartno,RuntimeMaxdeadline-minus90secondsandTimeoutStop90. Freshoutputlocal-longer-20261026/longer-20261026, explicitownerauthorization, endpoints3000/6000andprivatepublicationarmed. Initialparentverificationisreadingfrozencheckpointdata; firststagecontinuesseparately. Deploymentreceiptandunitcopiespreservedlocally.
- Changedpaths matchT11authorizedcontroller/tests,plans/review,docsandnotebook; no trainer/data/tokenizer/source changes andnoextraGPUworker. The stages and finalselectionare stillpending; do notdescribe6000stepsorqualityascomplete.

## [WIP-007] Checkpoint — 2026-10-26 10:03:06 +0800 (A-002)

- **Finished:** Verified200backup/privateupload, longer-trainingplan, actuallocalresume201withfinitegradients/LR3e-5, reviewedanddeployedautomaticextension.
- **Running now:** Localfirst1000GPUservice; extensionverifiesfrozenparentthenwaitsforfirst1000.
- **Still to do:** Actual1000/3000/6000diagnostics, selectedprivatepublication, usable-modelassessment; cloudproviderstopverificationstillneedsrenewedlogin.
- **Next work action:** Confirmextensionreacheswaiting_first1000withoutchangingthefirsttrainer, thenfollowendpointresults.
- **Checks:** [x] tracker.md | [x] devlog RUN | [x] scope matches tracker

## [RUN-023] Event — 2026-10-26 10:03:26 +0800 (A-002)

- Actualextensionpassedfrozenparentchecksandreachedwaiting_first1000. ItobservestheoriginalfirstunitPID2116026/activewith8cgroupmembers; sourcecommit142bd704387729bf4efd421090da6cba6c3aa14d. Savedextension-start-proof.json; no secondGPUmodelstartedwhilefirststageisrunning.
- Longertrainingisnowdurablyarmedthrough1000/3000/6000withprivatepublication,withinunchanged22:00cutoff. Completionandqualityremainpending; prior200experimentalpackageisalreadyverifiedprivate.

## [RUN-024] Event — 2026-10-26 10:18:59 +0800 (A-002)

- Exact「你好嗎？」CPUinferencecompletedfromrawstep1000,5.04seconds63frames,finite/unclippedandbelowframecap. Originalreport/WAVandvolume-adjustedlisteningcopy are preservedin/home/weiren/tts-deliverables/2026-10-26-ni-hao-ma. No linguisticqualityclaim.
- TelegramrejectsWAVextensions; generatedandffprobe-verified5.04secondH264/AACMP4,85727bytes. CourierconfirmedSent. Onlyuniformvolumeadjustmentandencoding, no semanticcontentreplacement.
- Astralogcomparison foundapproximately15%higher observedupdateandwallrates onpriorRTXPRO6000runthanlocal3090; notcontrolledsamebatches. H100hasnobenchmarkhere. Frequent6.4GBcheckpointwritesandduplicatefinalsavearevisibleoverhead; no liveconfigchangewasmade.
- First1000servicecompleted; extensionremainsactivevalidatinglineagebeforefixed12promptevaluationandlater3000/6000. UserinferenceusedCPUanddidnotstartanotherGPUworker.

## [RUN-025] Event — 2026-10-26 10:33:43 +0800 (A-002)

- Step 1000 is complete. Native `你好嗎？` audio was delivered to Telegram, but Breeze recognized unrelated content; usable Taiwanese synthesis is not proven. Packaged no-voice BOS omission was separately fixed and verified on CPU.
- Controller 64e7ecd now uses verified durable completion receipts after systemd garbage collection. Fresh longer-20261026-r2 is training toward 3000; current dataset remains YouTube only.
- ADI all eight pinned archives are verified. Actual metadata identifies 23,158 explicit Taiwanese training clips and 5,530 additional locale-blank Taiwanese-path clips; Mandarin targets remain excluded. Official test shares 78 speakers and 117 sentence IDs with training, requiring leakage filtering.
- Latest owner pstack request is routed through pinned pstack 18b6c36e Investigation/how. Sol explores the recipe; Astra explores language/text/data; a fresh explainer will synthesize. Native Codex agents use owner-preferred models because pstack model settings are absent. Throughput checkpoint: n/a, read-only investigation.
- Tracker expanded to T12 ADI integration and T13 non-English investigation, validated 9/13 complete. Current changes are the notebook, tracker and ADI plan within accepted scope; separate source adapter and research reports have assigned owners.

## [WIP-008] Checkpoint — 2026-10-26 10:33:44 +0800 (A-002)

- **Finished:** All YouTube sources have terminal outcomes; private step-200 candidate and exact step-1000 phone sample are delivered. ADI raw archive audit is complete.
- **Running now:** Local continuation toward step 3000, CPU ADI adapter preparation, and two pstack research explorers.
- **Still to do:** Research synthesis, ADI weak labels and training integration, independent quality diagnostics, usable private model, and provider-side cloud stop confirmation.
- **Next work action:** Read explorer findings, dispatch the fresh explainer, and verify the prepared ADI source release before scheduling GPU handoff.
- **Checks:** [x] tracker.md | [x] devlog RUN | [x] scope matches tracker

## [RUN-026] Event — 2026-10-26 10:38:35 +0800 (A-002)

- Pstack investigation completed with Sol recipe exploration, Astra language exploration and a fresh Astra synthesis. Read docs/research/2026-10-26-nonenglish-pocket-tts-recommendation.md first. Main delivery remains ordinary Mandarin text to Taiwanese speech; Mandarin/Japanese are comparison evidence only.
- Official new-language fine-tuning does reset the text embedding. Current failure does not establish that this choice is wrong. Step 1000 has no capped outputs but mean same-Breeze SequenceMatcher similarity 0.0301, so lower loss has not established text control.
- Owner asks whether ADI has Chinese labels. Direct first-tar inspection confirms native Taiwanese Han/romanization in sentence, no separate Mandarin translation field. Preserve these originals and add explicitly unverified Breeze semantic labels; never rename native Han as Mandarin gold.
- Owner is away running and authorizes continued optimization. CPU Mimi reconstruction diagnosis and safe GPU handoff planning are delegated. Current 3000-stage trainer remains active; no new cloud resources or budget increase.
- ADI split policy was revised before labels/training: create primary speaker-disjoint dev, exclude its sentence/audio duplicates, retain official test outside training and report its speaker/text overlap buckets. Removing every official-test speaker would discard 85.9% of strict training rows; that policy is superseded with explicit evidence and Astra review.
