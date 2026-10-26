2026-10-26 — GPT-6 Astra

# ADI integration for the current Mandarin-input Taiwanese-speech experiment

Use ADI audio now through the existing corrected Breeze semantic-weak-label path. Preserve native ADI transcripts as reference metadata, not as supposedly equivalent Mandarin translations. Keep the existing4096-token tokenizer and verified model/optimizer/EMA checkpoint. Prepare the data on CPU while the current authorized training continues; schedule GPU ASR and the new training branch at a clean stage boundary, without overlapping trainers.

## Verified source and prior work

The official dataset card describes ASR train/test WebDataset archives and companion TSVs, with a generic sample schema; it does not promise Mandarin translations or TTS suitability. Its license field is `other`. The API currently resolves revision `ffcc36b9afe31ec50f7742c0c4c614f1a5b40151`, seven train archives and one test archive, with gating disabled. Pin this revision rather than moving main. [Official dataset card](https://huggingface.co/datasets/adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hokkien/blob/ffcc36b9afe31ec50f7742c0c4c614f1a5b40151/README.md)

Reuse `docs/research/2026-10-25-adi-hokkien-evidence.json`:31,482 train rows,23,158 declared nan-tw,2,792 zh-CN,5,532 blank locale. Filename provenance identifies28,688 nan-tw and2,794 zh-CN paths. Test contains323 rows. The raw TSV uses literal escaped tab separators, and romanization contains actual ASCII0x3F replacements. I independently checked raw HTTP bytes and the first three JSON members of the cached `data/taigi-pilot/source/train/train-000000.tar`; the tar also contains `大屯橋（T?a-t?n-ki?）`, `食薰（tsia?h-hun）`, and `穩當（?n-t?ng）`. This is not a website rendering artifact. Actual tar fields include client_id, sentence_id, sentence, locale, path, audio_ext and split.

Reuse the existing bounded acquisition implementation and cached first tar rather than install another framework or redownload it. `docs/plans/2026-10-25-taigi-pilot-results.md` already records corrected ADI release `cf0ad740136ec22e`:34 weak-label rows made with the fixed16k resampling adapter. That establishes execution feasibility, not label or synthesis quality. The old pre-resampling-fix release `2066a2e8654ec2f5` remains excluded. The current step1000/longer checkpoints used YouTube only; do not claim they already incorporated these ADI samples.

## License handling

The pinned dataset's actual LICENSE is TRAIL–TWM Responsible AI License0.1, dated2025-09-30. It permits dataset use/modification; distributed dataset derivatives require TRAIL-D and attribution. Its model-derivative distribution provisions specify TRAIL-M, accessible weights or code, and license/modification/use notices. Preserve the exact license bytes/hash and provenance in the adapted release and candidate documentation. Do not label ADI as MIT based on the upstream README citation template, or erase base-model notices. Keep the requested HF repository private; do not change visibility automatically. The license's derivative-model distribution language needs to be reflected accurately rather than assuming private hosting removes obligations. None of this prevents the currently authorized local preparation/training. [Pinned dataset LICENSE](https://huggingface.co/datasets/adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hokkien/blob/ffcc36b9afe31ec50f7742c0c4c614f1a5b40151/LICENSE)

## Small adapter, explicit identities

Extend the existing acquisition path narrowly into an immutable ADI source release. Read JSON/audio member pairs from tar without extractall; reject absolute/traversal names, duplicate members, missing pairs, inconsistent metadata/path/split and malformed text. Record archive/member identity and hashes, original compressed bytes hash, decoded/resampled audio hash, actual sample rate/channels/frame count/duration, original metadata and parser version. Never overwrite the cached archive or original transcript.

Primary cohort: declared nan-tw plus matching nan-tw filename. A blank locale with unambiguous common_voice_nan-tw path may be an explicitly labelled provenance-inferred cohort; audit sample audio before including it. Exclude zh-CN paths/locales from Taiwanese targets and quarantine conflicts rather than guessing from Han characters. Retain exclusion counts by reason. Reuse validated decoding to mono24k float training audio, finite/nonempty checks and duration measurement. Select intact0.8–20s spoken clips; reject out-of-range clips with counts rather than silently truncating their audio or labels. Do not automatically run RE-USE on already recorded read speech; first preserve and inspect raw ADI audio.

The minimum useful release is a deterministic bounded cohort of at least1,000 eligible clips across multiple speakers if the cached shard supports it, with at most50 per speaker initially. This is a planning target, not a verified count or hours claim. Start with cached first-shard coverage, then acquire additional pinned shards if needed. Record actual audio hours. Do not scan all seven shards before producing the first usable audited release if the cached shard already suffices.

## Split and leakage policy before labels or training

Keep official test out of every training release. Parse train/test metadata first; identify overlap by namespaced client_id, sentence_id, original path and actual audio hash. Reserve official test speakers and duplicate recordings; remove overlapping train candidates and report exclusions rather than claiming the official split is already leakage-free. Where test labels or speaker IDs are missing, report the limitation and retain conservative grouping.

Derive development data from the remaining training speaker groups with a fixed seed/hash, aiming for about10% and enough examples for a stable bounded validation set. All clips from one speaker stay together. Exact audio duplicates link groups and cannot cross splits. For repeated sentence IDs or normalized transcript duplicates across proposed dev/train, remove the duplicate training examples or allocate the sentence to one side; do not union every common sentence with all speakers until the entire corpus becomes one unusable component. Preserve at least one independent held-out set or report that the proposed subset cannot support the claim.

Keep both native and generated labels for a given recording in the same split. Namespace group IDs with ADI/revision so they cannot accidentally collide with YouTube IDs. Recheck exact audio/content overlap against retained YouTube training/validation. Do not use dev/test outputs to train or revise the tokenizer. A row-level random split is not acceptable.

## Labels aligned to the owner's actual goal

The primary training field remains Mandarin-like semantic weak text generated by the already approved Breeze model on each exact ADI interval, with corrected16k input and pinned model/revision/adapter. Record it as `asr_pseudo`, unverified, not a gold translation. Preserve ADI's original sentence, Han-only display form, romanization annotation and sentence/client IDs beside it. ADI audio is a better-defined source, but Breeze can still hallucinate or transcribe Taiwanese orthography rather than translate; source quality does not make its new labels verified.

Do not strip parentheses and rename the remaining Taiwanese Han text “Mandarin”. `食薰` is not the same training instruction as an independently established Mandarin semantic equivalent. Native Han-to-Taiwanese TTS is a legitimate auxiliary task but not the requested direct Mandarin-input task. For this minimal branch, do not silently mix native-label rows into the same untagged input objective. Park native-orthography auxiliary training or a separate Mandarin→Taiwanese text translator as distinct proposals rather than adding architecture now.

Audit at least20 clips across speakers, duration bands and locale cohorts: compare audio, original ADI text and generated Breeze text for content, silence/noise, speaker changes and nonsensical repetition. Include everyday utterances and interrogatives, not only dictionary words. Keep the existing narrow repetition gate, reject empty/nonfinite/obvious corrupt labels, and record all reasons; do not claim linguistic validation from automated screening. Broader uncertain labels remain visibly weak, not silently corrected by invented translations.

The romanization's lost characters cannot be reconstructed by normalization. Keep original bytes; do not manufacture tones or globally delete question marks. If a Han-only reference display is needed, remove only a recognized trailing parenthetical romanization annotation, store the transformation/version, and preserve genuine punctuation. That reference normalization does not alter the main Breeze training target.

## Frozen tokenizer and checkpoint compatibility

Measure the actual new target texts with the existing tokenizer JSON: total tokens, byte-fallback token fraction, percentage of rows containing fallback, unknown IDs, token-per-Han-character distribution and worst cases, punctuation coverage, and per-token train occurrence counts. Compare with the YouTube manifest and fixed prompts, including `你好嗎？`. Unknown count alone is insufficient because byte fallback avoids unknowns while creating nearly unseen token sequences. Verify encode parity against the existing `.model` on every accepted ADI target without weakening the converter gate.

Prefer preserving the4096 vocabulary/IDs/embeddings. Report whether the new labels materially add training occurrences for previously rare punctuation and Taiwanese characters. Large fallback inflation or damaged dominant text is a reason to inspect labels, not to retrain a new tokenizer and silently invalidate the checkpoint. A vocabulary rebuild/resizing is outside the ten-hour minimal branch unless separately justified; all source bytes and fallback statistics should remain available.

Choose the latest completed, verified checkpoint/optimizer pair at a natural boundary—step3000 if completed and finite, otherwise the preserved1000. Do not read a live writer's checkpoint or edit its input manifests. Fork into a fresh ADI experiment with lineage recording the parent pair hashes, unchanged tokenizer hash, new data release/split hashes and explicitly permitted data-manifest changes. The existing fork helper currently permits only run_dir/max_steps; changing data requires a deliberate new identity, not bypassing its checks. Keep optimizer/EMA state and constant3e-5 LR, batch1/accum8 and architecture unchanged initially. Record the dataset switch at the cumulative optimizer step; no warmup or embedding reset.

## Minimal ten-hour execution sequence

1. First0–1h, CPU: reuse cached archive/audit; verify actual members; produce source selection and speaker/sentence/audio overlap report, license receipt and split manifest. Download further shards only as needed. Existing longer training continues untouched during this work.
2. Next1–2h, at a completed training boundary: one resident Breeze process generates labels for the bounded ADI cohort. Reuse batch4 and duration sorting, exact16k semantics and caches; no concurrent trainer. Measure actual throughput rather than promise a completion time from clip count. Audit20 examples and run tokenizer/parity/data validation before any new optimizer update.
3. Next2–7h: one fresh ADI adaptation branch, initially up to1,000 additional updates, preserving the old model. Make ADI the majority of the small experiment rather than bury it beneath22,606 YouTube rows. A simple deterministic manifest can include all accepted ADI training rows plus a group-balanced YouTube replay subset of at most one-third that row count (roughly75/25 by rows, actual sampling/hours reported). No new sampler is necessary. Keep separate ADI-dev and original YouTube-held-out diagnostics; do not combine losses and misinterpret a changed validation distribution as improvement.
4. At+250/+500/+1000 updates: retain finite checkpoints, actual learning rate and fixed generation diagnostics. Compare against the same parent on the same ADI dev set, alongside the existing12 prompts and exact `你好嗎？`. If time/finite/divergence bounds permit, extend this explicit ADI branch to a second1,000-update block; do not automatically restart the previous YouTube-only6000 schedule afterward.
5. Final1–2h reserved: evaluate/export the exact raw checkpoint, make the licensing/provenance-complete private candidate package, preserve failed/older variants and report actual content/listener results. Keep the unchanged totalUSD100 and22:00Taipei cutoff; shrink the cohort or optimizer target if measured throughput consumes the reserve, never silently drop held-out boundaries or validation.

No promise of usable Taiwanese tonight follows from this plan. ADI adoption addresses corpus source/coverage and supervision quality; native-listener checks and semantic paired prompts still decide whether the result follows Mandarin input. User authorization already covers adoption and preparation; no extra permission gate is proposed. Root owns scheduling the next stage boundary, actual downloads/ASR/training, dependency validation and private delivery.

Self-check: official card/license/API reviewed, existing audit/pilot/source reused, three cached tar JSON members read to verify corruption. Only this requested plan written. No paid job, GPU execution, training interruption, source edit, Git mutation, delegation or Agentflow startup.

## Superseding owner decisions after the full source audit — 2026-10-26

- Prepare all eligible declared `nan-tw` recordings across all seven training archives, without the earlier 1,000-row or 50-per-speaker pilot caps. All eight train/test archives are downloaded and verified at the pinned revision; their combined size is 730,798,080 bytes.

- Replace the proposal to exclude every official-test speaker. That rule would remove 19,893 of 23,158 declared Taiwanese training rows (85.9%) for a zero-shot speaker claim the owner did not request. No ADI training had occurred when this split decision was locked.

- Build development data from whole speaker groups in the original training split, using a fixed hash order and aiming for 10% of decoded audio duration. Client IDs are speaker proxies. Development guides model selection; it is not an unbiased final product test.

- Official-test recordings and exact audio duplicates remain excluded from training. Official-test speakers and sentence types may appear in training; report separate seen/unseen speaker and sentence buckets and do not call the aggregate a test of unseen speakers.

- Keep different recordings of the same phrase within either split. Remove training rows whose sentence ID or normalized native reference occurs in development, and remove exact audio duplicates globally with development taking precedence. Report all exclusions and unique-reference counts.

- Keep blank-locale records with a Taiwanese source filename in a separate quarantined cohort. Do not silently call their locale verified. Preserve the original UTF-8 transcript and its literal damaged romanization bytes; a Han-only display remains a native reference, not a Mandarin translation.

- Generate Mandarin semantic candidates only through the existing pinned Breeze adapter. They remain unverified weak labels. Preserve the native reference separately and inspect conflicting or unreliable pairs before admitting a new training branch.

- CPU preparation uses at most four workers and records the actual limit. A 200-clip local integration probe is not a complete prepared corpus. The superseded sequential partial remains identifiable; root owns the complete cloud preparation, GPU admission, and final dataset counts.

- The owner subsequently raised the total cloud allowance to US$150 and authorized an H100, while retaining the 22:00 Taipei cutoff. This adapter neither provisions resources nor changes an existing training run.

Self-check: This addendum records the owner's later decisions and the measured source counts. It does not claim completed full-corpus preparation, semantic-label quality, or ADI-trained model quality.
