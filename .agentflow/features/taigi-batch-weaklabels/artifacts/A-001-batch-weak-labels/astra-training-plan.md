# Training plan for Chinese input and Taiwanese speech

- Recorded: 2026-10-25 18:49 +0800 (Asia/Taipei).

- Advisor: Model `gpt-6-astra`; reasoning effort inherited.

- Inspected commit: `6fd16f5a0f9bdfd3689b8306772146a92b63cb5f`. Concurrent uncommitted batch/merge work was not reviewed. This report is advice, not authorization to expand implementation scope.

**Summary**

- Finish the current bounded batch, then run a small local learning experiment with actual held-out sources and generated audio. Fifteen minutes of source video is a pipeline milestone, not sufficient evidence of useful general Chinese-to-Taiwanese speech.

- The existing 24-layer warm-start, new text embeddings and LSD objective are a reasonable starting point. Twenty steps and finite loss establish execution only; the main unknowns are label meaning, complete speech segments and whether generation actually follows Chinese input.

- Prefer the already identified ADI sentence candidates for the next data expansion; propose bounded interview windows afterward. Keep the approximately US$49.30 remaining cloud allowance unspent until local free generation shows a useful signal. No additional cloud resources are needed for the next experiment.

## What the inspected code establishes

- `training/configs/taigi_smoke.yaml` starts the released English 24-layer model, resets the text embedding, uses LSD, freezes latent statistics, and performs only 20 single-example updates with no validation. The new-language example `training/configs/finetune_language.yaml` supports warm-start plus reset embeddings, but its comment describes 976 hours of Czech training. Its learning rate and batch choices are precedent, not evidence for fifteen minutes of weak cross-language pairs.

- `training/modules/builders.py` removes pretrained `conditioner.embed.*` when `reset_text_embedding: true`; the rest of the compatible model transfers. This is necessary when token IDs change even if vocabulary size happens to remain equal. New tokenizer means a fresh adaptation run; do not silently resume the 512-token smoke checkpoint with different IDs. The requested `training/finetune.py` does not exist; fine-tuning uses the ordinary trainer and YAML configs.

- `freeze_head` in `training/train.py` freezes only `flow_net`, not the transformer or EOS predictor. Keep it **false** for the first new-language experiment. Keep Mimi frozen as the builder already does. Do not freeze the backbone or enable CFG/head distillation: an English teacher does not supply the missing Chinese-to-Taiwanese mapping. In this run “teacher” means the larger model we are adapting, not a second frozen language teacher supervising it.

- Keep `flow.type: lsd`. Changing to ordinary flow matching changes the head's time inputs and initializes that head anew in the builder. There is no demonstrated benefit here that justifies throwing away the pretrained head. The current LSD path predicts recorded audio latents plus EOS; it has no separate translation or pronunciation correctness loss.

- Sentence mode preserves the complete supplied text and audio and rejects overlong targets; absent independent prompts produce zero prompt frames. This avoids target leakage, but does not prove the upstream segment is a complete sentence. No-prompt training on many voices also does not establish controllable voice cloning.

## Immediate local experiment after the producer finishes

1. Freeze one release and record **accepted speech seconds**, unique recordings, rows, source groups, and split membership. The reported 905 source seconds / 56 windows is an upper bound before speech filtering, not training duration. Group all excerpts from a recording together; related previews from the same program/episode or reuploads may need a shared family group. A video-ID-only holdout can still contain the same scene or cast from training.

2. Hold out several genuinely distinct source families if available. Use one fixed development set for checkpoint selection, and reserve an untouched source group for a final check only if the small dataset permits it. Otherwise call the result development evidence, not independent test quality. For ADI expansion, hold out client IDs and prevent repeated sentence IDs from crossing splits. Report actual group counts; a percentage alone hides a weak split.

3. Build the tokenizer from training labels only, then freeze it. Check exact encode/decode parity, unknown-token rates, token lengths and unseen Traditional Chinese characters on held-out text and fixed user-style prompts. Byte fallback can avoid unknown tokens but cannot teach the meaning of unseen words. Do not enlarge 512 to 4,000 merely because the larger number sounds better; choose a vocabulary that fits the available text and passes coverage checks.

4. Use a separate run directory with explicit settings: 24-layer pretrained initialization; reset embeddings; unfrozen flow head; LSD; sentence mode; `precompute: false`; `compile: false`; `stats_ema_steps: 0`; `flow_batch_multiplier: 1`; `text_dropout: 0`; `voice_dropout: 0`; `sample_cfg_coef: 1`. These settings keep the first comparison close to the verified smoke and avoid testing untrained unconditional Chinese behavior through CFG. Keep `ema_decay: 0.999` for continuity, but evaluate raw weights first: after a few hundred steps EMA still contains much of the initialization.

5. Start with microbatch 1 and accumulation 8, learning rate `1e-4`, warm-up 10 updates, and a **100-update ceiling**; validate/checkpoint every 25 updates. Apply the existing bounded runner with a 720-second training deadline and reserve 180 seconds for export and diagnostics (15 minutes total local budget); measured throughput determines the actual steps reached, so do not promise 100 updates within that window. This is a conservative proposed diagnostic, not a proven optimum. Record examples seen (`updates × effective batch`) and approximate passes through the accepted rows: even this budget can repeatedly expose a tiny corpus. Measure actual memory/seconds per update on representative long rows before increasing microbatch size. Do not adopt the repository's effective-batch-64 comment as a guaranteed quality threshold for this new task.

6. If the trainer can fit recorded audio but free generation remains blank, repeats, or ignores changed Chinese, inspect labels and the conditioning path before extending the run. If both held-out diagnostics and prompt sensitivity improve, propose one continuation capped around 1,000 total updates, with early stopping on repeated regression. Compare a lower learning rate only when instability or overfitting provides a reason; avoid a broad search on this small corpus.

## Evaluation that can answer the real question

- Generate the same fixed 12–20 Traditional Chinese prompts at initialization and every saved checkpoint with fixed seeds, sampling settings and zero voice prompt. Include seen training sentences as a memorization check, unseen sentences, and pairs differing in negation, a number or a location. Distinguish “audio changes” from “the intended meaning changes correctly.” If independent same-speaker prompts are later added, evaluate that as a separate condition using the same fixed references across checkpoints.

- Record termination, zero-length/silence failures, clipping/nonfinite samples, duration, repeated phrases and Breeze's recovered Chinese. Report counts and examples. Breeze-based agreement is an automatic consistency measure with the labeling system, not independent Taiwanese pronunciation or semantic truth. Chinese characters do not establish that the generated language is Taiwanese rather than Mandarin.

- Run Mimi reconstruction on a small fixed held-out audio subset when the GPU is free. Compare source and reconstructed audio with the same automatic checks. If the frozen codec itself loses useful speech information, generator loss alone cannot explain the failure. This is an attainable local diagnostic, not a demand for another model or download.

- `validate()` uses a loader with `shuffle=False`, but the loader repeats forever and the flow objective draws random noise. Fifty validation batches may revisit the same rows; they are not fifty independent samples. Count unique rows, bound evaluation to a documented pass, account for any dropped partial batch, and use isolated fixed RNG state for comparable checkpoint diagnostics. Batch-average loss also differs from a corpus-wide frame-weighted score; name the aggregation used. Minimal implementation: keep validation batch size 1, set `num_valid_batches` to the verified number of held-out rows, and wrap the validation model/noise computation in `torch.random.fork_rng` with the active CUDA device and a fixed `torch.manual_seed`. Evaluate the same baseline before optimization and each checkpoint inside that isolated RNG scope. Confirm actual loader success/row coverage so rejected rows cannot silently cause cycling. Preserve the training RNG around added deterministic evaluation; test equal metrics at unchanged weights and unchanged caller RNG states. A standalone evaluator is also valid, but adds more wiring than this small wrapper.

- Teacher-forced validation sees preceding true audio frames, so low loss can coexist with poor free generation or weak dependence on text. Add a small fixed comparison with correct versus shuffled Chinese conditioning on the same held-out audio/noise if practical. A gap supports use of text; it does not establish translation quality. No automated check currently verifies Taiwanese tones or naturalness independently, and the user need not label every clip to keep this limitation explicit.

## Next data stage and budget

- Prefer the existing ADI shortlist: the pilot report identifies 911 initial candidates after sentence deduplication and client caps. Acquire and measure a finite first tranche, for example 200 candidates, run the corrected Breeze adapter, and report accepted seconds and failures before admitting the rest. These counts do not imply hours: the TSV has no durations. Retain Chinese semantic labels, not unconverted Taiwanese orthography, as the training input contract.

- Treat 1–5 accepted hours as a proposed next feasibility scale, **not a sufficiency claim**. Expand incrementally only while accepted quality and local generation diagnostics improve. The current short previews may teach dramatic voices, music and edited dialogue; more windows from those same previews do not add much semantic diversity.

- Long interviews are a useful proposed source of broader sentences, but five roughly fifty-minute files must not silently bypass the current 300-second admission cap. A separate bounded long-source implementation should download each approved source once, retain exact provenance, and select a finite number of nonoverlapping windows across episodes. Require complete speech boundaries and isolate overlapping speakers or uncertain language; do not count Mandarin interview sections as Taiwanese because Breeze outputs Traditional Chinese. This extension is a proposal, not work authorized by this advisory report.

- RE-USE and Breeze remain serial local production stages until measured otherwise. Raw/enhanced agreement flags instability; it does not prove either transcript is correct or enhanced audio is better. Preserve originals, quarantine empty/repetitive/implausibly short labels and obvious boundary truncations, and separately flag numbers, negation and names when external subtitle evidence disagrees. Checks requiring new language/overlap models should be proposed explicitly, not presented as already implemented.

- Do not spend cloud money just to produce more optimizer steps on a tiny corpus. After a local signal and a larger frozen release, price any cloud run from its actual GPU rate, measured throughput, storage, transfer and shutdown margin; enforce the remaining **total** budget. The approximately US$0.70 already spent is an estimate, not an invoice. Distillation and CPU benchmarking should follow useful teacher behavior, not precede it.

## Delivery and verification boundaries

- This advisory task ran no training, downloaded no models, allocated no cloud resources and wrote no dataset. Evidence is the inspected local code plus `docs/plans/2026-10-25-taigi-pilot-results.md`, `docs/plans/2026-10-25-youtube-weak-label-feasibility.md`, the earlier research record, and this feature's design. Historical research statements are not new measurements or newly verified external literature.

- The integrating agent should record the chosen recipe and rejected alternatives in the notebook/route record, update STATUS with accepted data counts and current experiment state, and commit the final config/evaluation changes with this report. Necessary verification includes actual release split/hash checks, tokenizer parity, real loader coverage/no target prompt, a successful finite train/validation run, checkpoint restoration, and fixed free-generation artifacts. Add focused tests only for new evaluation behavior such as RNG preservation and unique-row accounting. Existing passed pipeline tests are not training-quality acceptance.

Self-check: Recommendations distinguish the authorized bounded batch from proposed expansion, execution from quality, weak labels from verified semantics, and repeated batches from independent evidence. No claim of good Taiwanese synthesis is made. The immediate next step is a bounded local experiment with held-out source groups and comparable generated audio; broader data acquisition and cloud spending remain proposals subject to the captured owner scope.

## Supplement — actual 100-update diagnostic and next training decision

* _2026-10-25 19:29:10 +0800 (gpt-6-astra/inherited)_

**Decision**

- Do not extend the tiny local run merely because training loss is finite. Every measured post-update held-out loss is worse than its initialization baseline. Finish its fixed-prompt raw-weight generation and ASR consistency comparison, and preserve the failed-to-improve result.

- For the new full-corpus release, use a first **200-update diagnostic ceiling**, not an automatic 2,000-update run. A conservative proposed learning rate is `3e-5`, retaining warmup 100, microbatch 1 and accumulation 8. This rate is an untuned risk-reduction choice, not a measured optimum. Keep the architecture and objective unchanged.

- Validate and save raw checkpoints every 50 updates. Preserve these four candidate raw checkpoints and the seeded initialization reference. Decide whether to continue only after this new release's own held-out and generated-audio evidence exists; 2,000 updates remains a conditional upper bound inside the original time and money ceiling.

### Observed evidence

Read the actual `diagnostic-v4/training.yaml`, `training/progress.jsonl`, `preparation.json`, merged metadata and baseline generation report under `/mnt/HDD2/proj_tts/pocket-tts/data/taigi-batch-v1/`.

- The immutable filtered release is `c7977dd525b91eea`: 165 training rows containing **381.14 seconds** across four playlist groups, and 28 validation rows containing **72.68 seconds** from one other playlist group. Two pathological ASR rows were quarantined. This is roughly six minutes of accepted training audio, not the full 26.1 source-hour inventory.

- The run uses an 813-piece BPE tokenizer, freshly reset text embeddings, the pretrained 24-layer model, LSD, sentence mode, learning rate `1e-4`, warmup 10, microbatch 1 and accumulation 8. One hundred updates consume 800 example slots, approximately 4.85 passes over 165 rows if every row loads once per cycle. No target voice prompt or latent-statistics updates are enabled.

| Update | Fixed held-out loss |
| --- | ---: |
| 0 | 0.544309 |
| 25 | 0.563791 |
| 50 | 0.559163 |
| 75 | 0.582931 |
| 100 | 0.580498 |

- Final loss is 0.036188 higher than baseline, about **6.65% worse** on this fixed diagnostic. The flow component increases from 0.542623 to 0.580481, while EOS loss falls from 0.016859 to 0.000169. Better EOS fitting therefore does not mean the audio-prediction component improved.

- The final logged training loss is 0.438620. Inspection of the training loop shows this log contains the **last microbatch's metrics**, not the mean of all eight accumulated microbatches or the training corpus. Do not present its difference from validation as a measured corpus-wide generalization gap.

- At inspection, the baseline produced six finite, non-silent audio files without clipping; three of six reached the ten-second frame cap. Those waveform checks do not show intelligible or correctly conditioned speech. Final raw generation and Breeze comparison were still in progress; no improvement claim is available yet.

### What this result does and does not imply

- The held-out regression is consistent with overfitting or unhelpful adaptation on a very small, differently sourced weak-label set. One held-out playlist, one run and an initialization with random new text embeddings cannot distinguish those causes or establish the best learning rate. The curve also is not strictly monotonic: step 50 improves over step 25 while remaining worse than baseline.

- The full-corpus experiment will have a different accepted duration, vocabulary and baseline. Do not compare its absolute loss directly with 0.544309 or resume this pilot using new token IDs. Record actual accepted rows, seconds, unique sources and groups after filtering; 26.1 source hours is only the input inventory total.

- Keep the same model, LSD objective, zero dropout, frozen Mimi and current data contract for the first comparison. Freezing layers, switching objectives, enlarging the model or increasing batch size would introduce additional explanations without addressing the observed evidence.

### Bounded continuation rule

1. Freeze the full release and tokenizer. Verify its train/validation row loading, record the exact fixed validation subset if capped at 100 rows, and run its own seeded step-zero validation and fixed-prompt generation. A fixed cap is a subset diagnostic; do not call it evaluation of every held-out row or every source group unless it actually covers them.

2. Run at most 200 updates with the proposed lower learning rate, warmup 100 and validation/checkpoints every 50. Retain enough raw checkpoints to compare the measured candidates rather than automatically preferring the latest. The existing checkpoint writer keeps an optimizer sidecar only for the newest checkpoint; preserving a raw model is sufficient for evaluation, but does not by itself support exact optimizer resume from an older selected checkpoint.

3. If all post-update validation scores are worse than this release's baseline and generated samples remain blank, repetitive, frequently capped or insensitive to changed prompts, stop training and retain the artifacts. Do not spend the remaining allowance solely on more steps. Investigate labels, source-group mismatch and conditioning evidence next.

4. If repeated checkpoints show held-out improvement and fixed generation avoids degeneration, a further bounded continuation may be justified. Compare raw weights on the same prompts and settings; use negation/number pairs and some training examples to distinguish basic fitting from unseen-prompt behavior. Breeze agreement remains same-model consistency, not independent Taiwanese quality. If loss and generation disagree, keep the outcome unresolved rather than selecting the model on one convenient number.

5. Any continuation must preserve the frozen data/tokenizer, record the changed training ceiling through the launcher's supported identity procedure, and retain optimizer state where needed. Do not silently edit a frozen run identity or reset the cloud spending clock. The 200-update stage replaces the initial automatic 2,000-update commitment; it is not an additional unbudgeted search. Keep backup/diagnostic reserves and the existing absolute VM stop deadline.

Self-check: This supplement distinguishes actual measurements from a proposed untuned recipe, last-microbatch logs from corpus statistics, and a six-minute pilot from the larger source inventory. It does not claim overfitting is proven, that a lower learning rate will fix the problem, or that longer training will produce useful Taiwanese speech. No code, model, GPU process, cloud resource or Git state was changed; only this advisory document was appended.

### Accepted first-stage recipe and checkpoint selection

- The coordinator accepted 200 updates, learning rate `3e-5`, warmup 100, validation/checkpoints every 50 and four retained raw checkpoints. No automatic continuation to 2,000 updates is planned; completion triggers the existing backup/stop flow without a quality claim.

- Generating from the latest checkpoint is sufficient for this initial pilot. Call it the **final-update candidate**, not the best or promoted model. Do not add automatic minimum-validation checkpoint selection now. Summarize the baseline and every measured held-out score in the evidence; if baseline remains lowest, report no held-out loss improvement even when the pipeline completes successfully.

- If an earlier retained checkpoint improves while the latest regresses, its raw weights can be evaluated in a bounded follow-up after verified backup. This keeps selection based on actual full-corpus evidence and preserves all candidates without inventing a new orchestration policy. Keep the latest optimizer sidecar; older raw candidates alone do not support exact optimizer resume.

Self-check: This clarification records the accepted bounded recipe and recommends no additional checkpoint-selection machinery for the initial pilot. Pipeline completion remains distinct from model improvement.

## Supplement — reproduced missing EOS supervision and minimal correction

* _2026-10-25 19:34:54 +0800 (gpt-6-astra/inherited)_

- **Blocking defect found and locally fixed:** The longest utterance in each training batch had no prediction position after its last real audio frame. With sentence mode and batch size one, every target mask entry was valid, so EOS received no positive target at all. Its decreasing loss could therefore reward learning never to stop. Do not interpret the earlier pilot as a clean test of learning rate or generalization; rerun from a fresh seeded initialization with the corrected objective.

- **Actual CPU reproduction:** Built the real Mimi architecture from `english_2026-04_24l.yaml` on CPU, without downloading weights or using the GPU. For 2.00, 2.04 and 3.22 seconds of input, `encode_batch` returned 25, 26 and 41 latent frames respectively, all valid, zero positive EOS supervision positions and zero voice frames. Random codec weights suffice for this frame-count check; this is not a codec-quality measurement.

- **Voice conditioning is matched:** Sentence manifests have no independent prompt; `_sample_sentence` returns empty prompt audio, `encode_batch` preserves zero voice frames, and the conditioner uses `[prefix BOS, text, latent BOS, previous audio]`. Setting `voice_dropout=0` keeps the supplied voice segment, whose length here is zero; it does not insert a hidden target prompt. The evaluator supplies the same empty segment. A CPU tiny-model comparison of full-sequence training conditioning against incremental cached conditioning matched within `4.77e-7` maximum absolute error.

- **Generation contract:** The builder loads the pretrained backbone/flow head and codec while resetting only the new text embeddings. Training normalizes codec latents, generation reverses that normalization, and both use the same LSD decoder. The evaluator reuses the trainer's raw-weight generation and Mimi decode path. Public `TTSModel` normally adds a voice prompt and uses different EOS threshold/minimum/trailing-frame defaults; this diagnostic intentionally tests the no-prompt training condition and is not an exact public-API sampling-parity claim. Those defaults cannot repair absent positive EOS training targets, so no sampling workaround was added.

- **Smallest correction:** `training/modules/model.py` now appends one padded latent position and one false mask entry inside `forward`. Consequently, the final real audio latent becomes the input for a terminal prediction. That prediction trains EOS; its false mask excludes it from the flow target and latent statistics. Existing shorter rows retain their original terminal position, and caller tensors/masks remain unchanged. No tokenizer, voice conditioning, codec, architecture or generation change was made.

- **Fail-before/pass-after evidence:** New `training/tests/test_terminal_eos.py` failed both the single-row and longest-ragged-row cases before the change: logits had three positions where four were needed. Afterward the test checks positive continuation gradients on real frames, a negative gradient that increases terminal EOS, no gradient on later padding, exact real-audio input alignment, unchanged real flow targets and an unmodified caller mask. The new regression plus sentence sampling, smoke and validation-RNG suites passed **54 tests in 4.62 seconds**, CPU only. New-test Ruff checks pass; the installed old Ruff reports existing `ANN101` and formatting issues in the pre-existing model file, so the coordinator will apply the repository's current lint/type tooling.

- **Current interpretation:** Actual generated-ASR evidence is poor: all six final prompts have zero character similarity, repetitive recovered text and frame-limit hits. Missing EOS supervision explains a demonstrated termination-training failure; it does not establish the cause of every text-control or speech-quality failure. Keep the fixed-prompt, held-out and ASR comparisons for the fresh corrected run, without promising useful Taiwanese synthesis.

Self-check: The root cause was reproduced before editing, the fix is confined to terminal prediction/masking, and tests prove that EOS is not counted as an audio flow target. Previous speculative interpretations are qualified by the newly demonstrated defect. No GPU or cloud workload was launched, no model downloaded, no Git change made and no agent delegated. Source changes are limited to the model forward path and its focused new regression test.
