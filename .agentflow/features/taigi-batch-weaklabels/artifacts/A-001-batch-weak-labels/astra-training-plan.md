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
