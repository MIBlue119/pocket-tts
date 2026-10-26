# Bounded fallback: broader dictionary pairs with perturbed audio history

## Decision and failed-gate evidence

The predeclared lower-LR1200gate **failed**: matched CFG1/four-step seen mean.4175595 and retrieval5/8, versus800mean.3793651/retrieval5/8. Improvement.03819 is below.07 and retrieval is below6/8. No caps occurred. Step1000was better (.4814484,7/8), but its intermediate success does not turn the final1200gate into a pass. Heldout1200mean.047619 with a long repetitive下午 output does not establish generalization. The prior conditional data-only expansion is not admitted.

Root now selects a distinct finite fallback: **filtered broader dictionary pool plus teacher-history Gaussian noise sigma.05**, at most1000updates/45minutes, starting from actual best raw1000 checkpoint with optimizer/EMA preserved and LR5e-5. Selecting1000avoids knowingly starting from the degraded1200state; it is a new branch, not a reclassification of the failed trial. Fresh matched1000heldout7 evaluation is required before training to fix the baseline. Broader data and noise change together; any gain cannot identify their individual causal contributions.

## Minimality check

One optional scalar and one training-only history perturbation, default0preserving existing behavior; reuse current sampler, loader, checkpointing, evaluator and tokenizer. No scheduled sampling, model-generated training rollouts, ASR/RL reward, architecture change, fullYT transfer or automatic nativeADI label admission. Implement and prove this narrow change before GPU launch.

## Scope and invariants

- **INV1 — Data/labels/splits.** Source semantic release:59fa3a25a6cbc4c0d891, originally867train/79valid dictionary weak pairs. Inputs remain Mandarin parallel cells; native references and weak-label provenance stay separate. No native→Mandarin renaming and no automatic ASR-label admission.
- Before expansion subtract the16same-speaker reserved source IDs and every audio-hash, normalized Mandarin-target and native-target alias. They may have been reserved only by the284subset. Preserve all existing release diagnostic exclusions. Report actual remaining N, excluded IDs/reasons and manifest hashes; do not assume final count867.
- Keep speaker-heldout79 untouched and disjoint by source speaker IDs, audio hashes and normalized target aliases. Assert the seven development prompt targets and source recordings are absent from expanded training. Original seen8 remain permitted training probes.
- **INV2 — Checkpoint/model identity.** Preserve tokenizer4096, learned text embeddings, architecture, Mimi, latent statistics and optimizer semantics. Resume exact best1000 raw checkpoint with optimizer/global step and EMA into a separate run directory; no silent embedding reset or fresh optimizer. Parent verifies hash, resumed step1000 and first LR5e-5.
- **INV3 — Fixed recipe.** Keep LR5e-5 constant, batch16×accumulation4, LSD/FBM4, textdrop.2, voice absence as verified from actual batches, stats_ema_steps0, unchanged EOS/EMA/seed/checkpoint behavior. Change only data membership, explicitly selected history noise and bounded step/output settings. Verify actual loader output has matching transcript/audio offset and no voice frames.

## Run and budget

**INV5 — Runtime/budget.** Train global1000→2000 maximum; evaluate1200,1600 and2000 in this separate branch, preserving the parent1000baseline and distinguishing these new steps from the failed old1200run. One45-minute wall-clock allowance including checks and evaluation, no overlapping GPU task. At recorded$4.50/H100-hour the tranche is at most$3.375 compute; existing storage/tax and total$150ledger remain the root's responsibility. Deadline22:00Taipei and provider stop controls remain unchanged. Do not extend because training loss decreases.

**INV6 — Evaluation identity.** Primary generation settings remain CFG1, four decode steps, temp.3, seed17, no voice, original EOS and max-frame limit. Evaluate seen8 and existing speaker-heldout7 using identical settings at baseline and checkpoints. Run originalCFG2 final evaluation only as secondary compatibility evidence. Never treat fixed-T diagnostic output as ordinary generation success.

## Explicit acceptance and stop rules

1. Admission: root selected this fallback after the1200gate failure; INV1leakage checks and INV2resume checks pass; INV4CPU code proof passes; fresh matched parent1000development baseline is saved; INV5budget/time and INV6settings are verified.
2. Immediate stop: nonfinite values, new systematic caps/duration collapse, manifest/label leakage or identity mismatch. At new1200stop if seen mean is below parent1000 by more than.15, own-source retrieval falls below4/8, or at least2/15seen+development outputs cap or cannot decode. Any nonfinite sample stops immediately. Otherwise finish only the finite2000endpoint; these early-stop thresholds never substitute for the stricter final acceptance gate.
3. **Acceptance (INV1–INV6)** requires **new-text development evidence**: own-source-ASR retrieval≥3/7 and mean Mandarin-target similarity improvement≥.10 over matched parent1000baseline on the existing seven, no caps/duration collapse, and seen8mean within.05 of parent1000(.4814484, so minimum.4314484) with retrieval≥6/8. These predeclared operational thresholds are small-sample experimental gates, not significance, translation accuracy or a product-quality claim. Record both axes, not a combined undisclosed score.
4. Choose the checkpoint by these seen/development gates, then perform one final audit on the16same-speaker reserved prompts. They must not drive intermediate checkpoint selection or be added to training. Their earlier evaluation history means this is a reserved audit, not a pristine independent test.
5. At final selected checkpoint, if within INV5reserve, repeat existing four-row fixed-T probe4 with the same noise/seed to compare oracle/block8/fullself errors and content against800; this supports mechanism interpretation but never substitutes for INV6free generation. If the finite endpoint fails new-text acceptance, stop this expansion rather than extending because seen memorization improved. Passing admits only another explicitly bounded evidence-driven tranche; fullYT/nativeADI still need their own admission decision.

## Actual seven-source calibration, verified locally

`diagnostics/h100-source-asr-v1/asr-calibration.json` contains category `speaker_heldout`, exactly7rows. Each of the7 current `heldout-prompts.jsonl` rows joins **one-to-one** by source utterance ID, audio SHA256, Mandarin target and native reference. Raw/normalized source outputs are7/7distinct. All31predeclared source controls are included (8seen+16reserved+7heldout); the JSON also carries2punctuation diagnostics.

| Development input | Native reference | Frozen source ASR |
|---|---|---|
| 剁雞肉 | 鏨雞肉 | 斬雞肉 |
| 你說什麼？ | 你講啥？ | 喂 你說什麼 |
| 肥皂 | 油胰 | 友誼 |
| 有錢人 | 好額人 | 有錢人 |
| 你運氣真好耶！ | 你真好運呢！ | 你真幸運 |
| 下午 | 下晝 | 下午 下午 |
| 吃午餐 | 食晝 | 家道 |

Own-source retrieval compares each generated normalized ASR against these same7frozen source-ASR strings using the established normalization/SequenceMatcher scorer. Count a success only when its own reference is strictly better than every wrong reference; ties and empty outputs fail. Retain all pairwise scores and raw strings. Source-ASR errors such as友誼/家道 remain diagnostic references, never relabeled semantic truth. Separately score against immutable Mandarin targets; source mean target similarity is.517460, not a performance ceiling. These seven have already been repeatedly examined and are **development/validation**, not an untouched test or proof of unseen-speaker cloning. Expected low scores on excluded lexical targets make this a demanding generalization test; do not relax it after observing outputs.

## Selected source change and required CPU proof — INV4

Actual trained parity is numerically tight; English native generation is healthy. Probe4shows oraclecorrect and oraclewrong both preserve content, while self-fed errors grow sharply for some cases and block4resets help. This supports history fragility/history dominance, not sufficient independent text control.

Add `teacher_history_noise_std` (default0, finite/nonnegative), applying sigma=.05 normalized Gaussian noise only to valid teacher-forced predecessor audio inputs during training. Clean flow targets, BOS, text/voice, EOS labels, latent statistics and eval/generation remain untouched; sequence assembly already shifts history right. This sigma is a heuristic based on observed one-step MSE≈.0024, not an established optimum. Require a tiny CPU invariant test: sigma0equivalence; perturbed valid history only; clean targets byte-identical; BOS/causal shift intact; evalclean. No scheduled sampling, rollout reward, new framework or parameter sweep. The root has selected this fallback; the implementation must satisfy these proofs before GPU admission. Reject nonfinite/negative sigma, apply no noise when sigma0or model.eval(), and reject unsupported teacher/head-distillation combinations when sigma>0. Ensure noise does not mutate the original normalized target tensor in place. With padded terminal EOS position, perturb valid predecessor history only; preserve padded targets and masks. Save actual test output and reviewed diff as the implementation receipt.

## Alternatives ruled out for this next step

FullYT3000-step evaluation near.00355 with caps3/12 supplies no positive training-admission evidence. NativeADI transcripts describe pronunciation text rather than Mandarin semantic inputs, so untagged mixing changes the task. Resetting Mandarin embeddings discards measured learned sensitivity. Voice-conditioning bootstrap has no isolated speaker-condition diagnosis and cannot use target audio as a covert prompt. Frozen-head/RL additions do not follow a demonstrated head defect; official default CTC reward is English and unsuitable here.
