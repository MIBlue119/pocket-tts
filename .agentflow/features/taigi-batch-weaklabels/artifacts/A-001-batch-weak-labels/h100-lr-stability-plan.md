# Bounded lower-learning-rate stability trial

The 800-step semantic control failed its predeclared generation gate. This is a separate diagnostic continuation, authorized by the owner's request to optimize training within US$150 total and the existing22:00 cutoff. It does not promote a model or admit full-corpus training.

Actual evidence: correct text improves oracle-history endpoint MSE on8/8 seen rows; trained full/streaming paths agree within5e-6; untouched English generation works. Oracle reconstruction preserves source-ASR content, including with wrong text. Self-fed fixed-length generation loses content for手剎車/一串香蕉, while resetting complete history helps. This supports history-dependent rollout instability, not independently verified text semantics.

## Invariants

- INV-1: Resume exact800 raw model, optimizer and EMA with identical284-train/79-dev manifests and tokenizer. Parent files remain unchanged. Any identity mismatch prevents launch.
- INV-2: Change only optimizer LR2e-4→5e-5, run directory and maximum step. Preserve batch16×accumulation4, FBM4, text dropout.2, constant schedule, warmup100 and frozen latent statistics. Actual resumed LR must be5e-5.
- INV-3: Two fresh stages end at1000 and at most1200. Entire trial, including matched diagnostic evaluation, has a20-minute wall limit measured from the recorded launch. Original22:00 shutdown andUS$150 total remain binding.
- INV-4: Primary generation isCFG1/four-step, seed17, temperature.3, zero voice, same EOS and100-frame short-prompt bound. Preserve originalCFG2 gate as failed; no retrospective threshold changes.

## Acceptance criteria

- At1000, stop if both mean Mandarin-target string similarity and own-source-ASR retrieval decline versus matched800. Baseline target mean=.379365; baseline retrieval is computed from the frozen eight source-ASR references with strict own>all-wrong comparison.
- At1200, evidence for another bounded experiment requires target mean improvement≥.07 and retrieval≥6/8, no new health/duration-collapse failure, and no degradation on the existing speaker-heldout7 under identicalCFG1/four-step settings. Run matched800 heldout evaluation before comparing; report same-speaker reserved16 separately when available.
- Run originalCFG2/one-step at the final candidate as secondary evidence. These are noisy same-ASR diagnostics, not CER, pronunciation, or product-quality claims.
- If failed, stop further extensions on this284-row corpus. Decide another recipe/data experiment from the negative result. No history-corruption or scheduled-sampling code is included.

## Minimality check

One lower-LR continuation reuses existing trainer, checkpoints, frozen data and evaluator. The simpler alternative—blindly continuing the previous LR—already lacks a stable text-control result. New history-noise training code would add an unvalidated mechanism, so it is excluded. This trial has no matched continued-2e-4 sibling; any improvement cannot be attributed solely to LR rather than additional updates.
