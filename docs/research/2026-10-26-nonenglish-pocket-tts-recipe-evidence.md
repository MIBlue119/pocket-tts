---
title: Evidence and staged recipes for adapting Pocket TTS beyond English
date: 2026-10-26
input_shape: concept
subject: Taiwanese Hokkien from Traditional Chinese semantic text, Taiwanese Mandarin, and Japanese
---

# Evidence and staged recipes for adapting Pocket TTS beyond English

## Bottom line

The delivery target is **ordinary Traditional-Chinese Mandarin semantic text → correct, natural
Taiwanese Hokkien audio**. Japanese and Taiwanese-Mandarin findings in this note are comparative
controls, not parallel model-delivery proposals. A private Hugging Face artifact, the USD 100 total
budget and the 22:00 deadline are operational constraints; none of the research evidence justifies
promising that a new 24-layer teacher can reach unmeasured production quality inside them.

The current step-1000 result does **not** show that Pocket TTS cannot learn the requested
Chinese-to-Hokkien task. It shows that the present experiment has not yet established that it can:
the 316.1M-parameter 24-layer FlowLM has a newly randomized 4,096-row text table, has processed only
about 0.35 nominal passes over 22,606 training rows, and is being optimized with settings far from
Kyutai's measured new-language recipe. Its teacher-forced validation loss improved, but its fixed
free-running outputs remained repetitive and unrelated to the input. Those observations can coexist;
the loss is not a generated-speech or text-adherence metric.

The highest-information next experiment is a **small, clean Hokkien memorization/control test**, not
another long weak-label-only run. It should first prove that the current initializer, optimizer,
tokenizer, text conditioning, codec, prompt policy, and free-running path can learn 32–64 existing
independently annotated/native-read pairs. This auxiliary control requires no new owner labeling and
does not by itself prove Mandarin-semantic input fidelity. Only after it succeeds should a 24-layer
teacher mix in weak labels. Distill to six layers only after the teacher passes free-running language
and semantic checks.

For provenance:

- Kyutai officially provides a generic new-language training recipe and official models for a limited
  set of European languages. These are language-specific checkpoints, not one demonstrated
  multilingual teacher. It does **not** provide an official Mandarin, Japanese, or Taiwanese
  Hokkien checkpoint as of this review.
- The current Japanese and Mandarin Pocket-TTS checkpoints are community artifacts. The Japanese
  author publishes a detailed, reproducible recipe; the Mandarin model card documents an effective
  frontend and measured outputs but does not publish enough training provenance to treat its recipe
  as reproduced evidence.
- Mandarin semantic text paired with Hokkien audio is a legitimate direct supervision format: Meta's
  published Hokkien work trained Mandarin-character text to Hokkien speech units from roughly 1,000
  hours of Mandarin-subtitled Hokkien drama. That is feasibility evidence for the task definition,
  not evidence that 17.9 hours of noisy pairs is sufficient for Pocket TTS.

This note is research only. It does not change or direct any running job.

## What is actually released and supported

| Item | What the source establishes | Provenance and boundary |
|---|---|---|
| Released Pocket TTS inference model | The stock English config is a 6-layer, `d_model=1024` FlowLM with a 4,000-token lookup table and a 24 kHz, 12.5 Hz Mimi latent path. | **Official Kyutai.** See [English config](../../pocket_tts/config/english.yaml) and [official model card](https://huggingface.co/kyutai/pocket-tts). It is the deployment student, not the recommended new-language training starting shape. |
| Current training pipeline | Kyutai trains a 24-layer teacher, then depth-distills it to a 6-layer student with CFG baked in. The repository says this works better than training a six-layer model from scratch. | **Official Kyutai.** [Training README](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md), [scratch config](../../training/configs/scratch.yaml), [depth-distillation config](../../training/configs/depth_distill.yaml). |
| New-language warm start | Replace the tokenizer, reset the whole text embedding table, load the English 24-layer teacher, and update the full FlowLM. Kyutai reports that on 976 hours of Czech, this reached 12% WER at 10k steps versus 46% from scratch, with both later around 10–12%. | **Official Kyutai measured result.** [finetune_language.yaml](https://github.com/kyutai-labs/pocket-tts/blob/main/training/configs/finetune_language.yaml). It is same-language transcript-to-speech adaptation, not cross-language semantic text-to-speech. |
| Official new-language scale | The guide calls 100 hours a practical minimum and 1,000+ hours desirable; its reference recipe is effective batch 64, `flow_batch_multiplier: 4`, LR `2e-4`, 1,000 warmup steps, and up to 250k updates. | **Official Kyutai guidance, not a universal law.** [Training README](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md) and [finetune config](../../training/configs/finetune_language.yaml). |
| Japanese Pocket TTS | A community author reports plain kanji/kana input, a new 4k tokenizer and reset text table, an English-24L warm start, 700k teacher updates at effective batch 64, then CFG/depth distillation. The language-model stage kept Mimi frozen; a later, separate decoder-only adaptation addressed Japanese reconstruction artifacts. | **Community, author-primary evidence.** [Model card](https://huggingface.co/takuma104/pocket-tts-ja), [source recipe](https://github.com/takuma104/pocket-tts-ja), and [B1 config](https://github.com/takuma104/pocket-tts-ja/blob/main/configs/train/B1_4k_mix8.yaml). It demonstrates architectural feasibility, not official Kyutai support. |
| Mandarin Pocket TTS | A community 6-layer model reports a 9,174-token vocabulary made by retaining the English BPE base and appending Han characters/full-width punctuation, plus explicit number/date/text normalization. | **Community model-card claim.** [PocketTTS-ZH-6L](https://huggingface.co/hustfyb/PocketTTS-ZH-6L). Its public files contain inference artifacts and a model card, but no sufficiently complete training recipe or data lineage; it does not specifically establish Traditional-Chinese quality. |
| Chinese semantic text to Hokkien speech | Mandarin subtitles can supervise Hokkien acoustic-unit generation even without native Hokkien orthography. | **Published non-Pocket research.** Meta's [ACL 2023 paper](https://aclanthology.org/2023.findings-acl.307/) used about 1,000 hours for its Mandarin-text-to-Hokkien-unit component. This supports the interface, not the current data scale or Pocket-TTS convergence. |

The 24-layer English teacher used by this project is defined directly in
[english_2026-04_24l.yaml](../../pocket_tts/config/english_2026-04_24l.yaml). The active trainer log
reports **316.1M trainable parameters** for the FlowLM plus LSD objective. By contrast, the released
six-layer inference family is the CPU-oriented deployment target. Treating “Pocket TTS is a 100M
model” as the training recipe conflates these two stages.

### Current official checkpoints versus this run's initializer

The exact active initializer is `english_2026-04_24l`, not the old `b6369a24` release. The latter is
the signature comment in the older 6-layer
[english_2026-01.yaml](../../pocket_tts/config/english_2026-01.yaml). This matters when reproducing or
comparing results.

Kyutai has since published additional 24-layer configs with the same visible FlowLM dimensions
(24 layers, width 1,024, 4,000 text rows) and the same visible Mimi dimensions. In particular,
[english_2026-09_24l.yaml](../../pocket_tts/config/english_2026-09_24l.yaml) says it is the April
teacher after a robustness fine-tune with the sampler head frozen. French, German, Portuguese,
Italian, Spanish, and Dutch also have separate `*_24l.yaml` configs in the checked-out source.
This establishes **shape compatibility**, not superior transfer to CJK or Hokkien and not exact
Mimi-weight identity.

There is also a source-snapshot discrepancy worth preserving instead of smoothing over: the live
[official Hugging Face card](https://huggingface.co/kyutai/pocket-tts) labels the repository “6
languages” and lists English, French, German, Portuguese, Italian and Spanish, while current official
[GitHub README](https://github.com/kyutai-labs/pocket-tts#main-takeaways) additionally lists Dutch.
Neither source describes a single multilingual checkpoint. Each language config points at its own
model/tokenizer path.

**Proposal, not evidence:** do not switch the current long run's base. After the clean control below
passes or identifies an initializer-sensitive failure, compare April English against September
English on the *same* control data, tokenizer, optimizer, seeds, prompts and frozen Mimi. Preserve
the already-verified Mimi or prove tensor identity before mixing components. Resetting the text
table means the newer checkpoint can contribute only its non-text-table state; no primary source
shows that a European-language 24-layer checkpoint transfers better to Chinese/Japanese text or
Hokkien acoustics.

## Current experiment: facts, not interpretation

The current evidence is in the ignored run artifacts under
`data/taigi-full-corpus/local-longer-20261026/` and the durable project record in
[batch weak-label results](../plans/2026-10-25-batch-weak-label-results.md).

| Property | Observed value |
|---|---:|
| Training / validation rows | 22,606 / 148 |
| Training audio | 17.8754 hours |
| Median / mean row duration | 2.14 / 2.85 seconds |
| Model | English 24-layer teacher warm start; 316.1M trainable FlowLM/objective parameters |
| Mimi | Loaded from the English config, evaluation mode, all gradients disabled |
| Tokenizer | Train-only BPE, 4,096 pieces, byte fallback, complete conversion parity; text embedding fully reset |
| Tokenizer use on train set | 3,881 of 4,096 IDs observed; mean 9.66 tokens/row; 165 rows use at least one byte-fallback token; zero unknown-token rows |
| Optimizer | AdamW, constant LR `3e-5`, 100 warmup updates |
| Batch | microbatch 1, accumulation 8, effective batch 8 |
| Flow samples | `flow_batch_multiplier: 1` |
| Conditioning | full sentence, zero voice-prompt frames, text/voice dropout both zero |
| Validation loss | 0.52786 at step 0; 0.50007 at 200; best logged 0.48715 at 950; 0.49025 at 1,000 |
| Step-1,000 generation | 12/12 finite and no frame-limit hits, but mean same-Breeze character-sequence similarity 0.0301; only 2/12 were nonzero and outputs were repetitive/off-target |

At effective batch 8, 1,000 updates expose the optimizer to about 8,000 sampled rows. Dividing by
22,606 rows gives **0.354 nominal row passes**, or about 6.33 sampled audio hours at the observed mean
duration. This is not a full pass over the data. Bucketing, replacement, and variable durations mean
that “nominal pass” is only an exposure accounting aid, not proof that every row was seen.

The tokenizer coverage numbers argue against “the tokenizer cannot represent Chinese” as the sole
explanation. They do not show that the random embeddings have learned useful semantics or that weak
labels are correct.

### Why lower teacher-forced loss can coexist with bad generated speech

The training path prepends text and then feeds a BOS frame plus **ground-truth previous Mimi
latents** into the causal transformer. See
[conditioner.py](../../training/modules/conditioner.py) and the `backbone_z` path in
[model.py](../../training/modules/model.py). The loss then asks the flow head to recover each true
next latent and separately trains EOS. Validation calls the same model on encoded reference audio;
see [train.py](../../training/train.py).

Free-running generation is harder: it samples a latent, feeds that sampled history into the next
step, and must decide when to emit EOS. A model can therefore lower held-out loss by improving local
acoustic continuation from correct histories while still:

- underusing or ignoring the text prefix;
- drifting after its first incorrect generated frame;
- learning common short-audio patterns or speaker/acoustic priors;
- repeating frequent weak-label/audio associations;
- making poor EOS decisions under its own history.

The present setup differs from the official CFG-training recipe because text dropout is zero, but
that is not established as the cause of text ignoring: dropout by itself does not force causal
dependence on text. All examples have zero voice prompt, while the audio comes from many speakers,
leaving speaker and style underdetermined. The target rows are short and the Chinese labels are
Breeze-generated semantic weak labels, not verified verbatim transcripts.
Teacher-forced flow/EOS loss tests none of “is this Taiwanese?”, “does it mean the requested
sentence?”, or “is the pronunciation authentic?”. The evaluator correctly records Breeze similarity
as same-model consistency, not independent quality evidence.

Kyutai's guide gives a useful but conditional diagnosis: if generated speech is already acoustically
good but does not follow the transcript, inaccurate transcripts are the likely cause. The current
step-1,000 samples are repetitive and off-target, and the text table is newly random with little
exposure, so that rule does **not** isolate weak labels yet. The clean memorization control is what
separates label/speaker ambiguity from an optimization or conditioning failure.

## Codec and “codebook” compatibility

Pocket TTS does **not** train against a discrete Mimi codebook in this repository. `MimiModel`
returns unquantized continuous 32-dimensional latents, and `DummyQuantizer` explicitly removes the
quantization logic. See [mimi.py](../../pocket_tts/models/mimi.py) and
[dummy_quantizer.py](../../pocket_tts/modules/dummy_quantizer.py). The CALM paper likewise describes
a transformer that predicts continuous audio-VAE frames rather than discrete codec tokens:
[Continuous Audio Language Models](https://arxiv.org/abs/2509.06926).

So the compatibility question is latent geometry, not token/codebook identity:

- The current builder loads the Mimi weights named by the model config, calls `eval()`, and disables
  all Mimi gradients; see [builders.py](../../training/modules/builders.py).
- The FlowLM, latent normalization, speaker-prompt projection, and decoder must agree on the same
  encoder/decoder latent space. Replacing the encoder after FlowLM training is incompatible unless
  the language model is retrained or compatibility is demonstrated.
- A decoder-only adaptation can preserve encoder latents in principle. The Japanese community model
  did exactly that after measuring Japanese codec degradation, but that is community precedent, not
  a Kyutai-supported training recipe.

**Evidence gate before any language run:** round-trip untouched held-out audio through the frozen
encoder and decoder for Taiwanese Hokkien, Taiwanese Mandarin, and Japanese. Compare original versus
reconstruction using language-appropriate ASR/reading tests, speaker similarity, and targeted
phonological examples. For Hokkien this must include checked syllables, nasalization, tones and tone
sandhi contexts; for Mandarin, tones and retroflex/neutral-tone contrasts; for Japanese, mora timing,
long vowels, geminates, pitch-sensitive minimal pairs, and difficult kanji readings. If information
is already lost by reconstruction, more FlowLM steps cannot restore it reliably.

## Text representation and tokenizer choices

The text tokenizer controls efficiency and the identity of every embedding row; it does not supply
pronunciation knowledge by itself. The current code requires tokenizer vocabulary size to exactly
match `lookup_table.n_bins`, and the lookup table has one additional internal padding row. Kyutai's
trainer defaults to a 4,000-piece unigram model, identity normalization, digit splitting, byte
fallback, and no whitespace squashing; see
[train_tokenizer.py](../../training/scripts/train_tokenizer.py).

| Choice | What it preserves | Cost / risk | Evidence status |
|---|---|---|---|
| Keep the English tokenizer and table | All pretrained English text embeddings and stable IDs. Useful for an English-only domain shift. | Han/kanji/kana may fall back to byte sequences, producing long inputs whose pretrained rows have byte, not Chinese/Japanese semantic meaning. | Not Kyutai's recommended new-language route. Byte fallback makes strings representable, not understood. |
| Replace tokenizer and reset the full text table | Efficient target-language pieces; simplest consistency contract. | Discards all learned text-row meanings. The backbone and new table must co-adapt; a low LR chosen for same-language fine-tuning may be too conservative. | **Only official measured new-language route.** Czech used this with full-model LR `2e-4`. The Japanese community model also used a fresh 4k table. |
| Extend a stable English vocabulary and copy matching rows | Can retain English/code-switch tokens and initialize only new characters. | Requires stable old token IDs, deliberate initialization of appended rows, a larger table, and partial state loading. Current `reset_text_embedding` drops the whole table; the stock builder does not implement row-wise preservation. | Community Mandarin model claims this frontend, but does not publish enough training detail to establish the adaptation recipe. Treat as an ablation requiring code and verification. |

For the requested models:

### Mandarin semantic text to Taiwanese Hokkien audio

Train the tokenizer on the **actual production input representation**: normalized Traditional
Chinese Mandarin semantic sentences. Do not train it primarily on Tai-lo if users will type ordinary
Traditional Chinese. Native Hokkien orthography is not required for the direct task, but it is highly
valuable as an auxiliary annotation and independent evaluation reference. Mandarin text does not
uniquely specify Hokkien lexical choice, reading, tones, sandhi, or regional variety; consistent
target recordings and targeted coverage matter more than simply increasing Chinese vocabulary size.

Meta's direct Mandarin-to-Hokkien-unit result supports sentence-level semantic pairing. The current
sentence loader is therefore the right shape for non-monotonic translation pairs; fabricated
word-level alignments would be misleading. However, Breeze itself warns that its output is Mandarin
characters rather than native Hokkien orthography, that mapping is non-one-to-one, and its benchmark
CER is suitable mainly for relative comparison. See the
[Breeze-ASR-26 model card](https://huggingface.co/MediaTek-Research/Breeze-ASR-26).

### Traditional-Chinese Taiwanese Mandarin TTS

Use verbatim Traditional-Chinese transcripts paired with Mandarin audio. Freeze one normalization
contract before tokenizer training and inference: Traditional/Simplified policy, full-/half-width
punctuation, Arabic numbers, dates, currency, Latin abbreviations, whitespace, and sentence-final
punctuation. Include explicit held-out challenges for polyphonic characters, names, numbers,
classifier words, neutral tone, and code-switching. The community Mandarin model's strongest
transferable lesson is that text normalization is part of the model interface; its reported quality
is not evidence for untested Traditional-Chinese inputs.

### Japanese TTS

A fresh 4k tokenizer with guaranteed kana and common-kanji coverage is the best evidenced starting
point. The community Japanese author found byte-fallback kanji difficult and forced Joyo kanji,
kana, and ASCII into the tokenizer character set. Plain written Japanese can work without G2P, but
the text remains pronunciation-ambiguous. Coverage must deliberately include on/kun readings,
jukuji/rare readings, counters, names, loanwords, long vowels and geminates. JSUT was explicitly
designed around daily-use-kanji readings ([JSUT paper](https://arxiv.org/abs/1711.00354)); JVS adds
100-speaker variation ([JVS paper](https://arxiv.org/abs/1908.06248)). Their licenses and permitted
uses must be checked before any training plan treats them as available data.

Do not initially mix Mandarin-TTS and Chinese-to-Hokkien rows under the same untagged Chinese input:
the same sentence would have two target languages with no conditioning signal to choose between
them. Start with separate teachers. A later multilingual experiment needs an explicit, trained
language/task condition and a tokenizer whose IDs remain fixed across all splits; neither behavior
is established by the current recipe.

## Proposed staged recipe

Everything in this section is a **proposal derived from the evidence above**, not a measured quality
promise.

### Stage 0 — prove data and codec contracts

1. Reuse a small independently annotated/native-read evaluation set, separated by source and
   speaker. Do not make new owner listening or annotation a prerequisite for this diagnostic.
2. Run the frozen-Mimi reconstruction control described above.
3. Audit text/audio semantic agreement, language purity, clipping, overlap, music, speaker identity,
   duplicates, numbers, names, negation, and code-switching. Weak-label agreement from two passes of
   the same ASR is not independent evidence.
4. Decide the product's speaker condition. For voice cloning, pair every target with a non-overlapping
   same-speaker prompt containing different words. For a no-prompt fixed voice, begin with one
   consistent speaker. Mixed speakers with no prompt are an avoidable one-to-many target.

For Hokkien, a compact high-quality control can use material with both Han characters and Tai-lo,
such as SuíSiann, subject to its use and text-modification conditions; the official page reports a
single speaker and about 4.7 hours in v0.2.1:
[SuíSiann dataset](https://suisiann-dataset.ithuan.tw/). The purpose here is control and evaluation,
not a claim that this corpus alone trains a general model.

### Stage 1 — tokenizer and text-table control

Freeze the tokenizer before any resumable run. Record its hash, normalization rules, character
coverage, byte-fallback rows, mean/p95 token length, and every special/padding ID. A tokenizer change
means a fresh model run.

Use two bounded variants only if resources permit:

- **A, official baseline:** fresh 4k target-language tokenizer, full text-table reset, full FlowLM
  unfrozen from update 0. This is the only Kyutai-measured language-transfer strategy.
- **B, diagnostic staged adaptation:** briefly train only the new text table, then unfreeze the full
  FlowLM, or use a larger LR for the table than the backbone. This can test whether a random table
  is destabilizing the pretrained backbone, but current configuration does not expose freezing or a
  separate table LR; it needs an explicit implementation and is not an official recipe. Embedding-only
  training cannot by itself learn Hokkien acoustic realization, so it must remain short.

Do not infer that the current full reset failed. It was combined with LR `3e-5`, effective batch 8,
`flow_batch_multiplier: 1`, zero prompt, weak labels, and less than one nominal data pass. Those
variables have not been separated.

### Stage 2 — small overfitting / text-control test

Reuse 32–64 independently annotated or native-read 2–8 second examples already available from a
licensed source, preferably one speaker and with naturally occurring text contrasts. This first
tests whether the current pipeline can condition on text and produce Hokkien. If its text is native
Hokkien orthography rather than ordinary Mandarin semantic text, it is an **auxiliary pipeline
control only**, not proof of the product task. Test Mandarin-semantic → Hokkien fidelity only with
pre-existing gold semantic pairs when they are actually obtainable without asking the owner to
listen to and relabel a new set. Never relabel Breeze-generated pairs as “verified.”

Generate from the initialized checkpoint and then at frequent fixed checkpoints using identical
seeds, temperature, CFG and prompt audio. Require the model to follow **training texts first**. A
useful control gate is:

- recognizable target language rather than Mandarin leakage or acoustic babble;
- correct coarse meaning on memorized examples when independent references permit judging it;
- existing number/negation minimal pairs, if present, produce the intended different speech;
- no systematic looping, prompt copying, silence, or premature/runaway EOS;
- fixed-Mimi reconstructions already pass the relevant phonology checks.

For exposure accounting, 64 rows with effective batch 8 yield eight optimizer updates per nominal
pass. Checking at roughly 400, 800 and 1,600 updates corresponds to about 50, 100 and 200 nominal
passes. These are diagnostic checkpoints, not expected convergence guarantees. If the model cannot
follow the **training** texts by about 1–2k updates, do not scale the same recipe; isolate tokenizer
IDs/normalization, text-gradient use, prompt policy, codec reconstruction, and optimizer choice.

The single highest-priority next experiment is therefore:

1. Select up to 64 pre-existing independently annotated/native-read pairs, preferably from one
   consistent speaker (or with an existing independent same-speaker prompt). This requires no new
   owner annotation. Record whether the text is native Hokkien or true Mandarin-semantic input.
2. Keep the current `english_2026-04_24l` initializer, current 4,096-piece tokenizer, reset text
   table and current frozen Mimi. This is a diagnosis of the active pipeline, not a new-base test.
3. First reproduce the current optimization block—LR `3e-5`, 100-step warmup, effective batch 8 and
   `flow_batch_multiplier: 1`—for at most 800–1,600 updates, generating the fixed control set every
   100 updates. With 64 rows, 800 updates expose about 100 nominal passes and 1,600 about 200. Stop
   as soon as the causal answer is clear; this is a memorization test, not a quality benchmark.
4. Pass the auxiliary control only if free-running audio follows the memorized text, speaks Hokkien,
   terminates reliably and does not loop. Claim semantic-task success only if the inputs were genuine
   gold Mandarin-semantic pairs. Teacher-forced loss is recorded but cannot pass either gate.

If this exact-recipe control passes, the current long run's strongest unresolved causes are weak-label
quality, speaker/prompt ambiguity and insufficient exposure. If it fails, run one matched optimizer
ablation on the same 64 pairs using Kyutai's new-language block—LR `2e-4`, 1,000-step warmup,
effective batch 64, FBM 4 and text/voice dropout 0.2—then inspect text gradients and tokenizer IDs
before trying a new initializer. Equalize sampled rows or wall-clock cost when comparing the two;
equal update counts would give the official block eight times as many rows and 32 times as many
per-frame flow targets. This ordering preserves causal value and the stated budget. It does not
claim that either control yields useful held-out speech.

### Stage 3 — 24-layer teacher on real data

Start from the released English 24-layer teacher, keep the Stage-0-passing Mimi fixed, use the
successful Stage-2 tokenizer and prompt policy, and update the full FlowLM. Begin with independently
annotated data that is already available, then add weak data through an explicit sampling ratio
rather than allowing 17.9 hours of weak YouTube segments to define the entire target distribution.

The official baseline to compare against is LR `2e-4`, 1,000 warmup steps, effective batch 64,
`flow_batch_multiplier: 4`, text/voice dropout 0.2, and a constant schedule. It is not automatically
optimal for this much smaller, noisier, cross-language corpus. Use the Stage-2 matched control before
selecting any setting, and select by free-running gates rather than teacher-forced loss alone.

Evaluate at 0, 500, 1k, 2k, 5k, 10k and then wider intervals. Continue only when independent
free-running evidence improves. For the Hokkien task, report meaning accuracy and “actually Hokkien”
separately. A Mandarin-character ASR can score a model that simply speaks Mandarin, so add an
independent Hokkien-versus-Mandarin check and native-orthography or native-speaker evaluation.

### Stage 4 — distill only a passing teacher

Once the 24-layer teacher passes fixed held-out generation, use the same tokenizer, Mimi latent
space, prompts, and text normalization to distill a six-layer student. The official reference is
200k depth-distillation updates with CFG 2 baked in. Distillation compresses a behavior already
present in the teacher; it does not repair mistranslation, Mandarin leakage, bad labels, or codec
loss.

### Stage 5 — decoder adaptation only if reconstruction requires it

If Stage 0 shows language-specific reconstruction loss, investigate decoder-only adaptation in a
separate experiment while leaving the encoder unchanged, then re-run every reconstruction and
FlowLM compatibility check. The Japanese community result is a concrete precedent, but it used
90k decoder updates and specialized adversarial/spectral losses not present in Kyutai's released
training path. This is a new workstream, not a flag to enable casually.

## Step counts: evidence and caveats

The present dataset's nominal exposure is:

| Cumulative updates | Rows sampled at effective batch 8 | Nominal corpus passes | Approx. sampled audio hours at current mean duration |
|---:|---:|---:|---:|
| 1,000 | 8,000 | 0.35 | 6.3 |
| 3,000 | 24,000 | 1.06 | 19.0 |
| 6,000 | 48,000 | 2.12 | 38.0 |
| 10,000 | 80,000 | 3.54 | 63.3 |
| 15,000 | 120,000 | 5.31 | 94.9 |
| 50,000 | 400,000 | 17.69 | 316.3 |

These figures do not predict quality. They ignore duration weighting, repeated sampling, label
noise, and the fact that an optimizer update at effective batch 8 is not equivalent to Kyutai's
batch-64 update. With `flow_batch_multiplier: 1`, the current run also draws one quarter as many
flow targets per backbone position as the reference setting. At the same update count, the official
recipe therefore sees eight times as many rows and 32 times as many per-frame flow-noise draws.

The external anchors are deliberately broad:

- Kyutai's 2,000-hour English reference says loss is already falling by about 2k updates,
  intelligibility begins improving around 15k, reaches about 1% WER by 50k, and acoustic quality
  continues through 150–300k.
- Kyutai's 976-hour Czech transfer reached 12% WER at 10k updates and used a 250k cap.
- The community Japanese teacher used 700k updates with more than 1,500 hours of real Japanese plus
  augmented and synthetic data, then separate distillation/decoder stages.

None of those supports a claim that Hokkien will become usable at 3k, 6k, 15k, or 50k on 17.9 hours
of weak data. The only defensible schedule is gate-based: first memorize clean controls, then show
held-out free-running progress, then increase data or updates.

## Evaluation matrix

| Signal | What it answers | What it cannot establish |
|---|---|---|
| Teacher-forced flow/EOS validation loss | Does the model predict held-out true latent frames from true history more easily? | Free-running text adherence, target language, pronunciation, naturalness. |
| Training-set free generation | Can the pipeline at least memorize and condition on its text? | Generalization. |
| Held-out fixed generation | Does behavior transfer to unseen text/speakers/sources? | Authentic phonology without an appropriate evaluator. |
| Frozen-Mimi reconstruction | Can the acoustic representation preserve the target language before LM training? | Whether FlowLM learns text conditioning. |
| Target-language ASR / kana CER / native orthography | Intelligibility under that recognizer and representation. | Naturalness, speaker similarity, or unbiased quality if the same model made labels. |
| Language ID plus contrastive number/name/negation set | Mandarin leakage, gross wrong-language output, semantic contrasts. | Full translation quality or dialect authenticity. |
| Native-speaker blinded listening | Pronunciation, fluency, tone/sandhi, naturalness. | Broad coverage unless the test set is designed and reported. |

For Hokkien, keep two scores separate: **semantic fidelity to the Mandarin input** and **Hokkien
linguistic authenticity**. Breeze's own model card states that Mandarin-character output is a
non-one-to-one proxy and reports average benchmark CER around 30%; it must not be the sole labeler
and sole judge of the generated model.

## Decision summary

1. Preserve the 24-layer-teacher → six-layer-student architecture distinction.
2. Keep full text-table reset as the first official baseline; do not blame it before an adequately
   exposed control. Treat stable vocabulary extension as a separate community-inspired ablation.
3. Run a codec reconstruction gate for every target language. There is no discrete codebook to
   swap; latent encoder/decoder identity is the contract.
4. Prove text control on 32–64 clean examples before another long weak-label run.
5. Spend the current delivery budget only on Mandarin-text-to-Hokkien speech. If Mandarin or Japanese
   are pursued later, do not mix their untagged targets into this teacher.
6. Resolve speaker conditioning: independent same-speaker prompts for voice cloning, or consistent
   single-speaker targets for no-prompt training.
7. Compare optimizer settings on bounded controls. The current `3e-5`/batch-8/FBM-1 recipe is not
   the official new-language baseline, while the official `2e-4` recipe is not proven safe for this
   noisy 17.9-hour corpus.
8. Select checkpoints by fixed free-running outputs and independent language/semantic evaluation,
   never by teacher-forced loss alone.
9. Do not distill or adapt Mimi until the preceding stage has a specific, measured reason.

## Pstack-guided explorer record

The research pass followed the bounded explorer structure in
`/tmp/pstack-taiwanese-20261025/skills/how/references/explorer-prompt.md` from pstack commit
`18b6c36e6e201b110acc62472de987416ac4e23b`. The Pocket-TTS worktree examined was commit
`64e7ecd24d7d17790e8221d5bdca6c661c0e6067`. This records the snapshot; it does not imply pstack
authored or validated the technical claims.

### Components found

- Text: tokenizer JSON/SentencePiece conversion, exact-size lookup embedding, BOS/text assembly.
- Audio: frozen continuous-latent Mimi encoder/decoder and dummy quantizer.
- Teacher: 24-layer causal FlowLM, teacher-forced true previous latent, LSD flow and EOS losses.
- Student: separate six-layer depth/CFG distillation stage.
- Data: sentence manifests, optional independent voice prompt, raw-audio or precomputed-latent path.

### Flow

`manifest text/audio` → `tokenizer + text table` and `frozen Mimi encoder` → `BOS + prompt + text +
shifted true audio latents` → `24-layer FlowLM/LSD/EOS loss` → free-running sampled latents →
`frozen Mimi decoder`; after a passing teacher, depth-distill to six layers.

### Files read

- `training/README.md`; `training/configs/{finetune_language,finetune,scratch,depth_distill}.yaml`
- `training/{train.py,train_utils.py}`; `training/modules/{builders,conditioner,model}.py`
- `training/dataloader/{loader,encode}.py`; `training/scripts/train_tokenizer.py`
- `pocket_tts/models/mimi.py`; `pocket_tts/modules/{dummy_quantizer,text_conditioner}.py`
- `pocket_tts/config/{english,english_2026-01,english_2026-04_24l,english_2026-09_24l}.yaml`
- Active step-1,000 `training.yaml`, trainer log, fixed-generation report, train/valid manifests,
  tokenizer JSON and durable batch weak-label results record.

### Boundaries

- No code, configuration, dataset, checkpoint or running job was changed; only this report was added.
- Japanese and Mandarin were researched only as transfer/front-end comparisons for the Hokkien goal.

### Non-obvious things

- The released six-layer model and the trainable 24-layer teacher are different lifecycle stages.
- “Codebook compatibility” is a misleading frame here because training uses continuous Mimi latents.
- Current official language support is a collection of language-specific configs/checkpoints, not a
  verified multilingual teacher.
- The same falling loss can reward true-audio continuation while free-running text control remains
  unusable.

### Open questions

- Exact Mimi tensor equality across official checkpoint files was not established from YAML shapes.
- No primary source establishes a best initializer, tokenizer size, data threshold or expected step
  count for these three target tasks.
- Native-speaker scores, independent Hokkien language ID, frozen-Mimi reconstruction scores and
  gold-control memorization results do not yet exist. They are experiment gates, not current facts.

## Primary sources

- Kyutai: [Pocket TTS training README](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md), [new-language config](https://github.com/kyutai-labs/pocket-tts/blob/main/training/configs/finetune_language.yaml), [tokenizer trainer](https://github.com/kyutai-labs/pocket-tts/blob/main/training/scripts/train_tokenizer.py), [official model card](https://huggingface.co/kyutai/pocket-tts).
- Architecture paper: [Continuous Audio Language Models](https://arxiv.org/abs/2509.06926).
- Hokkien direct-supervision precedent: [Speech-to-Speech Translation for a Real-world Unwritten Language](https://aclanthology.org/2023.findings-acl.307/).
- Current weak-label evaluator: [MediaTek Research Breeze-ASR-26](https://huggingface.co/MediaTek-Research/Breeze-ASR-26).
- Japanese community implementation: [author model card](https://huggingface.co/takuma104/pocket-tts-ja) and [source/recipe](https://github.com/takuma104/pocket-tts-ja).
- Mandarin community implementation: [author model card](https://huggingface.co/hustfyb/PocketTTS-ZH-6L).
- Language-control corpora: [SuíSiann](https://suisiann-dataset.ithuan.tw/), [JSUT paper](https://arxiv.org/abs/1711.00354), [JVS paper](https://arxiv.org/abs/1908.06248), and the official [TAT corpus listing](https://www.aclclp.org.tw/use_mat.php). Dataset listing is not permission to use it; each license/application condition still applies.
