---
title: Taiwanese Hokkien training diagnostics through semantic step 1000
date: 2026-10-26
status: measured; step 1200 pending
---

# Taiwanese Hokkien training diagnostics through semantic step 1000

## Decision snapshot

No completed checkpoint is a usable ordinary-Traditional-Mandarin-to-Taiwanese-Hokkien model.
The H100 semantic control learned some of its eight assessed training phrases, but the original
step-800 gate failed and matched held-out evidence remained absent. A lower-learning-rate
continuation improved the same-Breeze seen score at step 1000 from `0.379365` to `0.481448`, with
strict own-source ASR retrieval rising from 5/8 to 7/8. This is better memorization evidence, not a quality
or generalization result.

The completed probes rule out a large full-sequence/streaming wiring discrepancy at step 800 and
show that the model uses text under ground-truth audio history. They also show that ground-truth
history can dominate the requested text and that errors grow sharply when the model consumes its
own predictions. This supports an error-accumulation or exposure-gap hypothesis, but it does not
establish a single root cause.

The authorized budget remains **US$150 total**. The next hard decision cutoff is **22:00
Asia/Taipei on 2026-10-26**. The active owner controls the step-1200 run and any later pipeline
action. Step 1200 was still pending when this note was written and has no result here.

## Experiments and measurement boundary

The H100 control started from the released English 24-layer teacher, replaced its text table with
the existing 4,096-token Mandarin tokenizer, and reset that table. Its single-speaker training
subset contains 284 clips, 258 distinct Mandarin targets, and 603.504 seconds of audio; the frozen
validation manifest contains 79 clips. The recipe used batch 16, gradient accumulation 4, flow
batch multiplier 4, text dropout 0.2, LR `2e-4`, and zero voice-prompt frames during evaluation.

All generation comparisons below use raw checkpoints, seed 17, temperature 0.3, and no voice
prompt. The original gate used CFG 2 and one LSD decode step. The matched diagnostic setting uses
CFG 1 and four decode steps. Breeze-ASR-26 produced both much of the earlier weak supervision and
the diagnostic transcriptions. The reported SequenceMatcher values are therefore same-recognizer
consistency measurements, not CER, gold Taiwanese transcription, pronunciation scores, or human
quality judgments.

## Quantitative outcomes

| Checkpoint and setting | Seen mean | Own-source ASR retrieval | Other completed evidence |
|---|---:|---:|---|
| Step 0, CFG 2 / one step | 0.000000 | 0/8 | Two outputs hit the frame cap and three were too short for Mimi decoding. |
| Step 700, CFG 2 / one step | 0.365476 | 4/8 | Best original-gate seen score before the endpoint. |
| Step 800, CFG 2 / one step | 0.242560 | 3/8 | Original gate failed; held-out was 0.071429 (2/7 nonzero), reserved 0.012500 (1/16), general panel 0.020833 (1/12). |
| Step 800, CFG 1 / one step | 0.273810 | 4/8 | Inference change alone did not rescue the failed phrases. |
| Step 800, CFG 1 / four steps | **0.379365** | **5/8** | Matched held-out and reserved panels were both 0.000000. |
| Step 1000, CFG 1 / four steps, LR `5e-5` | **0.481448** | **7/8** | LR-only continuation from step 800; no completed matched step-1000 held-out or reserved result is recorded. |

Own-source retrieval requires the output to score strictly above every wrong source transcript;
ties fail. It is separate from the simpler nonzero-target count. The two happened to agree at
CFG-1/four-step step 800 (5/8) and step 1000 (7/8), but not at CFG-1/one-step step 800, where
own-source retrieval was 4/8 and the nonzero-target count was 3/8.

The step-1000 change restored the model, optimizer, and EMA from step 800 and changed only the run
endpoint, run directory, and LR from `2e-4` to `5e-5`. It did not change the data, tokenizer,
architecture, Mimi, batch, dropout, sampler, or latent statistics. The seen improvement is real
under this fixed weak diagnostic, but it is not enough to admit the checkpoint for use.

The original CFG-2 evaluation curves and probe-presence metadata are available in the verified
[Weights & Biases run `o4gwsy86`](https://forge.coreweave.com/wandb/weirenlan/pocket-tts-taigi/runs/o4gwsy86).
No audio or model checkpoint was uploaded with that run.

## What probes 1–4 establish

### Text matters under oracle history

Probe 1 used the exact eight seen rows with identical source history and noise across correct-text,
nearest-length wrong-text, and null-text arms. For one-step endpoint MSE, the median
wrong/correct ratio was 3.22 over the first four frames and 6.04 over all frames; the median
null/correct ratio was 1.86 and 2.30. Correct-text four-step all-frame endpoint MSE improved over
one-step for all 8 rows, with mean MSE falling from 0.002432 to 0.002018. The checkpoint therefore
does not wholly ignore text on these seen examples. Oracle-history success still does not show that
the model can preserve content after it starts feeding back its own latents.

### Full and streaming backbones agree

Probe 2 compared full-sequence and same-history streaming computation on all eight rows in FP32.
The worst maximum absolute discrepancies were `1.21e-6` for backbone `z`, `4.77e-6` for EOS logits,
and `1.91e-6` for decoded latents. This removes a substantial prefix, BOS, KV-state, or shifted
history mismatch from the supported explanations for the observed generation failures.

### CFG and decode steps help, but do not solve the task

Probe 3 moved the seen mean from 0.242560 under the original CFG-2/one-step gate to 0.273810 under
CFG-1/one-step and 0.379365 under CFG-1/four-step. Four-step decoding recovered five nonzero seen
targets, but three still scored zero, and matched CFG-1/four-step held-out and reserved panels both
scored zero. Inference settings affect the measurement, but no tested setting turned step 800 into
a usable model.

### Oracle history dominates content; self-fed error accumulates

Probe 4 encoded each of the first four source rows once, reused the same per-frame noise for every
variant, ignored EOS, and forced every output to the source length. Source Mimi reconstruction,
oracle-correct text, and oracle-wrong text produced the same Breeze string on all four rows. This
does not mean wrong text is harmless. It shows that the ground-truth acoustic history can carry
enough content to mask the text error in this bounded diagnostic.

The mean normalized-latent MSE over all frames rose from 0.002638 for oracle-correct to 0.008220
with ground-truth state rebuilt every four frames, 0.111611 with an eight-frame reset, and 0.799814
for the fully self-fed fixed-length rollout. Block-4 audio retained the source Breeze string on all
four rows. Block 8 lost the expected string for `手剎車`; the fully self-fed rollout degenerated for
`手剎車` and `一串香蕉`. Because block resets rebuilt the entire KV state from all preceding
ground-truth frames, this pattern is evidence of accumulated self-history error rather than a
one-frame correction artifact.

These results support training against a history-distribution mismatch as a hypothesis. They do
not prove it is the only cause: the corpus remains weakly supervised, the control is one speaker,
the text table is newly learned, and the evidence covers only a few short seen phrases.

## Why a generic inference failure is not the known root cause

The untouched English 24-layer teacher generated four finite, non-silent, unclipped, uncapped
positive controls through the same `TrainableTTS.generate` and Mimi path at CFG 1 and CFG 2.
Breeze recovered `Hello. How are you` for both settings and a recognizable mixed-script rendering
of the water request. This demonstrates that the common generation and codec route works in the
environment; it says nothing about Taiwanese quality.

The local full-corpus checkpoint reached step 3000 with a finite checkpoint/optimizer pair and
step-3000 validation loss 0.461412, yet its fixed CFG-1 panel remained pathological. The mean
same-Breeze similarity was 0.003546, only 1/12 rows was nonzero, three rows hit 125 frames, and many
transcriptions were counting or repeated-token loops. That run used 22,606 training rows and
17.8754 hours of unverified Breeze weak labels. More updates and a finite teacher-forced loss did
not produce usable free generation.

Together, the healthy English control, near-exact step-800 streaming parity, and failed local
step-3000 adaptation make a generic decoder or inference implementation fault an unsupported
primary diagnosis. They instead focus attention on adaptation: supervision quality, newly learned
text conditioning, optimization stability, and robustness to self-generated history. None has yet
been isolated as the sole cause.

## Evidence index

- H100 setup and frozen identity:
  `data/taigi-adi-20261026/semantic-control-20261026/provenance.json`
- Original evaluation series:
  `data/taigi-adi-20261026/semantic-control-20261026/seen-diagnostics-summary.json`
- Step-1000 LR-only evaluation:
  `data/taigi-adi-20261026/semantic-control-20261026/lr-stability/eval1000-seen/report.json`
- Probe 1: `data/taigi-adi-20261026/semantic-control-20261026/probe1-step800/results.json`
- Probe 2: `data/taigi-adi-20261026/semantic-control-20261026/probe2-step800/results.json`
- Probe 3 four-step:
  `data/taigi-adi-20261026/semantic-control-20261026/probe3-cfg1-steps4/report.json`
- Probe 4 and its Breeze pass:
  `data/taigi-adi-20261026/semantic-control-20261026/probe4-step800/results.json` and
  `data/taigi-adi-20261026/semantic-control-20261026/probe4-step800/asr-results.json`
- Native English control:
  `data/taigi-adi-20261026/diagnostics/english-teacher-native-control/completion-receipt.json`
- Local step-3000 completion and evaluation:
  `data/taigi-full-corpus/local-longer-20261026/longer-20261026-r2/step-3000/completion-receipt.json`
  and
  `data/taigi-full-corpus/local-longer-20261026/longer-20261026-r2/step-3000/evaluation-cfg1-seed17-asr-20261026T1130/report.json`

This document records completed evidence only. It does not promote a checkpoint, modify a running
job, authorize new spend, or predict the pending step-1200 result.
