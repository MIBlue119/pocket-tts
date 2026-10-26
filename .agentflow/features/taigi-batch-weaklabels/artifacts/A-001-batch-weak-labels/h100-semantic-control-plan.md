# H100: one bounded semantic text-control experiment

2026-10-26. Research only: no GPU operation, model/config change, or alteration of the running 3090 experiment. Inspected local training source and actual cloud-prepared receipts `ef8b74e578f2d45d1fb2`.

## Decision

Run **800 optimizer updates from the cached English 24-layer teacher**, using the existing 4096-token Mandarin tokenizer with a newly reset embedding, on **one existing ADI training speaker**. Do this before the full multi-speaker weak-semantic release. Keep the 3090 step-3000 run intact. Do not wait for its multi-GB optimizer upload: it is a different experiment, and step-1000 free generation has not established a useful semantic initializer.

This answers a deliberately smaller question: can the complete training/inference path learn different Mandarin labels paired with different Taigi utterances when speaker variation is removed? It cannot demonstrate ordinary-sentence translation or natural speech, and success does not identify which changed recipe parameter helped.

## Actual data selection

Use only accepted rows from Sol's semantic release whose preserved group is:

`adi:ffcc36b9afe31ec50f7742c0c4c614f1a5b40151:speaker:7148c1ddc1ad16beb838`

Original client ID: `64018a4c52c167912f6eb10d451155413420b48897525e5f14016c77e91073ee37a93dd5568ef3b27b28a8b725ea2036b53155b122a6b740382a27837bd31d7e`.

An independent join by full original clip path AND sentence ID, preserving prepared splits and removing diagnostic aliases plus validation-target aliases, found:

- 284 training clips, 258 distinct Mandarin targets, 603.504 seconds of audio for this speaker.
- 16 reserved dictionary-supported diagnostic clips from this same speaker. All remain outside training.
- Full preliminary weak release: 867 training / 79 validation clips, after the above filters. Sol's certified release/report is authoritative if stricter checks change these counts.
- 77 prepared diagnostic clips in total: 70 original prepared-train / 7 prepared-valid. Raw mapping counts of 80 must not be reported as usable prepared clips.

Do not move any prepared validation row into training. Use the existing speaker-disjoint weak validation manifest (79 preliminary rows) for a separate teacher-forced diagnostic. It is not same-speaker validation. Evaluate the 16 reserved same-speaker controls as their own slice. Select the one speaker only after semantic/alias exclusions, rather than re-admitting original rows based on speaker identity.

No voice prompt is needed for this one-speaker control. The manifest stays `path,start,duration,transcript`. Sentence mode supports an optional independent `prompt` object; `training/dataloader/loader.py::_sample_sentence` rejects target overlap, repeated transcript and duplicate waveform. A future full multi-speaker run needs explicit same-speaker independent prompt construction or acknowledgement that it is an unconditional speaker mixture. `max_voice_prompt_sec` alone does not create prompts. A fixed default prompt at serving time can preserve a text-only user interface, but is not validated by this zero-prompt control.

## Concrete settings

Apply in a fresh run directory; do not reuse a directory containing optimizer/checkpoint files.

```yaml
model_config: pocket_tts/config/english_2026-04_24l.yaml
model_overrides:
  flow_lm.lookup_table.tokenizer_path: <existing copied job-200 tokenizer.json>
  flow_lm.lookup_table.n_bins: 4096
start_from_pretrained: true
reset_text_embedding: true
freeze_head: false

data:
  train_jsonl: <certified weak-semantic top-speaker train.jsonl>
  valid_jsonl: <certified weak-semantic preserved-speaker valid.jsonl>
  sampling_mode: sentence
  precompute: false
  max_duration_sec: 20.0
  loader_procs: 1
  num_bucket_batches: 1
  max_voice_prompt_sec: 5.0
  prompt_trim_max_sec: 0.0
  final_punct_dropout: 0.0

flow:
  type: lsd
flow_batch_multiplier: 4
eos_loss_weight: 0.1
text_dropout: 0.2
voice_dropout: 0.0
optim:
  type: adamw
  lr: 0.0002
  schedule: constant
  warmup_steps: 100
  max_norm: 1.0
batch_size: 16
grad_accum_steps: 4
max_steps: 800
compile: false
stats_ema_steps: 0
ema_decay: 0.999
seed: 42
log_freq: 10
valid_freq: 200
num_valid_batches: 4
ckpt_freq: 100
num_ckpt_keep: 3
sample_sentences: []
sample_temp: 0.3
sample_cfg_coef: 2.0
```

Set the actual run directory and resolve manifest/tokenizer paths from the release receipt. The existing tokenizer was actually loaded with Hugging Face Tokenizers and checked on all 805 weak labels plus 64 diagnostic labels: **zero `<unk>` rows**, at most 14 tokens per target, vocabulary size 4096. SHA-256 `1019fc7f25a85e29e95c88ef36d09b1dfedab59a32ebadb8c495595fe6fa9d07`. This rules out missing-token coverage for these strings, not poor token segmentation or unlearned embeddings.

Why these settings:

- Local official `training/configs/finetune_language.yaml` supplies English 24L, embedding reset, LR 2e-4, effective batch 64 and FBM 4. FBM 4 gives more noise samples per audio position; it does not multiply the number of independent text/audio examples. Effective batch 64 is a recipe choice, not a proved minimum for this dataset.
- Warmup 100 is an explicit bounded-pilot deviation from the official 1000. Otherwise an 800-step experiment never reaches the recipe LR. It gives 12.5% warmup, then 700 full-LR updates. Do not call this warmup validated.
- **`precompute: true` is unsupported with `sampling_mode: sentence`.** Both `DataArgs.__post_init__` and the loader reject it. Do not switch to aligned mode or fabricate word boundaries for Mandarin semantic translations. On-the-fly Mimi is the working path today.
- One loader and one bucket avoid shard starvation on small data. Physical16/accum4 is the initial setting only if the root's H100 real smoke fits. If not, physical8/accum8 preserves effective64; do not spend the experiment on repeated OOM tuning.
- Keep pretrained codec normalization (`stats_ema_steps: 0`) to avoid estimating it from ten minutes of one speaker. Evaluate raw weights because EMA0.999 retains much of initialization during only800 steps.
- Compile is off for the short run. Its benefit must be measured before paying compile startup; no claimed H100 speedup here.
- 800 x64 =51,200 sampled examples, about180 passes over284 clips. This is an intentional capacity/conditioning control, with substantial memorization risk.

The built-in validation cycles the loader. Four batches at16 give a fixed64-row subset, **not a complete79-row validation**, and five batches would repeat part of the data. For an exact whole-validation report, run the existing external evaluator with batch1 and the actual row count. Keep these metrics separately labeled.

## Generation and decision gates

Capture step0, then short fixed-seed raw-weight generations every100 updates and the complete slices at200/400/600/800. Use the existing isolated evaluator, zero voice, seed17, temperature0.3, CFG2, one LSD decode step, identical bounds per slice. For short ADI words/phrases use100 frames (8 seconds); retain the existing bound for the ordinary-sentence panel. Log capped generations. Do not use live `write_samples` as the quantitative comparison: it does not fix/restore RNG and writes PCM before checking finite waveform values. `sample_sentences: []` above leaves evaluation to the existing dedicated evaluator.

Fix eight **seen** nonidentity semantic mappings before training:

| Mandarin input | Native reference |
|---|---|
| 一百塊錢 | 一百箍 |
| 手剎車 | 手禁 |
| 跌倒 | 跋倒 |
| 一串香蕉 | 一枇弓蕉 |
| 叫車過來 | 喊車 |
| 公雞 | 雞公 |
| 腋下 | 胳下胴 |
| 眼睛看不清 | 目睭花 |

The16 same-speaker reserved Mandarin inputs are: 飛機、海豚、男孩子、廁所、有膽就不要逃跑！、不好意思、頭髮、茶葉、蝴蝶、牙籤、小孩子、屁股、你不要再哭了！、一起、鯨魚、菠菜. Report these separately from seen phrases.

The7 prepared speaker-held-out diagnostic inputs are: 剁雞肉、你說什麼？、肥皂、有錢人、你運氣真好耶！、下午、吃午餐. Without supplying their speaker prompts, this evaluates unseen content whose references come from held-out speakers; it does **not** test cloning those speakers. Retain the existing12 ordinary Mandarin sentences as an out-of-domain panel, separately scored.

Before scoring generated short phrases, run the same frozen Breeze on their source recordings once and preserve those outputs. This exposes phrases Breeze itself cannot recognize. Do not alter the dictionary labels to match Breeze. Use its raw outputs, normalized string score and wrong-input comparisons only as automatic diagnostics; no CER or independently verified semantics claim.

Operational go/no-go at600 and800 (predeclared heuristic, not a quality standard):

1. Any nonfinite loss/gradient/waveform: fail and stop. Repeated duration-cap collapse on the seen panel is also a failed text-control result; do not extend because validation loss decreased.
2. A positive conditioning signal needs stable phrase-specific output: at least6/8 seen generations should match their own source-ASR reference better than all seven wrong references at both600 and800, with no tied/identical source-ASR references counted as discriminable successes. Also require the mean Mandarin-target string score to improve by at least0.15 absolute from step0. These numerical thresholds are experiment triage, not estimated perceptual accuracy. If source ASR supplies fewer than six discriminable references, call that gate inconclusive instead of moving its threshold after observing generations.
3. Save and compare all16 unseen same-speaker controls. Improvement there is useful generalization evidence within this tiny dictionary domain; failure there restricts the result to memorization even if seen phrases pass.
4. At800 stop the control. If positive, use the remaining time for **one** full weak-semantic adaptation decision, with same-speaker prompts or a clearly recorded speaker-mixture limitation. Do not immediately repeat800 updates on the same tiny speaker data. If failed/inconclusive, preserve the evidence and let the independent3090 step3000 result inform the next decision. Do not distill either model without acceptable free generation.

No user listening task is required while the user is away. Human/perceptual quality remains unverified; machine diagnostics can justify further experimentation, not a claim that the requested product is solved.

## Runtime and budget bound

There is no measured throughput for this exact H100 batch/duration/config in this investigation. Estimate from actual updates21–50 after the root's admitted smoke, including all four accumulated microbatches:

`projected_train_seconds = remaining_optimizer_updates / median_measured_updates_per_second`.

Then add measured evaluation, checkpoint and export time, not a guessed H100 multiplier. Illustrative arithmetic only:800 updates at0.5 updates/s take26.7 minutes of training; at0.2 take66.7 minutes; at0.1 take133.3 minutes, before evaluations. These are scenarios, not predictions.

Bound this control to **90 minutes of training plus30 minutes of evaluation/export**, or the current remaining cloud budget/time allowance if smaller; preserve at least45 minutes before22:00 for final artifacts/shutdown. If the first50 updates predict that800 cannot fit, run only to600 if that fits and mark the800 stability criterion untested. Otherwise terminate at the next saved checkpoint rather than exceeding the admitted budget. Follow the existing root budget ledger/watchdog; this research does not authorize additional VM spend.
