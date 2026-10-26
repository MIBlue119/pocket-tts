# Full ADI and YouTube experiment

Status at 19:33 Taipei on 2026-10-26: preprocessing and transfer are running; new training has not started. No checkpoint has met the demonstration goal.

The owner requests a Taiwanese Pocket-TTS model suitable for a social-media demonstration, taking ordinary Traditional Chinese input. The new training must include the completed ADI data and yesterday's processed YouTube corpus. The desired start is about 20:10, but complete data takes priority. The cumulative budget remains US$150 and the existing cloud cutoff remains 22:00 Taipei.

## Inputs and processing

ADI prepared release `ef8b74e578f2d45d1fb2` contains 22,745 strict `nan-tw` clips, about 16.2 hours. Breeze-ASR-26 produces Mandarin semantic weak labels. Native Taiwanese reference strings are retained as references; they are not relabeled as Mandarin translations.

YouTube frozen release `2535bb49f48a2cf1` contains 22,754 accepted weak-label segments: 22,606 train and 148 valid, totaling 64,763.02 seconds. These came from 288 successfully processed videos; two acquisition failures are explicitly outside this accepted release. Replaying all 288 raw parents reproduced the frozen content inventory exactly: no missing or extra segments.

All ASR workers retain Breeze revision `7b992682e7f5ceedd0a41ebec240f01ba469d19e`, adapter `breeze-resample/v2`, batch size 4, and the original source hashes. There are 15,924 retained local cache entries, a verified 212-entry RTX prefix, and disjoint remaining shards of 1,705 local, 2,452 RTX and 2,452 H100 entries. The H100 performs useful preprocessing and staging before switching to training. Neither a partial dataset nor a waiting idle GPU is treated as a training run.

Cloud cache outputs are mirrored every minute. The finalizer waits for complete shard markers and exact identity/hash coverage, verifies the full 22,745-entry union, then runs the unchanged release writer. Completion requires accepted plus quarantined rows to equal 22,745 and all five final manifest hashes to match.

## Training release

The combined builder preserves YouTube source groups and ADI speaker splits. It excludes frozen development/demo collisions from training and records exclusions. A structural audit found short clips with words or characters repeated dozens of times in their weak labels. Training rows with empty labels, character runs of at least six, or repeated two-to-eight-character units at least eight times are quarantined. Validation remains unchanged. These conservative structural checks do not establish semantic accuracy or Taiwanese pronunciation.

Independent review corrected two implementation issues before use: comparing raw and merged utterance-ID namespaces, and repeatedly hashing large shared audio files per segment. Preservation now compares exact content signatures and each unique audio destination is hashed once.

## Bounded training recipe

Continue the exact step-1000 model, optimizer and EMA at learning rate `5e-5`, retaining its learned 4,096-token Mandarin embedding. This is an initialization choice, not a quality endorsement: the earlier step-1200 small-data experiment failed its generation gate. The new experiment changes the data while keeping history noise at zero. It does not use the previously proposed noise intervention.

Keep batch size 16, accumulation 4, flow multiplier 4, full-sentence sampling and on-the-fly Mimi encoding. The repository does not support precomputed latents with this sentence sampler. The first endpoint is 1200, followed by generation assessment before a possible bounded continuation up to step 2000. Timeouts, checkpoint transfer margin and the independent provider cutoff all remain binding.

Preparation verifies both source families, audio/manifests, split separation, tokenizer coverage, duration limits, the exact restart pair and EMA, and real train/validation loader batches. The frozen YouTube corpus already passed a CPU loader check and complete tokenizer coverage check: 220,249 tokens, zero unknowns, maximum segment duration 19.86 seconds. The complete merged corpus still requires its own checks.

Two hundred additional updates sample at most 12,800 examples; this is not a full pass through the new corpus. Manifest inclusion and actual sample exposure must be reported separately. The current sampler drops a partial final batch and does not log source IDs, so update counts alone cannot prove that every individual row was consumed.

## Demonstration evidence

Eight short Mandarin prompts were frozen before the release, including `你好嗎？`, a hot-tea request, rain and umbrella advice, station directions, food, a nine-o'clock meeting, safe arrival and thanks. Exact normalized phrase matches are excluded from parent and new training data; paraphrase independence is not implied.

Generate matched parent and candidate panels with CFG 1, four decoding steps, temperature 0.3 and seed 17. Preserve original audio, commands, checkpoint/config/tokenizer hashes and every attempted output. Test seeds 17, 29 and 43 for the strongest candidate before discussing stability.

The desired result is intelligible Taiwanese conveying the requested meaning without repetition, gibberish or truncation, with at least four useful clips including a sentence beyond greetings. Same-Breeze transcription is a consistency diagnostic, not independent evidence of Taiwanese pronunciation. A private experimental Hugging Face backup is authorized; it must be labeled honestly. A public demonstration claim awaits actual output evidence.

## Recovery artifacts

Operational artifacts are under `data/taigi-adi-20261026/adi-full-training-20261026/`: launch plan and preparer, frozen demo prompts, quality audit, data-finalization controller, combined-source builder, transport/materialization helpers and receipts. Shard identities and repartition receipts are under `local-asr-full-20261026/cloud-shard-20261026/`. Cloud lifecycle ledgers remain under `data/taigi-full-corpus/cloud/`; their original cost clocks are preserved.
