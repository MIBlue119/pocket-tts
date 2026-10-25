* _2026-10-25 19:36 +0800 — bounded benchmark complete; no production implementation changes_

# Use the rented GPU efficiently

- **Keep one corpus supervisor and one enhancement process.** Two enhancement workers were slower, and two Breeze workers provided no meaningful throughput gain.

- **Prioritize Breeze batch size 4.** The same 80 seconds of audio took 6.826 seconds sequentially versus 3.655 seconds in one batch: **1.87× throughput**, with all four decoded texts exactly matching. Peak allocated memory rose from 8.388 to 15.885 GiB.

- **Production adoption still needs varied inputs and cache/order checks.** This benchmark used repeated copies of one 20-second sample. No production code, corpus state, hardware or budget deadline was changed by the benchmark.

## Completed benchmark results

All model loads and warmups occurred before timed work. The process comparisons completed equal total source audio (120 seconds); their wall times include worker completion. Each row is one bounded measurement, not a confidence interval or full-corpus speed claim.

| Workload | Serial / 1 worker | Batched / 2 workers | Result |
|---|---:|---:|---|
| Breeze, four 20-second inputs | 6.826 s, batch 1 | 3.655 s, batch 4 | 1.87× throughput; exact text agreement |
| RE-USE, 120 source seconds | 14.731 s | 16.045 s | Two workers take 8.9% longer |
| Breeze, 120 source seconds | 10.892 s | 10.952 s | No meaningful gain from two workers |

RE-USE sampled device memory grew from 5,951 to 11,898 MiB with two processes; Breeze grew from 9,613 to 19,222 MiB. Extra model copies consumed memory without improving this workload. Batch 4 used one resident model, corrected 16 kHz arrays and identical `language=zh`, `task=transcribe` decoding settings.

The first benchmark attempt failed because Transformers consumes keys from input dictionaries; the harness had reused its warmup dictionary. The correction creates fresh dictionaries per call. This affected only the measurement script. One retry used a shorter 240-second service cap to keep both attempts within the original five-minute window; it completed and all benchmark processes exited. No trial was interrupted or overlapped. The existing fixed trial had already finished (one complete long video and one stale failed short-video supervisor record).

Raw reports and logs are under original-root `data/taigi-full-corpus/cloud/concurrency-benchmark/`, with `breeze-batching.json`, `results.json`, per-worker logs and GPU samples. `benchmark-services.log` records both attempts and successful cleanup.

Before enabling batching, verify a group of distinct candidate durations, final partial batches, stable candidate order, exact cache identity and interrupted resume. Match outputs to candidates explicitly, and do not cache a partial or mismatched batch as complete. Keep the same provenance, 290-video inventory and budget invariants. This is the smallest evidence-backed optimization; leave parallel video orchestration as a parked alternative.

## Actual observations

The 11:19:58–11:20:22 UTC GPU trace showed 97–99% utilization throughout. While the long enhancer and short ASR recovery overlapped, reported memory was 15,520 MiB. After the short process left, memory fell to 5,951 MiB while GPU utilization stayed at 97–99%, with roughly 554–559 W power draw. These are sampled device observations, not peak allocation guarantees.

An earlier trace showing zero utilization belonged to a stopped/restarting trial and is excluded from the throughput conclusion. A process snapshot during transition showed about 119% CPU usage for one enhancement worker (roughly 1.2 CPU cores). This does not establish steady CPU saturation; the VM has 24 allocated vCPUs. Full-stage elapsed time still includes acquisition, model loading, segmentation, hashing and output writes, so concurrency could hide some gaps even if the inference itself is already GPU-bound.

During long-video Breeze transcription, 11 samples between 11:28:05 and 11:28:25 UTC ranged from 76% to 98% GPU utilization, with 9,573 MiB device memory. The ASR process used about 119% CPU at the end of that window. This indicates some inference gaps but still substantial GPU activity; batching needs a measured throughput comparison.

Receipts: original-root `data/taigi-full-corpus/cloud/throughput-observation-active.txt`, `throughput-observation-32s.txt`, `throughput-active-processes.txt`, and `throughput-breeze-active.txt`.

## Benchmark protocol used

1. Require both `taigi-corpus-trial-fixed.service` and `taigi-short-asr-recovery.service` inactive, and no existing GPU compute processes. Never stop them to make room. Check that the fixed cloud deadline has sufficient remaining time.

2. Warm Breeze once. Compare four identical 20-second enhanced inputs through the corrected 16 kHz path, first one by one, then a single pipeline call with `batch_size=4`. Preserve language/task decoding parameters. Record both elapsed times, all texts, exact text agreement and peak allocated CUDA memory. Equal-length inputs establish only this workload; variable-length candidates require a later acceptance check before adoption.

3. Compare one resident worker processing six 20-second inputs against two resident workers processing three each: both complete exactly 120 source seconds. Test RE-USE and Breeze separately. All workers load and warm before a barrier releases measured work. Include actual process completion in wall time, record each worker's CUDA peak, and sample device utilization/memory. Do not compare one worker's latency against two workers' combined work.

4. Enforce a five-minute total benchmark cap. Terminate only owned benchmark processes on failure. Preserve reports and return the GPU to the root agent; do not automatically launch the corpus or leave background workers.

5. Prefer the simpler serial path if aggregate speed does not improve materially. A proposed adoption threshold is at least 15% faster total work, no invalid outputs, no OOM and enough memory for observed peaks plus at least 20% device reserve. This threshold is a decision rule, not measured confidence. A marginal result needs another matched trial before production changes.

The standalone measurement script lives under ignored cloud receipts and does not modify production source. No concurrency implementation is authorized by this report alone.

## If later measurements justify parallel videos

Use one wrapper owning two independent existing corpus runners, each with its own inventory, output root, state lock and attempt tokens. Keep a single fixed global inventory, cost ledger and absolute deadline. The wrapper must stop its entire owned process group on cancellation, inspect each child exit, and expose missing/failed/pending outcomes instead of declaring success when one shard finishes.

Normalize the full inventory first to establish all 290 video IDs, original playlist memberships and connected source-group identities. Build candidate shard inventories and normalize them independently; **assert exact global group and membership equality per video before launching**. If a general inventory loses cross-playlist connections after filtering, refuse the split or add an explicit normalized-queue interface in a reviewed change. Do not silently recompute different validation groups.

A real read-only check of this inventory found that filtered two- and four-way shards preserve identity, but naive round-robin splitting is badly unbalanced:

| Shards | Video counts | Source hours per shard |
|---|---|---|
| 2 | 145 / 145 | 3.339 / 22.757 |
| 4 | 73 / 73 / 72 / 72 | 1.718 / 10.937 / 1.621 / 11.820 |

If sharding is adopted, assign videos to the currently lightest estimated-duration shard, retaining original queue order within each shard. Unknown durations need a declared conservative estimate and explicit outcomes; none can be dropped. Verify a disjoint union of exactly 290 IDs and preserve the inventory hash and shard mapping across restarts. The aggregate merger must use each completed video's actual release path, retain source groups and freeze one final release before training. Never let multiple supervisors write the same state or output directory.

Four workers are only a later benchmark candidate, not a default justified by 96 GB VRAM. Two workers already held both models while the GPU was nearly saturated. Keep the existing serial behavior unless measured end-to-end progress improves enough to offset added contention and recovery complexity.

Self-check: Batch 4 improved the measured Breeze workload; duplicate model processes did not. Full-corpus throughput and variable-input correctness remain unverified until integration acceptance tests pass.
