* _2026-10-25 19:05 +0800 — read-only compute research; no resources changed_

# Compute plan for all 290 videos and teacher training

- **Recommendation:** keep full-corpus preprocessing on the working local RTX 3090, then rent one dedicated RTX PRO 6000 for teacher training. This saves cloud budget and avoids an untested RE-USE extension on Blackwell. Cloud preprocessing remains an option after a real compatibility test.

- **Budget:** the USD 50 total includes the earlier estimated USD 0.70. Reserve USD 7 for storage, transfer, applicable tax and uncertainty; limit new GPU charges to USD 42.30. These are spending limits, not a promise that the corpus and a converged teacher fit.

- **Current evidence:** one-GPU Spot capacity is reported available, and the current RTX PRO 6000 and H100 Spot price is USD 0.79/hour. Neither price nor capacity is reserved. The existing OCR VM remains stopped and unchanged.

- **Next action:** finish the local full-episode acceptance run and measure its throughput. Before any paid run, prove checkpoint resume and shutdown controls, select an explicit training step target, and recheck the live price and remaining budget.

## Scope and evidence

The full inventory is 290 unique videos, 26.1 known source hours and a longest video of 3,200 seconds. Every video must retain an explicit outcome, including unavailable, no speech, failed or pending after a budget pause. Accepted training audio can be shorter than the source total. Chinese Breeze labels are semantic weak labels, not verified Taigi transcripts.

The design requires one resident enhancement model per video, followed by one resident Breeze model per video; only one GPU stage runs at once. Use the full-corpus design and its queue contracts, not the earlier 25-video pilot restrictions. The older proposal for simultaneous training and inference is not the default here: freeze the accepted release first, then train.

Measured locally: RE-USE enhanced 20 seconds in 6.97 seconds after 2.44 seconds loading, using 3.063 GiB peak allocated CUDA memory. A Breeze raw/enhanced comparison peaked at 8.395 GiB. These small samples do not establish long-video speed, host-memory peaks, training memory or cloud performance. The earlier 50-step RTX PRO 6000 teacher smoke test establishes that trainer path only; it does not establish the Mamba extension's Blackwell compatibility.

## Available tools and current cloud facts

The available interface is `/home/weiren/.nebius/bin/nebius`, profile `newtaipei-ocr`; no Nebius MCP tools were found. Read-only calls succeeded for platform listing, instance get, project get and tenant capacity advice. The documented CLI command `billing v1alpha1 spot-price list` is absent in the installed CLI; the unauthenticated REST endpoint works. No CLI upgrade is needed for this research.

```bash
/home/weiren/.nebius/bin/nebius -p newtaipei-ocr --no-browser compute platform list --parent-id project-u00t740wkc00npa9bswr8c --format json
/home/weiren/.nebius/bin/nebius -p newtaipei-ocr --no-browser capacity resource-advice list --parent-id tenant-e00g44hnvbppxnbqak --all --format json
curl --max-time 30 -fsS https://api.nebius.cloud/billing/v1alpha1/spot-price
```

The capacity endpoint requires a **tenant**, not a project. An initial project-scoped request was rejected; the corrected tenant request succeeded. Its 2026-10-25 11:03:37 UTC response, based on availability effective at 10:48:42 UTC, reported:

| Option | Preset | Reported Spot availability | Important constraint |
|---|---|---:|---|
| RTX PRO 6000, us-central1 | 1gpu-24vcpu-218gb; 96 GB GPU memory | 7, limit 8 | Existing project supports it; on-demand reported LOW without a positive available count |
| H100, eu-north1 | 1gpu-16vcpu-200gb; 80 GB GPU memory | 8, limit 8, on each listed fabric | Not in the current project's platform list; requires a suitable regional project/network |
| RTX PRO 6000, eu-south1 / uk-south2 | 1gpu-24vcpu-218gb | 8 / 7 | Also needs a suitable regional project/network |

These advisory counts must not be summed across fabrics as an entitlement or treated as guaranteed allocation. The current project is `default-project-us-central1`; its platform list contains RTX PRO 6000, B200, H200 and CPU, but no H100. B200/H200 are outside the requested host choice.

A fresh instance GET confirmed `computeinstance-u00mbthnfe1t1hp0fm` (`walnut-ocr-test-spot`) is `STOPPED`, resource version `14`, on the RTX preset. Do not start, repurpose or modify its disk for the preferred route. If the owner later chooses that fallback, first record its exact configuration and artifact state, use isolated working directories, and restore the original stopped configuration afterward. No fallback changes were made here.

## Prices and spending controls

Official on-demand prices are **RTX PRO 6000 USD 1.80/GPU-hour** and **H100 USD 4.50/GPU-hour**. Spot has a USD 0.79 floor and can change during use; the floor is not a fixed quote. [Nebius price sheet](https://nebius.com/prices).

The public price API returned snapshot `2026-10-25T10:59:42.170320Z`: RTX us-central1 USD 0.79 (effective 10:31 UTC), H100 eu-north1 USD 0.79 (effective 10:51 UTC), RTX uk-south2 USD 0.79, RTX eu-south1 USD 0.90. The API and its meaning are documented in [Getting spot prices](https://docs.nebius.com/signup-billing/spot-prices).

A 300 GiB network SSD costs `300 × 0.071 / 730 = USD 0.02918/hour`, or about USD 0.70/day; a 200 GiB disk costs about USD 0.47/day. All allocated disks, including stopped or unattached ones, continue billing. Compute stops billing when the VM is stopped. Rates exclude applicable taxes. [Compute pricing and storage billing](https://docs.nebius.com/compute/resources/pricing).

Proposed envelope, subject to actual prior charges and tax reconciliation:

| Allocation | Limit | Purpose |
|---|---:|---|
| Prior estimated spend | USD 0.70 | Replace with actual attributable usage when available |
| New compute | USD 42.30 | Includes setup, tests, failed attempts, transfer while running, preprocessing and training |
| Everything else and uncertainty | USD 7.00 | Storage through verified backup/cleanup, applicable tax, any charged backup traffic, price/rounding margin |
| Total | USD 50.00 | Never reset this total on restart or a replacement VM |

At a conservative RTX compute rate of USD 1.80/hour, USD 42.30 buys **23.5 running GPU-hours**. At the observed USD 0.79 it would buy 53.5 hours, but do not assume that price persists. H100 on-demand would buy only 9.4 hours. Default to the conservative envelope unless a provider-enforced Spot maximum price is configured and verified. A proposed USD 1.00/hour Spot ceiling would buy at most 42.3 GPU-hours, but its configuration is not implemented or tested here.

Before launch, subtract all actual prior compute and storage, account for applicable tax, and calculate an absolute UTC stop deadline from the worst permitted hourly rate. Charge setup and retry time to the same ledger. Reject admission when either the deadline or remaining budget cannot cover the next bounded stage plus backup time. Budget alerts alone are not a hard stop.

## Runtime and disk estimates

Pure extrapolation of the small local enhancement sample is `26.1 × 6.97 / 20 = 9.10 hours`. A speculative Breeze assumption of five seconds per 20 seconds of source adds 6.53 hours, giving **about 15.6 GPU-hours before downloads, segmentation, loading, retries and I/O**. Breeze's assumed rate is not a measured full-corpus rate; cloud speed is unknown. At the RTX on-demand ceiling, that subtotal alone is USD 28.12, leaving under eight GPU-hours from the conservative compute envelope for all other work and training. This is why local preprocessing is the stronger budget choice.

Do not promise completion from those estimates. Use the actual full-episode run to measure elapsed seconds per decoded source hour, accepted hours, model loads, bytes and memory. Reforecast after several varied videos; retain all pending videos if the run pauses.

Budget about 36 GB for decoded 48 kHz stereo float32 originals, 18 GB for mono enhanced audio, 18 GB for copied release audio, 10 GB for models and 20 GB for checkpoints. That is roughly 102 GB before environments, downloads, caches, latent precomputation and partial files. **300 GiB total allocated storage is preferable; 200 GiB is a tight minimum** requiring measured cache growth. Include boot storage in the total. Stop new acquisition below a 30 GiB free-space reserve, and bound checkpoint retention and temporary files without deleting the only verified copy.

## Execution and recovery proposal

1. **Validate locally first.** Confirm complete decoded coverage of a long video, atomic queue updates, hash-checked reuse, empty-source accounting and interruption resume. Freeze the source inventory and record the code revision and environment requirements. Avoid cloud billing while debugging acquisition or orchestration.

2. **If preprocessing moves to cloud, gate compatibility.** Keep Python 3.11 / torch 2.9.1+cu128 / Transformers 4.57.1 in the pipeline environment, and the separately pinned mamba-ssm 2.3.2.post1 CUDA-12 torch-2.9 CXX11-ABI wheel in the enhancement environment. Run actual RE-USE CUDA inference, then Breeze, then teacher forward/backward and checkpoint restore on the selected GPU. A successful import or available torch `sm_120` kernels do not prove the Mamba extension works. H100 compatibility is plausible, not verified. Time-box setup/compatibility to 30 paid minutes; on failure, stop and return preprocessing locally rather than repeatedly rebuild on a rented GPU.

3. **Run durably with one owner.** A service manager should own the corpus process, its children and log files; a queue lock prevents duplicate resume. Persist completion only after writing and verifying artifacts. Restart only unfinished stages, with bounded retries, preserving diagnostics and all 290 queue records. Keep models separate in time.

4. **Survive Spot loss.** Store queue, release files and checkpoints on persistent storage. Copy completed metadata and outputs incrementally to the local workspace; verify hashes before counting a backup as durable. A preemption notice may be too short, so periodic completed-artifact copies are primary. A replacement VM must reuse the same global cost ledger and original stop deadline. Do not automatically create or restart unlimited instances.

5. **Train a frozen release.** Record release hash, fixed source-group validation split and parent checkpoint hash. Measure representative teacher steps including validation, latent precompute and checkpoint save; calculate the affordable cumulative step target from measured seconds per step. Preserve optimizer, scheduler and EMA state when continuing. Save checkpoints frequently enough to bound lost work (proposed every 10 minutes), and verify restore plus generated audio before the long run. Convergence is an evaluation outcome, not guaranteed by reaching the budget.

6. **Enforce shutdown outside the job.** Before paid work, test both an in-VM deadline service and an independent local controller that can stop the exact dedicated instance through the API. The controller polls state and elapsed billable time, stops on deadline, idle timeout or failed health checks, and verifies `STOPPED`. In-VM process timeout alone can leave an idle billable VM. Give graceful checkpoint/backup a bounded window, then enforce stop; failures must not extend the budget silently.

7. **Close the billable footprint.** Download and hash-check final artifacts and recovery checkpoints. Delete only task-owned resources after verified backup, and verify their absence; otherwise set a short explicitly budgeted disk-retention deadline. Preserve the OCR resource and unrelated disks. Record final actual usage separately from this estimate.

This document is a reviewable plan only. No VM, disk, pricing policy, service, watchdog or SSH environment was created or changed; no training or corpus run was launched by this research task. Necessary implementation tests and notebook/STATUS/route records belong to the integrating agent's authorized execution work; this worker changed only this plan.

Self-check: All 290 videos remain in scope; prices and capacity are timestamped observations; runtime, compatibility, completion and cost savings remain estimates until measured.
