* _2026-10-25 19:15 +0800 — bounded compatibility run complete; healthy VM handed off to integrating agent_

# Dedicated corpus VM operations

- One dedicated RTX PRO 6000 Spot VM with 300 GiB storage was created under the existing project. The OCR VM was not borrowed.

- Actual RE-USE inference passed on Blackwell: a 20-second input took 2.467 seconds after model loading, with 2.833 GiB peak allocated CUDA memory. This is a short compatibility sample, not full-corpus throughput.

- Breeze passed actual corrected-16-kHz inference: 20 seconds took 2.313 seconds after loading, with 8.388 GiB peak allocated CUDA memory. All 24 local/remote model hashes match. Full-corpus processing and teacher training have not started.

## Resource and budget controls

- Instance: `computeinstance-u00wakm49f57fd1zp8`, name `taigi-full-20261025`; dedicated disk `computedisk-u00c2qdtad4fk9khqk`, 300 GiB NETWORK_SSD.

- SSH: `ssh pilot@204.12.171.245`; normal port 22. Boot took about three minutes before SSH became available. No network rule changes were needed.

- The cost clock started **2026-10-25 11:06:12 UTC** (19:06:12 Taipei), before creation. Absolute stop deadline is **2026-10-26 07:06:12 UTC** (15:06:12 Taipei). The initial stricter setup cutoff was **2026-10-25 11:36:12 UTC**. Both checks passed by about 11:14 UTC, so the cutoff was retired after the original absolute deadline was armed; the 20-hour clock was not reset.

- Conservative budget: prior estimated USD 0.70 + at most 20 hours × USD 1.80/hour = USD 36 new compute + USD 7 reserve = USD 43.70. Storage remains billable after stopping; cleanup needs its own verification. The USD 50 total cannot be reset on retries.

- Local `taigi-full-stop.timer` was armed before creation. Its helper passed simulated deadline, setup cutoff, exact-ID stop and no-early-stop tests; a real read-only API check passed before provisioning. The timer then successfully inspected the new instance. It checks every 30 seconds and retries through subsequent ticks if an API stop fails; this is not a provider-level guarantee against controller/network failure.

- Cloud-init armed `taigi-budget-stop.timer`; SSH inspection verified its setup-cutoff schedule. After both preflights passed, it was replaced by persistent `taigi-budget-deadline.timer` at the original absolute deadline, including across reboot. It powers off independently of the corpus process. The local controller also stops the exact instance ID. Deadline state lives at original-root `data/taigi-full-corpus/cloud/lifecycle.json`; do not overwrite it with a fresh clock.

## Environment and compatibility evidence

- Committed archive `24a4615` is extracted at `/mnt/HDD2/proj_tts/pocket-tts/.worktrees/taigi-batch-weaklabels`. The integrating agent will supply the final full-corpus commit before any production run.

- Separate environments: `/mnt/HDD2/proj_tts/.venvs/taigi-pipeline` and `/mnt/HDD2/proj_tts/.venvs/taigi-enhance`. Both use Python 3.11.17, torch and torchaudio 2.9.1+cu128. Transformers is exactly 4.57.1. The enhancement environment uses the repository's hash-verified mamba-ssm 2.3.2.post1 CUDA-12 / torch-2.9 / CXX11 ABI wheel. Optional causal-conv1d remains absent.

- Initial RE-USE import failed because the Mamba package imports Transformers. Installing exact Transformers 4.57.1 into the enhancement environment fixed that missing dependency; no extension rebuild or model change was needed. Setup receipts include the failure and correction.

- RE-USE measurements: load 4.523 seconds; actual 0.5-second forward 1.627 seconds; subsequent 20-second enhancement 2.467 seconds; peak allocated 2.833 GiB. Output shape and finite samples passed. GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB reported memory, driver 580.173.02.

- Model snapshots use `/mnt/HDD2/proj_bark/pretrained_model/huggingface/hub`. RE-USE was copied with symlinks followed. Breeze upload was slow; after the integrating agent authorized direct public download, an unauthenticated API check confirmed `private=false`, `gated=false` and the exact pinned revision. The upload was stopped and its remote process verified absent before pinned `snapshot_download(token=False)` fetched Breeze. All 24 expected model files matched local SHA256 hashes. No Hugging Face or W&B credentials were placed on the VM; inference tests ran offline.

- Setup ran under durable `taigi-setup.service` and exited successfully at 11:11:26 UTC. Breeze download also completed successfully. Its model load took 4.532 seconds and its nonempty transcript used `breeze-resample/v2` at 16,000 Hz. Transcript quality remains unverified. Logs, environment package lists, instance receipts and compatibility results are under original-root `data/taigi-full-corpus/cloud`.

- After setup, a fresh API GET independently confirmed the OCR VM still `STOPPED`, resource version `14`.

## Ownership and controls

The integrating agent explicitly acknowledged healthy-VM ownership and the original deadline. This worker launched no corpus or training service. The existing OCR instance remains excluded by an explicit guard in the watchdog.

Local controls:

```bash
systemctl --user status taigi-full-stop.timer
python /mnt/HDD2/proj_tts/pocket-tts/data/taigi-full-corpus/cloud/stop_watchdog.py --check
python /mnt/HDD2/proj_tts/pocket-tts/data/taigi-full-corpus/cloud/stop_watchdog.py --stop-now
```

Cloud controls:

```bash
ssh pilot@204.12.171.245 'systemctl list-timers taigi-budget-deadline.timer --no-pager'
ssh pilot@204.12.171.245 'sudo journalctl -u taigi-setup.service --no-pager'
```

## Training and acquisition readiness follow-up

- The gated 24-layer teacher was copied from the authorized local cache without credentials. The actual training `download_if_necessary` path resolved it offline at revision `492522650173a0653b7575cdc25ae09810e5d741`; its SHA256 is `8fecefe7206ccfb02fe61a1bd1496d3238f534056f19e5dc0421049cbfbfa4df`.

- The smoke config's existing 512-token tokenizer is staged at `data/taigi-pilot/tokenizer.json`, SHA256 `9f9b9b658fa68084897777fa02ea0d849aa2e93aac1fef5a35afd1e39c2a4498`. This is the earlier pilot tokenizer, not a claim that a new full-corpus tokenizer was trained. The public baseline English tokenizer was fetched with `token=False` at pinned revision `00eac05ed3d16bdc3f6b5d598874019c34a89214`, SHA256 `f498428e1eafee50492f7be13dc9bfafcfc12e508cd0eb1b01c92ecd5d8c6687`.

- Actual training, model builder, dataloader, checkpointing and latent-precompute imports passed with `CUDA_VISIBLE_DEVICES` empty; CUDA was not initialized. This verifies dependencies and offline asset resolution, not training execution.

- ffmpeg/ffprobe 6.1.1 were already present. Node was absent. Official Node v22.23.4 was installed at `/usr/local/bin/node` after checking its archive SHA256 against the official Node release checksums. yt-dlp stayed at 2026.10.21; its default extras added the missing EJS 0.8.0 package and required supporting libraries. Actual yt-dlp startup now reports `JS runtimes: node-22.23.4` and detects ffmpeg and EJS. Node must still be enabled through `js_runtimes` in the acquisition command. [yt-dlp's runtime requirements](https://github.com/yt-dlp/yt-dlp/wiki/EJS).

- Receipts: `readiness-verified.json`, `readiness-verified.log`, `readiness-setup.log`, `teacher-sha256.json` and `pipeline-freeze-after-readiness.txt` under original-root cloud receipts. No source edits, training launch or budget reset occurred in this follow-up.

## Actual cloud teacher acceptance on code 93e6967

- After explicit GPU handoff, a fresh isolated run completed exactly five optimizer updates with batch size 1 and accumulation 1. Remote model, conditioner, trainer and checkpoint code hashes matched commit `93e6967` before launch. The run used 17 accepted short-video rows for training and the first two accepted long-video rows for validation; their source groups differ.

- Both validation passes (steps 0 and 5, two batches each), all five training losses and all logged gradient norms were finite. Final train loss was 0.748228, train EOS loss 0.210641 and gradient norm 5.069883. Validation EOS loss was 0.086469 initially and 0.083011 at step 5; total validation loss rose from 0.508893 to 0.802116. These five updates establish execution correctness, not convergence or audio quality.

- The actual step-5 checkpoint, EMA tensors, optimizer tensors and exported model were loaded and checked on CPU: every floating tensor was finite. The EOS weight changed from the pinned pretrained model (maximum absolute difference 0.000585452), confirming the head received an update. SHA256 hashes and byte sizes were recorded for all three artifacts. This check did not perform a resumed training run.

- The trainer reported 6.5 GiB peak allocated GPU memory; one-second device samples peaked at 7,437 MiB. These short batch-1 examples do not establish the full production memory envelope. The service completed successfully in about 80 seconds, within its 180-second runtime limit and 60-second termination grace. Remaining loader/shared-memory helpers were cleared by systemd; the GPU returned to zero allocated device memory before handoff to the integrating agent.

- Isolated remote run: `/mnt/HDD2/proj_tts/pocket-tts/data/taigi-full-corpus/cloud-training-smoke/run`. Local receipts, configs, manifests, metrics, code hashes and verification are under original-root `data/taigi-full-corpus/cloud/training-smoke/`; service logs and the verification script are beside that folder. The three large checkpoint/export files remain on the dedicated VM for inspection and are not claimed as locally backed up.

- No corpus or production training job was launched by this acceptance run. Original budget and shutdown deadlines remain unchanged; no OCR changes or online credentials were introduced.

Self-check: Actual enhancement and Breeze compatibility are verified; full-corpus completion is not claimed. All resource changes concern the one authorized dedicated VM and its disk.


## Production handoff after Taiwan-source recovery

The full 290-video queue resumed at 2026-10-25 19:50 Taipei. The cloud unit is
`taigi-full-pipeline.service`, persistent and enabled, with `Restart=no`,
`KillMode=control-group`, `TimeoutStopSec=60` and `RuntimeMaxSec=64860`.
The current orchestration record is original-root
`data/taigi-full-corpus/production/job-200-tw/job.json`; the frozen corpus remains
`production/corpus/`. At 19:52, five videos were complete, one running and 284
pending. The original `job-200/` failure record is retained.

Two US-region retrieval failures were confirmed as Taiwan-only availability.
A local `taigi-full-proxy.service` owns reverse dynamic SSH forwarding through
the user's Taiwan machine: `ssh -N -T -R 127.0.0.1:18080 pilot@204.12.171.245`,
with batch mode, forward-failure detection and keepalives. The remote listener
was verified bound only to loopback. The producer has `http_proxy` and
`https_proxy` set to `socks5h://127.0.0.1:18080`; localhost is excluded. No SSH
private keys or service credentials were copied to the VM.

The backup watcher was stopped before the producer was interrupted, preventing
an accidental early VM stop during repair. With the corpus lock held, the
existing acquisition function fetched both full source files through Taiwan,
verified and promoted their source reports. The two original failed attempts
per source remain unchanged. Both sources subsequently completed enhancement
and ASR. Receipts are in `production/taiwan-egress-recovery.json` and
`production/transport.json`.

Local `taigi-full-backup.service` is active again against the new job path. It
mirrors the entire production directory every five minutes, preserves partial
copies, and verifies every remote regular file after the persistent service has
ended with no writers. Its latest receipt is
`data/taigi-full-corpus/production.watch.json`. A completion or interrupted-job
backup can trigger the exact-ID stop helper; the helper also stops the proxy.
The proxy independently ends at the original absolute deadline. None of these
repairs reset the budget clock.

The external `taigi-full-stop.timer` and remote `taigi-budget-deadline.timer`
remain independent. The original stop is still 2026-10-26 15:06:12 Taipei.
The watcher stops compute but does not delete disks. After verified final
preservation, inspect receipts and clean up this dedicated instance/disk;
compatibility smoke checkpoints outside the production mirror need an explicit
retention decision before disk removal. The OCR instance is excluded.
