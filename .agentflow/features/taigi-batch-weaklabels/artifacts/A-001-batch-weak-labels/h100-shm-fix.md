# H100 smoke exit failure: root cause and verified fix

The fresh H100 smoke run succeeded on 2026-10-26 at 11:01 Taipei after enabling lingering for the dedicated VM's `pilot` user. No training source code changed. The failed runs remain intact.

## Cause established by reproduction

VM `computeinstance-e00qm3e1gfftr7ey2y` (`89.169.111.99`) launched training as a system service with `User=pilot`. Its user had `Linger=no`; logind's live D-Bus `RemoveIPC` property was `true`.

Smoke r2 completed five optimizer updates and exported its weights, then failed during process teardown: Python `sem_unlink` reported missing semaphores and Torch's `MapAllocator` could not unlink a shared-memory object. The service exited 134.

A separate CPU-only probe reproduced the failure under the same `systemd-run --uid=pilot` lifecycle. It created one POSIX shared-memory object and one multiprocessing semaphore, checked their paths once per second, then explicitly unlinked the shared memory without suppressing errors. After disconnecting SSH, both objects existed through second 10 and disappeared at second 11 (02:59:07 UTC). The system journal records `user@1001.service` and `user-runtime-dir@1001.service` stopping at precisely that time. The probe stayed alive and then failed with the same missing-object unlink errors. This isolates user lifecycle cleanup from Torch, CUDA, and the trainer.

## Scoped change

Executed on this VM only:

```sh
sudo loginctl enable-linger pilot
```

`Linger` changed from `no` to `yes`. The live `RemoveIPC` policy stayed `true`; no system-wide cleanup policy was disabled. Lingering keeps this background-work user alive across SSH logout. Its persistent marker is `/var/lib/systemd/linger/pilot`. Reversal, after all background work ends, is `sudo loginctl disable-linger pilot`; reversal was not performed.

The identical 26-second IPC probe then retained both objects for every observation across SSH logout and exited cleanly. Its journal records successful deactivation at 03:00:11 UTC. Before/after observations and journals are preserved in the receipts below.

## Actual training verification

Ran fresh `training-smoke-r3`, preserving r2, with exactly the r2 configuration except `run_dir`. The same input manifests, tokenizer, five optimizer steps, one loader worker, offline Hugging Face cache, and offline W&B were used. Bounds remained `--max-seconds 240 --grace-seconds 60` with systemd `RuntimeMaxSec=360` and `KillMode=control-group`.

SSH disconnected repeatedly during execution; there was no persistent SSH session keeping the user alive.

- Attempt: `588c170e08944004bb33b4f1281ee893`.
- Actual receipt: `state=succeeded`, `child_exit_code=0`, `exit_code=0`, `wandb_status=sdk_finished`.
- Actual progress records: optimizer steps `[1, 2, 3, 4, 5]`, validation at step 5, and checkpoint at step 5.
- Trainer: peak GPU memory 6.7 GiB; clean `done`, with no missing shared-memory/semaphore errors.
- Checkpoint: 2,529,129,259 bytes; optimizer: 2,529,152,070 bytes; exported model: 1,344,708,416 bytes. The safetensors header was read successfully and contains 358 entries.
- GPU compute-process list was empty at 11:01:14 and again at 11:01:33 Taipei.
- The transient unit was garbage-collected after successful deactivation; journal evidence and the terminal attempt receipt establish completion, rather than default status values for a missing unit.

The parent independently verified the receipt, checkpoint pair, idle GPU, and linger setting. This task did not alter deadlines or budget controls; the parent subsequently retired the setup cutoff only after that successful preflight, preserving the final 22:00 deadline.

## Receipts

Raw evidence is stored both locally and on the VM under the original repository's ignored directory:

`data/taigi-full-corpus/cloud/h100-20261026/shm-fix/`

Files: `probe.py`, `baseline.jsonl`, `baseline-journal.txt`, `before-user.txt`, `before-removeipc.txt`, `after-user.txt`, `after-removeipc.txt`, `linger.jsonl`, `after-journal.txt`, `r2-unit.txt`, `r3-attempt.json`, `r3-training.yaml`, `r3-progress.jsonl`, `r3-trainer.log`, and `verification.json`.

Full successful training outputs remain on the VM under `data/taigi-full-corpus/cloud/h100-20261026/training-smoke-r3/training/`. No model-sized files were copied locally for this diagnosis.
