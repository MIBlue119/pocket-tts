* _2026-10-25 20:06 +0800 — operational deadline update; producer kept running_

# Updated USD 100 budget controls

- **All safety deadlines now use 2026-10-26 22:00 Taipei / 14:00 UTC**, epoch `1793023200`. The original cost start, `1792926372` (2026-10-25 19:06:12 Taipei), and prior estimated USD 0.70 were preserved.

- **Production was not stopped or restarted.** Its main PID stayed `21133`; its existing internal 18-hour limit and systemd 18-hour-1-minute cap were not changed. No longer training job was launched.

- **The current watcher still stops the VM when its existing job finishes.** It continues watching `job-200-tw/job.json` and `taigi-full-pipeline.service`, as requested. The integrating agent must change that target to the intended continuation job before relying on work lasting until the new latest shutdown time.

## Cost envelope

| Item | Preserved or updated value |
|---|---:|
| Total authorized budget | USD 100 |
| Original cost start | 2026-10-25 11:06:12 UTC |
| Latest shutdown | 2026-10-26 14:00:00 UTC |
| Entire interval from original start | 26.8967 hours |
| Conservative one-GPU hourly ceiling | USD 1.80 |
| Compute ceiling across the entire interval | USD 48.414 |
| Earlier estimated cost | USD 0.70 |
| Storage, tax and uncertainty reserve | USD 7.00 |
| Conservative planned total | **USD 56.114** |
| Unallocated budget | USD 43.886 |

The calculation counts setup, failures, retries and all running time from the original start; it does not reset the clock at this update. Storage assumption remains 300 GiB network SSD at USD 0.071/GiB per 730 hours, approximately USD 0.785 through the new deadline, included in the reserve. Allocated disks continue costing money after compute stops until cleaned up. Prior spend remains an estimate, not a reconciled invoice.

The public Spot API snapshot at 2026-10-25 12:00:40 UTC still showed USD 0.79/hour for RTX PRO 6000 in us-central1. The control calculation keeps the more conservative USD 1.80 rate. The frozen source assumption remains 290 videos and about 26.1 source hours; source duration is not rented GPU runtime. [Nebius pricing](https://nebius.com/prices).

## Controls and actual readbacks

| Control | Change and verification |
|---|---|
| Local `lifecycle.json` | Atomically updated to total budget 100 and deadline `1793023200`; original start and prior estimate unchanged. Previous ledger retained in a receipt file. |
| Local external API watchdog | `taigi-full-stop.timer` remains active. Its helper reads the shared ledger on every invocation; an actual `--check` observed the exact dedicated VM running with no stop due. |
| Remote poweroff timer | Persistent `/etc/systemd/system/taigi-budget-deadline.timer` now has `OnCalendar=2026-10-26 14:00:00 UTC`. Reloaded and restarted only this timer; actual next-elapse readback matches. |
| Local backup watcher | Unit deadline updated and watcher restarted. Actual live process arguments and its watch receipt both show `1793023200`; target remains the existing production job. Service is active and watching; backup is still in progress. |
| Local Taiwan-egress proxy | Wrapper now reads the shared deadline every five seconds and writes its current deadline and SSH PID to `proxy-runtime.json`. Live readback shows the new deadline and a living SSH child. |
| Remote proxy listener | Verified `127.0.0.1:18080` only. An actual HTTPS request through `socks5h://127.0.0.1:18080` returned HTTP 200. |
| Producer | Same main PID `21133` before and after all changes; still active. Runtime cap remains 18h 1min. |

## Proxy update limitation and resolution

The old proxy process had captured its deadline inside an already-running `subprocess.run(timeout=...)`. Editing the script could not extend that live timeout. One controlled reconnect was therefore required.

The reconnect occurred during enhancement of video `NoAfmwG5oUE`. Before and after, the active stage was still enhancement, no yt-dlp download process existed, and the producer PID remained unchanged. The before/after checks were about 4.3 seconds apart. The new wrapper's live shared-deadline polling means a future authorized deadline change can take effect without reconnecting the tunnel.

No outstanding control remains on the old active deadline. Historical receipts intentionally retain old values. No OCR action, source-code edit, notebook edit, Git operation, production restart or automatic longer-training launch occurred.

Receipts are under original-root `data/taigi-full-corpus/cloud/`: `budget100-controls-receipt.json`, `budget100-proxy-reconnect.json`, `budget100-remote-readback.txt`, `budget100-price-snapshot.json`, the preserved previous ledger/unit/script files, and `proxy-runtime.json`. The backup watch receipt is `data/taigi-full-corpus/production.watch.json`.

Self-check: All live safety deadlines agree; the original cost clock is preserved; ongoing production and Taiwan access were verified after the update. Extending a shutdown deadline does not itself authorize or launch additional training.
