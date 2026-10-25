# Stop-control recovery — 2026-10-26

- **Cause confirmed:** the Nebius federation login expired. The CLI asks for browser authentication and exits 7 before the instance GET, so both the backup stop helper and periodic API watchdog could not reach the stop operation. DNS resolves normally.

- **Fallback repaired without stopping the VM:** local stop_watchdog.py now checks the exact dedicated VM identity over authenticated SSH before any fallback shutdown. A live check verified its identity and passwordless sudo; no poweroff was sent.

- **Provider authentication remains unavailable.** SSH shutdown is recorded only as shutdown_requested, with provider_verified_stopped=false and exit status 3. It cannot falsely tell the backup watcher that Nebius confirmed a stop. Browser reauthentication is still needed to restore API confirmation.

## Failure evidence

The last successful API watchdog receipt was 2026-10-26 00:04:41 Taipei. Later service runs repeatedly exited 7. A bounded read-only CLI call reproduced the interactive-authentication request; authentication URLs and credentials were not logged. The only configured profile, newtaipei-ocr, has auth-type=federation and no service-account/key fields. No accounts, keys, or credential files were created or modified.

The actual production watch receipt records backup=verified, stop=failed, stop_exit_code=1 and ended_at 2026-10-26 04:38:54 Taipei. The expired login blocked the initial GET; the old helper hid captured stderr behind CalledProcessError, leaving the reason absent from normal watchdog receipts. The repair now records a safe error category without copying provider stderr or authentication links.

## Changes and checks

Only the ignored local stop_watchdog.py was changed; its previous version was preserved with a before-auth-repair timestamp. Existing receipts were retained, and new receipts append to watchdog-receipts.jsonl. stop-compute.sh, the budget ledger, timers, production services, source code, and OCR VM were not changed.

Every API/SSH path first checks the fixed instance computeinstance-u00wakm49f57fd1zp8, project project-u00t740wkc00npa9bswr8c, and name taigi-full-20261025. OCR instance computeinstance-u00mbthnfe1t1hp0fm and all other IDs are refused. The fallback uses StrictHostKeyChecking=yes and checks /var/lib/cloud/data/instance-id in the same SSH session before sudo systemctl --no-block poweroff. It issues that command only when the deadline is due or an explicit --stop-now request is made. --check remains non-mutating even with --stop-now.

Self-tests passed for exact API target, OCR exclusion, no early stop, and forced check-only behavior. Mocked SSH tests passed for the same gates, shutdown acknowledgement, and never reporting provider-confirmed stop. A real --check returned identity_verified=true, sudo_verified=true, shutdown_requested=false, provider_verified_stopped=false. The poweroff action itself was intentionally not executed during recovery, so end-to-end shutdown is not claimed as tested.

The local taigi-full-stop.timer remains enabled and active. The remote persistent taigi-budget-deadline.timer remains active for 2026-10-26 14:00 UTC, which is 22:00 Taipei. The original started_epoch 1792926372, deadline_epoch 1793023200, and USD 100 total budget remain unchanged.

## Guest shutdown and billing limitation

Nebius states that guest shutdown follows the instance recovery policy and can restart and continue billing; API/UI stop avoids that recovery. Its FAIL policy stops the instance without restarting. The saved creation/current metadata for this exact VM records recovery_policy=FAIL. Therefore guest shutdown is a reasonable fallback for this VM, but SSH acknowledgement or disconnect is still not evidence of provider STOPPED or stopped billing. [Official Nebius API definition](https://github.com/nebius/api/blob/main/nebius/compute/v1/instance.proto)

The fallback deliberately returns nonzero even after accepting shutdown, leaving the external timer able to retry API verification after login is restored. If SSH is unavailable too, it records stop_control_unavailable rather than claiming success. The independent remote poweroff timer remains in place.

Machine-readable evidence: original-root data/taigi-full-corpus/cloud/stop-recovery-receipt.json and appended watchdog-receipts.jsonl. Root retains ownership of recovery training and any actual stop decision.
