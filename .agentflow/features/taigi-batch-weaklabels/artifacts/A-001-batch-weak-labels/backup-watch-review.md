* _2026-10-25 19:21:51 +0800 (gpt-6-astra/inherited)_

Reviewed commit: d501685351b5d92689fb815e529fd756436f302d

# Backup controller review

**Summary**

- Fix one reproduced unattended-budget problem before relying on this controller: an abruptly failed service with stale or missing job JSON never triggers verified final backup and early VM stop.

- Command boundaries, explicit helper scope, streamed hash verification and the independent absolute deadline otherwise fit the requested backup-and-stop behavior. Nothing deletes remote or local data.

Outcome: BLOCKING

Minimality: PASS

Conformance: PASS

## B-1 — P2: Abrupt service failure can leave the VM idle until the full deadline

- **Location:** `training/scripts/watch_taigi_cloud_job.py:217–220`, the `inactive and terminal` condition in `run`.

- **Trigger:** The real service reaches `failed` or `inactive` after SIGKILL, an out-of-memory kill or another abrupt exit, while `job.json` still says `running` or was never written. These exits cannot execute the orchestrator's final JSON update.

- **Impact:** The watcher keeps copying files but never attempts final hash verification or the configured stop helper. Under the supplied `Restart=no` and no-writers-after-inactivity contract, the output is already stable. The VM can remain billable and idle for the rest of the absolute budget window even when the failed run's existing artifacts could have been backed up immediately.

- **Evidence:** A CPU-only fixture under `/tmp` supplied `ActiveState=failed`, `SubState=failed`, job state `running`, a future deadline and successful mirror/verification stubs. One real `run()` iteration returned `{'state': 'watching', 'backup': 'incomplete', 'calls': ['mirror']}`. Neither verification nor stop was attempted. Without `--once`, the same condition loops until the deadline. No SSH, cloud or GPU call was made.

- **Required action:** Under the stated service ownership contract, treat confirmed service termination as an aborted/failed job even if its JSON never reached a terminal state. Preserve that distinction in the receipt, recheck inactivity, verify whatever regular output files exist, and invoke the explicit stop helper after successful verification. Keep running, activating, deactivating or restarting services excluded. Add focused stale-`running` and missing-record failure cases. Do not label the training job successful merely because its backup is verified.

## Reviewed safeguards

- SSH destinations and unit names cannot begin with option syntax; remote Python commands use `shlex.join`. Rsync uses an argument list, `--protect-args`, an explicit argument separator and no deletion option. The operator must still supply the authorized production output root, excluding credentials, model caches and the cloud ledger.

- Final backup requires service inactivity before and after the last mirror. Remote regular files are streamed through SHA256 hashing with change checks; local files must match size and digest, remain inside the destination and not be final-component symlinks. Verification deliberately covers regular files, not symlink targets. The supplied no-writer/no-restart contract is necessary because the file walk is not an atomic filesystem snapshot.

- Polling or checksum errors do not claim backup success or stop early. Every remote command is bounded by the remaining absolute time. At the deadline, the explicit stop helper is invoked even with incomplete backup, preserving the spending bound; the receipt retains the incomplete state. Stop execution is separately bounded and recorded. The independent VM watchdog remains necessary if this local controller itself fails.

- Stop scope is one preselected executable helper with a fixed `--stop-now` argument. The planned launcher delegates to the existing exact-instance guarded watchdog. A zero helper exit is recorded as helper success; actual VM stop verification depends on that helper. This review did not invoke it or change any cloud resource.

## Scope, minimality and evidence

- Reviewed only the new watcher and its tests at the named commit. The earlier serial corpus implementation remains covered by its separate review. No parallel processing change is included here.

- Reused the coordinator's seven passing watcher tests, including actual local streaming hashes, checksum rejection, absolute time bounds and explicit helper selection. The additional small fixture was needed because abrupt service death with stale job JSON was not covered. Temporary files were removed afterward.

- A plain rsync loop would not verify final artifacts or safely distinguish active output from a stopped job. Reusing the existing guarded stop helper avoids adding cloud API credentials, resource selection or deletion behavior to this controller. The receipt, bounded polling, final hash manifest and explicit helper are justified by the authorized backup and budget outcome; no broader redesign is needed to fix B-1.

Self-check: One actionable blocking finding is tied to a reproduced trigger, consequence and bounded fix. The report distinguishes job success, backup verification and helper success. It relies explicitly on the operator's systemd ownership contract. No source, Git state, live workload or cloud resource was modified; only this report and necessary temporary test artifacts were written. No delegation or Agentflow invocation occurred.
