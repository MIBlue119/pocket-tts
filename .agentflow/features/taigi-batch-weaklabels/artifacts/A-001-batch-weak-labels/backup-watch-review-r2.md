* _2026-10-25 19:23:33 +0800 (gpt-6-astra/inherited)_

Reviewed commit: d4c109a8a4b9c0e55b4920d13d69ff2bc3e39d8f

# Backup controller follow-up

- The prior B-1 blocker is resolved. A confirmed ended service with stale or missing job JSON now proceeds through verified backup and the explicit stop helper, while recording the job as interrupted rather than successful.

- No new actionable blocker was found. The prior command-boundary, backup-hash, fixed-deadline and no-deletion assessment remains applicable. Actual cloud operation still depends on the configured exact-instance stop helper and independent deadline watchdog.

Outcome: PASS

Minimality: PASS

Conformance: PASS

## Evidence and scope

- Reviewed the watcher and test correction from `d501685351b5d92689fb815e529fd756436f302d` to the commit above. Reused the coordinator's 14 passing tests, including stale-running and missing JSON, unstarted/missing/restarting units, lingering cgroup writers and a changed invocation during mirroring. No repeated suite or remote call was necessary.

- `ended_invocation` requires a loaded, inactive/failed unit with `Restart=no`, no main/control PID and an empty cgroup. It also requires evidence of a previous start or failed attempt. The remote poll counts processes in the service cgroup and its children. The final poll repeats those checks and compares invocation identity and start timestamp before hashing.

- Receipt fields now distinguish the job's recorded state, actual outcome and incompleteness from backup completion. A stale record or non-success service result cannot claim successful training. The missing JSON case remains eligible for backup only when service history establishes that an attempt ended.

- The new checks are directly justified by B-1 and the stated no-restart/no-writer operator contract. Simply accepting every inactive unit would also accept a unit that had never run; the added history and invocation checks avoid that mistake without introducing a service framework or cloud API logic.

Self-check: The original blocking report remains preserved, B-1 is explicitly resolved at the exact reviewed commit, and code acceptance is not represented as proof of a stopped VM or final artifact delivery. No source, Git state, cloud resource, running process or GPU workload was changed. Only this follow-up report was written; no Agentflow invocation or delegation occurred.
