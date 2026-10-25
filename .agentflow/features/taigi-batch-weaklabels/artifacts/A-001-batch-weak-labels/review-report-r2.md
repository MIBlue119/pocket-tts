* _2026-10-25 18:58:51 +0800 (gpt-6-astra/inherited)_

Reviewed commit: 2d25d3581e4cd4281c5130a6ebfc3835c71c53eb

# Follow-up review of weak-label batch fixes

**Summary**

- Both reproduced blockers from the first review are resolved at this commit. The code can proceed to the actual dataset and training checks.

- Reused compact manifests must now match their intended parent-derived content. Attempt cleanup can find detached stages after their wrapper dies, including recovery from a token saved before the wrapper PID was recorded.

- This is code acceptance, not completion of the real batch or evidence of useful Taiwanese synthesis. The coordinator reports a successful pause after 19 completed jobs and a restart with the same plan; final data and training results remain pending.

Outcome: PASS

Minimality: PASS

Conformance: PASS

## Finding disposition

- **F-1 resolved:** `_validate_published` reconstructs both compact manifests from the expected metadata and compares their full rows, including split, path, interval and transcript. Refreshing the report's checksum no longer bypasses the intended-content check. The added regression reproduces the original changed-transcript-plus-refreshed-hash trigger and requires refusal.

- **F-2 resolved:** `execute` persists a random attempt token before starting the wrapper and passes it through the inherited environment. It invokes `stop_attempt` on every exit, including an already-dead wrapper. Cleanup scans same-user processes for the exact token, gives a live wrapper time to clean its stages, and terminates remaining owned stages if the wrapper is gone. Repeated scans cover descendants that appear during cleanup; failure to reach termination raises `CleanupError` and stops new admission.

- **Recovery and signal targeting:** Restart checks saved ownership records for all attempts, including previous cleanup failures. A token-only record covers interruption before the PID record update. PID descriptors pin signal targets, while the saved process identity and exact token are checked again before signaling, preventing PID reuse from redirecting a signal to unrelated work. Legacy records with a missing recorded live parent fail closed rather than assuming cleanup succeeded.

- **Host adapter:** The small `ctypes` adapter supplies the Linux x86-64 PID-descriptor syscalls because this Python lacks the standard bindings. It checks architecture, propagates operation failures, tolerates already-exited processes and closes acquired descriptors. The guide now names the Linux architecture, kernel facility and readable same-user `/proc` requirements. Existing stage launchers inherit the ownership environment; this is supervision of the trusted local pipeline, not containment of programs deliberately hiding their identity.

## Minimality and scope

- The original Ask and captured decisions remain unchanged: continue automated bounded weak-label production, freeze related sources together, run local training and diagnostics, and deliver Git evidence within the cloud ceiling. The unchanged code and training evaluation retain the first review's scope and source assessment.

- Each new mechanism addresses a reproduced failure: canonical compact comparison closes F-1; durable tokens, process discovery, unconditional cleanup and restart checks close F-2; PID descriptors protect unrelated processes while performing that cleanup. The focused tests and updated design/guide document those guarantees. No new service, model, dependency or cloud resource was introduced.

- I considered simply invoking the old wrapper cleanup on every exit and retaining only its PID. That smaller change still cannot find reparented stages after the wrapper disappears, as the original probe established. Persisting the token only after launch would leave a crash window. Reusing the existing metadata comparison without reconstructing compact rows still permits F-1. The implemented additions are necessary for the accepted guarantees; a broader process framework is unnecessary here.

## Evidence and remaining limits

- Reviewed the exact correction diff from `4532e87dead136d6521619d2a73777b83269323f` to the commit above, the updated design and guide, and the frozen second dispatch, facts and plan. Reused the first report's review of unchanged merge connectivity, seeded initialization, raw checkpoint loading, bounded generation and validation RNG behavior.

- Reused the coordinator's current evidence: 13 batch tests passed in 5.65 seconds, including abrupt wrapper death, token-only and dead-parent recovery, unrelated-process survival and ownership persistence before launch; the exact compact-tamper regression passed in 0.69 seconds. These directly exercise the two previously reproduced failures. No further test run was necessary for an uncovered correction risk.

- The coordinator reports gracefully stopping the real producer after 19 completed jobs, verifying its GPU processes were gone, and resuming the unchanged plan with the corrected runner. This report does not independently certify subsequent job counts, final merged data, tokenizer parity, actual training/validation, generated audio, checkpoint restoration or remote Git delivery.

- No GPU work, process interruption, model download, product edit, Agentflow invocation or delegation occurred during this follow-up. The first blocking report remains unchanged. The reviewer operates in a separate native context with shared filesystem permissions; read-only behavior is instructed, not operating-system enforced.

Self-check: Both original findings are explicitly resolved against the named commit and relevant test evidence. Each required verdict appears once. Code acceptance is separated from unfinished operational and speech-quality acceptance. Only this follow-up report was written, and the active producer was not touched.
