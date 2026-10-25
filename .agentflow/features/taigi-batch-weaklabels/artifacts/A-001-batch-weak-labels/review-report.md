# Independent review of resumable weak-label training work

- **Recorded:** 2026-10-25 18:55:29 +0800 (Asia/Taipei).

- **Reviewer:** Native `gpt-6-astra`, separate agent context; shared filesystem permissions, with read-only behavior instructed rather than enforced by the operating system.

- **Exact target:** `4532e87dead136d6521619d2a73777b83269323f`.

- **Base:** `ebf708f5ce06dbaf9a99571e14ca74ad23bf3a1f`.

**Summary**

- Fix two reproduced failures before accepting this code: reused training manifests can disagree with their parent labels, and abrupt wrapper death can leave a stage running while the batch retries.

- The changes otherwise fit the requested continued weak-label production and bounded training. Seeded validation and raw-checkpoint generation provide useful diagnostics, without claiming Taiwanese speech quality.

- The real batch and training acceptance remain unfinished at this review. Keep those results separate from passed unit tests and this code review.

Outcome: BLOCKING

Minimality: PASS

Conformance: PASS

## Blocking findings

### F-1 — P2: Reuse accepts a compact manifest with different labels

- **Location:** `training/taigi/merge.py:304–329`, `_validate_published`.

- **Trigger:** An existing merged `train.jsonl` or `valid.jsonl` is edited or regenerated, and its hash in `report.json` is refreshed. The source releases and merged metadata still contain the original labels.

- **Impact:** Repeating the same merge accepts and returns that existing release. Training can therefore read a different transcript, path, interval or split from the content represented by the release identity and parent metadata. A checksum stored beside editable content proves internal agreement only; it does not establish agreement with the requested merge.

- **Evidence:** A CPU-only probe reused the existing test fixture in a temporary directory under `/tmp`. It changed one compact training transcript to `WRONG LABEL`, updated only its recorded manifest hash, and repeated `merge_releases` with unchanged inputs. Output was `reuse_accepted_changed_transcript: WRONG LABEL`. The code already compares full metadata with canonical rows, but never reconstructs and compares the compact manifests.

- **Required action:** Compare both compact manifests against the exact rows derived from the validated expected metadata, including split membership. Add a focused test where the compact content and its report hash change together; reuse must refuse it.

### F-2 — P2: Abrupt wrapper death leaves descendants outside retry supervision

- **Location:** `training/taigi/batch.py:74–77` and `104–122`, `stop_wrapper` and `execute`.

- **Trigger:** The wrapper dies without running its cleanup, for example from SIGKILL or an out-of-memory kill, while one of its separately grouped stages is alive. The same issue affects recovery if the saved wrapper PID is already gone.

- **Impact:** `execute` only invokes cleanup while the wrapper is still running; `stop_wrapper` immediately returns when that wrapper identity no longer exists. The batch can start another attempt while the previous stage continues consuming GPU, disk or network resources. The stated attempt and overall run bounds no longer bound the work it launched.

- **Evidence:** A CPU-only wrapper probe launched a separate-session sleeping child, saved its PID, and called `os._exit(1)` to simulate loss of wrapper cleanup. `batch.execute` raised `wrapper exited 1`, but reported `stage_still_alive_after_execute: True`. Calling `stop_wrapper` with the saved wrapper PID and identity also reported `stage_still_alive_after_restart_cleanup: True`. The probe killed its child afterward; it did not touch the real producer.

- **Required action:** Retain a way to identify and terminate the attempt's stages even after the wrapper has exited, or refuse further admission when their termination cannot be established. Include abrupt-wrapper-death and stale-parent recovery coverage. Merely removing the `poll()` condition will not fix the early return or recover already reparented descendants.

## Scope, minimality and conformance

- **Reconstructed outcome:** The recorded original Ask requests continuation through Agentflow; the later Ask explicitly requests Astra for good training. The captured owner decisions require automated yt-dlp, RE-USE and Breeze weak labels, local training, Git delivery and a total cloud ceiling. The design translates that into bounded resumable jobs, a frozen grouped dataset, held-out diagnostics and truthful evidence.

- **Added concepts:** The finite plan, lock, atomic state, attempt history, checksums, deadlines and process identities support repeatable production and lifecycle bounds. Merge identities, duplicate removal, connected source groups and copied publication protect training data and held-out separation. Unique merged utterance IDs preserve distinct chunks. Fixed prompts, raw-weight loading, waveform checks and optional ASR support the requested training evaluation. The isolated validation RNG and step-zero baseline make checkpoint comparisons meaningful. Tests, user instructions, design, advisor report and route records support verification and delivery.

- **Simplification considered:** Replacing the batch with a shell loop loses verified resume and persistent attempt limits. Reusing the single-video process helper loses graceful wrapper cleanup because stages own separate process groups. Reusing the pipeline's existing split helper alone loses links across separately published releases and original-source hashes. Combining the merger with production loses the useful boundary between an active producer and a fixed dataset. Existing builders, checkpoint loading, generation, audio hashing and schema helpers are reused; no new service or dependency is necessary. I found no smaller design that preserves the requested guarantees.

- **Training source review:** The evaluator seeds the same builder used by rank-zero training, loads raw model state when given a checkpoint, supplies no voice prompt, bounds generated frames and preserves float WAV clipping evidence. Validation seeds and restores CPU and active-device Torch generators, retains `no_grad`, and logs the fresh-run baseline before optimization. Its scores remain teacher-forced, batch-averaged diagnostics. The documented batch-size-one and verified-row-count procedure is necessary because the loader cycles.

- **Split review:** Source-group connectivity is computed before removing duplicates, including the duplicate-bridge case. Original interval conflicts are rejected, distinct chunks receive separate merged IDs, and both splits must retain rows unless train-only is explicit. Perceptual reuploads and speaker overlap remain documented limitations.

## Evidence and limits

- Reused coordinator evidence from the frozen dispatch: batch 9 tests, merge 11 tests, evaluation 2 tests, validation RNG plus configuration 19 tests; scoped Ruff and type checks passed. These are coordinator results, not independently rerun suites.

- Ran only the two new CPU probes above because existing tests do not cover their triggers. Set `CUDA_VISIBLE_DEVICES=''` and disabled bytecode writes; fixture files lived under `/tmp`. No model generation, downloads, GPU tests, Agentflow execution or delegation occurred.

- Reviewed the target diff, design, review facts/plan, advisor report, user guide and relevant existing training/building/checkpoint/process helpers. The working tree contained coordinator updates to tracker/notebook during review; the product files and target HEAD matched the frozen commit at inspection.

- The dispatch records 10 of 56 real jobs complete at its capture time. This review does not certify final batch counts, actual merged-release training, tokenizer parity, generated speech, checkpoint restoration or remote Git delivery. Those acceptance items remain for the coordinator, and no speech-quality claim is supported yet.

Self-check: Both blocking findings have explicit triggers, observed consequences and bounded follow-up checks. Outcome, Minimality and Conformance each appear once with the required verdict. The report distinguishes source review, inherited tests, new probes and unfinished operational acceptance. Only this report and necessary temporary test artifacts were written; the active producer was not interrupted.
