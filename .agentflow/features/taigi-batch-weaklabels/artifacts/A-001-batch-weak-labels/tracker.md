# Tracker

## Identity

- **Work key:** A-001-batch-weak-labels.

- **Active Ask:** A-001.

- **Goal:** Expand resumable weak labels and prove fixed-release training.

- **Last update:** 2026-10-25 18:48:00 +0800.

- **Evidence commit:** uncommitted.

## Overall state

- **State:** active.

- **Reason:** Work remains.

- **Total:** 4.

- **Completed:** 0.

- **Remaining:** 4.

## Accepted task checklist

- [ ] **T-1:** Build resumable bounded batch processing in training/taigi/batch.py and tests/test_taigi_batch.py; obey design INV-1/2, preserve failures, verify resume and cancellation with focused tests. Source: A-001 and prior weak-label automation request.

- [ ] **T-2:** Build immutable grouped release merging in training/taigi/merge.py and tests/test_taigi_merge.py; obey INV-3/4/5, verify duplicate handling, split isolation and tamper rejection. Source: A-001 and prior ML pipeline/data expansion request.

- [ ] **T-3:** Run actual batch and bounded training/validation, save ignored artifacts and results in docs/plans/2026-10-25-batch-weak-label-results.md; failed jobs stay recorded, no false quality claim. Source: A-001 and prior training authorization.

- [ ] **T-4:** Review final commit, document CLI in docs/taigi-batch-weak-labels.md, deliver Git branch and notebook; resolve relevant review findings before delivery. Source: A-001 and prior personal-repo request.

## Accepted scope changes

- None.

## Current recovery

- **Current item:** T-1.

- **Last proven result:** None.

- **Active blocker or running process:** None.

- **Next safe action:** Dispatch bounded batch and merge implementation slices after committing design..

- **Expected changed files:** training/taigi/batch.py, training/taigi/merge.py, tests/test_taigi_batch.py, tests/test_taigi_merge.py, docs/taigi-batch-weak-labels.md, docs/plans/2026-10-25-batch-weak-label-results.md, .agentflow/, ag.json, .gitignore.

## Completion proof

- **All accepted tasks checked:** no.

- **Blocking accepted decision:** none.

- **Operation running:** no.

- **Next action remaining:** T-1.

- **Evidence status:** current.

- **Judgment:** active.

## Update meaning

- Saving this tracker is a recovery checkpoint, not a stop signal.

- For completed work, Evidence commit names the Git evidence commit, or is not applicable in a plain folder. Local file and test proof is still required.

- Work continues with the next unfinished item unless an independent stop condition applies.
