2026-10-26T02:00:53Z — GPT-6 Astra

# Longer-training extender review — PASS

Reviewed the final frozen source and tests independently. No remaining actionable blocker found for the explicitly authorized local exploratory1000→3000→6000 policy. This verdict approves the bounded controller behavior, not model quality.

Exact reviewed SHA-256 anchors:

- `training/scripts/extend_taigi_training.py`: `2b3a247a8a70f2fa8da9adda65ac29b68117546950313739787764b0b35d8585`
- `tests/test_taigi_extend_training.py`: `18f1dbeb52fb1afb6c16d6543c41b7938270a2e0da3900e6ada7bfa8a9bd06f6`

## Scope and correctness

The controller requires explicit longer-training authorization and a systemd invocation. It waits on the named user service using the existing loaded/ended/Restart=no/empty-cgroup checks, refuses an observed invocation change, and requires successful termination before any model/evaluation work. It then verifies an exact1000 checkpoint/optimizer pair, unchanged recipe/data relative to the parent200 config, matching parent identity, and original200 checkpoint/optimizer paths and hashes in the lineage. Later forks reuse the existing fresh-directory, hash-verified model/optimizer/EMA mechanism; the parent is not overwritten.

The earlier generation-improvement admission gate is deliberately not used, consistent with the owner's new instruction to train longer. All endpoint summaries must still have finite diagnostics and12 accounted-for consistency scores. The existing summary permits explicitly labelled decoder-too-short task failures but does not fill in missing ASR on emitted audio. Silent/empty outputs without a score remain a strict incomplete-diagnostics halt under root's chosen policy. Divergence above1.5×the fixed initial baseline halts further exploration. This is a completed-stage check, not continuous intra-stage validation monitoring; the trainer separately aborts nonfinite gradients before an optimizer update.

The absolute deadline, minimum1800-second reserve and explicit6000 ceiling bound admission. Each training/evaluation/export/publication subprocess has a time limit. A supervised numeric124 training timeout retains the partial child and permits selection of a previously evaluated finite candidate; a partial stage is not reported as its intended endpoint. Existing raw selection/export helpers preserve the exact evaluated weight variant and bind config/tokenizer/evaluation/provenance artifacts.

## Lifecycle findings resolved before freeze

The initial draft caught outer subprocess timeout/interruption and proceeded to fallback export. That would have kept the controller alive before systemd could clean nested training sessions. The final code records failure and returns without new export/publication for interruption, outer subprocess timeout, or failure before the first service ended. The accompanying tests verify no subsequent model command is launched. Graceful numeric124 remains a distinct path. This resolves the reproduced control-flow risk without changing shared process helpers.

The final lineage check also verifies both original200 model and optimizer path/hash records, rather than relying only on parent config identity. Tests reject model-hash/optimizer-path changes, recipe changes and a partial first-stage checkpoint.

## Publication and operational contract

Optional publication runs only after successful local selection/export, checks all five selected artifact hashes and output containment, invokes the existing private-only publisher under the absolute deadline, and verifies its private repository receipt. Upload failure is recorded separately and does not relabel completed local training/export as failed. It does not turn an experimental candidate into a usable model claim.

Root's proposed persistent user unit supplies KillMode=control-group, Restart=no and TimeoutStopSec90, with RuntimeMax recomputed immediately before launch. The controller's environment check is not an independent audit of those systemd properties; actual unit verification remains root-owned. Keep the first1000 service independently bounded. No changes to the running first stage or unrelated GPU process are required to deploy this new controller.

## Verification

Sol reports18 focused tests passing and clean Ruff/type checks on the final files. I inspected the actual tests and final source and independently matched both hashes above. Coverage includes owner/deadline requirements, exploration without the old quality gate, divergence, missing diagnostics, cancellation/outer timeout with no export, waiting failure before first-service completion, graceful timeout retention, user-unit polling, lineage/recipe/partial-step rejection, and publication success/failure preserving local results. No additional full-suite or GPU run was justified for this review; root's actual first-stage training is a separate integration observation, not proof of later training quality.

Self-check: read-only source/test review; only this requested review artifact written. No GPU/job launch, source edit, cloud/runtime mutation, Git operation, delegation or Agentflow invocation. Prior failed quality evidence and the verified200 parent remain preserved. Commit identity can be added by root after freezing these exact file contents.
