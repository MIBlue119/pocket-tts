* _2026-10-25 — independent focused review of commit 78bc38d3fd7385e62153b0013e508f9fe85a20bd_

# EOS correction review

**No blockers found.** The added terminal position gives even the longest row, including a batch of one, an EOS target after its final real audio frame.

Reviewed the exact two-file diff, surrounding forward/loss code, causal sequence assembly and the new regression tests. Padding occurs before normalization; the false mask excludes that new target from latent statistics and flow loss. The shifted teacher-forcing input now includes the last original frame, so terminal EOS sees the complete utterance. Causal attention prevents the new trailing position from changing earlier audio predictions. The caller's tensors are not mutated.

The two regression cases inspect the actual forward graph for a single row and unequal row lengths: positive EOS gradient at real frames, negative gradient at each terminal position, zero later gradients, unchanged flow targets and unchanged caller mask. Existing reported CPU test results were not repeated; this review launched no GPU work. The fresh training diagnostic and generated-audio evaluation remain the integrating agent's acceptance evidence, not a conclusion of this code review.

Self-check: No blocking correctness or scope issue identified in the reviewed commit; no source changes made.
