2026-10-26T02:25:15Z — GPT-6 Astra

# BOS prefix and collected-service recovery review — PASS

Independent focused source/test review found no remaining blocker in the four files below. This approves the specific inference-prefix and completion-detection repairs, not speech quality or a completed runtime deployment.

Reviewed SHA-256 anchors:

- `training/scripts/publish_taigi_model.py`: `c4d60d6f62dbbd970293a2b5a3b18011a37ef6701ea43a55b724cac2d075d509`
- `tests/test_taigi_publish_model.py`: `9e3f93b14d0e07545d99ee50dede32779c20196e6f036f946ac83a6cdbabba6e`
- `training/scripts/extend_taigi_training.py`: `6e0aeca8d90059e51654746c3dc1cb6e1f63bfc0f8ec33e2b074dee42da3a023`
- `tests/test_taigi_extend_training.py`: `7eef945d742a4d77f31c698d476a0e50580c750b63e3ded0c0509059abb051dd`

## Packaged no-voice inference

The previous example created an empty public streaming state, whereas training and native diagnostic generation prepend bos_before_voice. The repaired no-voice branch allocates exactly the configured BOS prefix length and calls the actual `_run_flow_lm_and_increment_step` API with that embedding as audio_conditioning under torch.no_grad(). This inserts the prefix and advances streaming state before text. Seed17 is reset afterward, preserving the intended generation RNG despite the ignored sampling work performed during prefill. The voice branch still uses get_state_for_audio_prompt without a second prefill; configs without BOS still use an empty state. Weight/tokenizer resolution and generation parameters are unchanged.

The new regression executes the packaged script with a real tiny CPU FlowLM and compares every resulting streaming-state tensor against the public BOS-only reference, including no gradient retention and one prefill call. Existing script tests now cover voice/no-voice, BOS enabled/disabled, and invalid audio. Sol reports32 tests passing and clean scoped Ruff/type checks on the final bytes. No duplicate GPU/full-suite work was performed for review.

This defect does not explain the user's original native step1000 “你好嗎？” audio: that evaluator already included the BOS prefix. Its weak content remains real evidence. Keep previously delivered/HF versions intact and run the repaired script against a fresh package copy for the separate runtime acceptance root has planned.

## Collected user-service history

The actual user service can remain loaded/inactive while systemd clears its runtime start/exit timestamps and InvocationID. The old ended_invocation check correctly rejected a never-started-looking unit but could then wait forever after a real successful first1000 run.

The added fallback requires loaded, inactive, Restart=no, Result=success, zero MainPID/ControlPID/cgroup process count, zero start history and blank current invocation. Only then does it inspect durable trainer evidence. It verifies the intended first run directory, acquires its launcher lock, orders attempts by finite numeric started_at rather than UUID filename, rejects ambiguous latest timestamps, and requires the latest receipt to have succeeded with integer zero child/launcher exit codes, finite ordered timestamps not in the future, and the verified config/input identity. Under that lock it also performs the existing exact1000 checkpoint/optimizer and original200 lineage checks. Returning verified paths avoids repeating the expensive checkpoint load. The new report preserves the terminal service snapshot and durable receipt path/hash. The ordinary historical completion path remains unchanged.

I independently read the real first-stage receipt `56ba8008f2954a6291eb28c67bdb3190.json`: succeeded, both exit codes0, started1792979723.695282, ended1792981055.8012311, exact first-config SHAe8aac3b1ea613f30da430fd494615fd829b496755d9a58585af33fe283dba498 and full tokenizer/lineage/data identity, with W&B sdk_finished. Root will exercise the final helper once against the actual multi-gigabyte checkpoint pair; this review did not duplicate that load during active local work.

Root confirms six new cases failed before the repair and the final24 focused extender cases passed in0.71seconds, with clean Ruff/type checks and exact hash match to this review. The added tests cover missing receipt, newer running/failed attempts despite reversed UUID ordering, wrong identity and invalid timestamp order, plus successful zero-history recovery. Existing tests cover exact1000/parent200 lineage and cancellation containment. Missing or unsuccessful evidence fails closed rather than treating default inactive state as completion.

## Delivery boundaries

Stop only the old waiting controller, preserve its report/output, and start the repaired controller with fresh output after committing these reviewed bytes. Do not restart or mutate the completed first1000 trainer, change the longer-training recipe, or alter the deadline. The root-owned systemd containment contract remains required. The no-voice script runtime proof and actual completion-helper preflight are deployment checks, not grounds to claim better Taiwanese speech.

Self-check: source/test inspection and existing receipt reads only; only this review artifact written. No source edits, GPU/job launch, cloud/runtime mutation, Git operation, delegation or Agentflow startup. Prior review and failed-quality evidence remain preserved.
