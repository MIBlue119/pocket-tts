# Preserve step 3000, then hand the GPU to ADI

Operational plan, inspected 2026-10-26 approximately 10:40 Asia/Taipei. Commands below are proposed, not executed by this planning agent. The primary agent owns mutations, verification, and publication. No live data/config/controller edits; total budget US$150 and final delivery by 22:00 today remain binding. The owner's later Nebius authorization permits an H100 if needed, subject to the primary agent's separate cost/capacity/shutdown checks; this document does not authorize an unbounded cloud launch.

Latest owner update: total budget is US$150 including prior spend, with the 22:00 deadline unchanged. The primary agent verified the old VM is STOPPED; receipt: original `data/taigi-full-corpus/cloud/stop-and-budget150-receipt.json`. New H100 capacity/cost admission remains a separate check.

Primary-agent update: main-only SIGSTOP has already been executed and verified (`2348782` in `Ts`, runner `2369556` still `Ssl`). The request receipt is the original data directory's `longer-20261026-r2/adi-handoff-request.json`. Do not repeat the pause action or manually CONT. Proceed to the completion guardian below. Current source `43a2185` differs from runtime `64e7ecd` by documentation changes reported by the primary agent; preserve both identities.

## Observed boundary

`taigi-local-long-extend.service` has main PID `2348782`, invocation `3262635f0c1a4b8b917714f05b3e36cb`, `Restart=no`, `KillMode=control-group`, `SendSIGKILL=yes`, `TimeoutStopUSec=90s`, and `RuntimeMaxUSec=11h 32min 31s`. Its working directory is this worktree. The controller is waiting for `run_experiment` PID `2369556`, whose trainer is PID `2369564`, at the step-3000 stage. PIDs must be rechecked before action.

`extend_taigi_training.py` has no stop-after-stage option or stop sentinel. Once step 3000 finishes, it evaluates and automatically admits step 6000 if the result is finite and below the divergence ceiling. Loss improvement is the admission criterion, not demonstrated text adherence. Editing the file now cannot safely change code already imported by the running controller.

Use main-process-only SIGSTOP while the independent child runner finishes. The child was launched with `start_new_session=True`; stopping only the controller does not stop training. The existing systemd deadline still owns the whole cgroup. Do not use cgroup freeze, kill-all, negative PID/group signals, or stop the service before the child completion checks below.

## 1. Freeze admission, preserve the running child

Set these task-specific shell variables in the worktree:

```bash
cd /mnt/HDD2/proj_tts/pocket-tts/.worktrees/taigi-batch-weaklabels
task_python=/mnt/HDD2/proj_tts/.venvs/taigi-pipeline/bin/python
task_old=/mnt/HDD2/proj_tts/pocket-tts/data/taigi-full-corpus/local-longer-20261026/longer-20261026-r2
task_parent=/mnt/HDD2/proj_tts/pocket-tts/data/taigi-full-corpus/production/recovery-20261026-r2/job-200
task_unit=taigi-local-long-extend.service
systemctl --user show "$task_unit" -p MainPID -p InvocationID -p ActiveState -p ControlGroup -p KillMode -p RuntimeMaxUSec
ps -p 2348782,2369556,2369564 -o pid,ppid,stat,etime,args
```

Check `job.json` still reports `stage=training`, `active_step=3000`, the exact expected active child, and the runner command matches that child's config. Require the same invocation and main PID immediately before signaling; if any differ, inspect again instead of applying stale PIDs. Save the service status, job JSON, `/proc/2348782/stat` start time, and process tree in a fresh handoff artifact directory. Then:

```bash
systemctl --user kill --kill-whom=main --signal=STOP "$task_unit"
ps -p 2348782,2369556,2369564 -o pid,ppid,stat,etime,args
```

The main process must show `T`; the runner and trainer must not show `T`. Confirm their progress advances on subsequent reads. Do not manually CONT the controller: it would regain permission to launch step 6000. If a pause cannot be verified, do not start another GPU task.

## 2. Establish real completion, then close the old service

Poll at sensible intervals, separately recording the latest `training/attempts/*.json`, progress, cgroup members, and GPU processes. A progress row at 3000 or the presence of a checkpoint alone is insufficient. After the latest attempt says succeeded, use the existing validators in a bounded CPU process:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 180 "$task_python" - "$task_old/step-3000" <<'PY'
import json, math, sys, time
from pathlib import Path
from training.scripts.continue_taigi_training import verify_parent, verify_pair
from training.scripts.run_experiment import digest, run_lock
p = Path(sys.argv[1])
cfg, identity = verify_parent(p / 'training.yaml')
run = Path(cfg['run_dir'])
with run_lock(run):
    attempts = [(json.loads(x.read_text()), x) for x in (run / 'attempts').glob('*.json')]
    assert attempts
    assert all(type(r.get('started_at')) in (int, float) and math.isfinite(r['started_at']) for r, _ in attempts)
    attempts.sort(key=lambda x: x[0]['started_at'])
    record, receipt = attempts[-1]
    assert len(attempts) == 1 or attempts[-2][0]['started_at'] < record['started_at']
    assert record['state'] == 'succeeded'
    assert type(record['exit_code']) is int and record['exit_code'] == 0
    assert type(record['child_exit_code']) is int and record['child_exit_code'] == 0
    assert record['identity'] == identity
    assert type(record.get('ended_at')) in (int, float)
    assert record['started_at'] <= record['ended_at'] <= time.time()
    checkpoint, optimizer, step = verify_pair(run)
    assert cfg['max_steps'] == step == 3000
    print(json.dumps({'step': step, 'files': [
        {'path': str(x), 'sha256': digest(x)}
        for x in (p / 'training.yaml', p / 'lineage.json', run / 'run_manifest.json', receipt, checkpoint, optimizer)
    ]}, indent=2))
PY
```

Save stdout as the separate handoff verification receipt. Lock acquisition must succeed: `run_experiment` writes its initial success receipt before W&B cleanup, then writes again and releases the lock. A stopped parent cannot reap the exited runner, so the runner may remain `Z` (zombie). `Z` is already exited; a live sleeping runner is not. Confirm no live trainer/runner/worker remains and no GPU process belongs to this job; inspect cgroup descendants too, not just the three sampled PIDs.

Then stop the umbrella:

```bash
systemctl --user stop --no-block "$task_unit"
systemctl --user show "$task_unit" -p ActiveState -p SubState -p MainPID -p ControlGroup -p Result
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Poll until inactive/failed, main PID zero, and its cgroup has no live processes. Systemd may CONT a stopped process while delivering TERM; the controller's TERM handler takes its interrupted cleanup path. Do not independently CONT it. Service stop can need its 90-second timeout; monitor through short calls. The old umbrella may end as failed/interrupted while the child is a verified successful 3000-step run. Keep both facts in the handoff receipt; never rewrite the old job or attempt to claim whole-controller success. No `step-6000` run should have been admitted. If a child failed/timed out, preserve its last complete pair as partial and do not label it 3000 completion.

## 3. Preserve the fixed comparison before ADI training

Once the GPU is free, independently evaluate the exact 3000 checkpoint with the existing 12 prompts, seed 17, 125 frames, temperature 0.3, CFG 1, and no voice prompt. This matches the 200/1000 reports. Use a fresh output directory and a separate systemd unit with `Restart=no`, `KillMode=control-group`, and a short `RuntimeMaxSec` capped by the remaining deadline. The command within that unit is:

```bash
"$task_python" -m training.taigi.evaluate \
  "$task_old/step-3000/training.yaml" "$task_parent/prompts.jsonl" \
  "$task_old/step-3000/manual-evaluation" \
  --checkpoint "$task_old/step-3000/training/checkpoint_00003000.pt" \
  --device cuda:0 --max-frames 125 --seed 17 --asr
```

Allow approximately 15 minutes including model loading; verify complete report, actual checkpoint step/hash, all 12 prompt IDs, and playable audio. Compare repetition, frame-limit hits, termination, and content with 200/1000. Report Breeze similarity as same-model consistency, not CER or independent Taiwanese accuracy. Do not extend the old branch merely because its validation loss fell.

Sol is already executing the six-clip CPU original-versus-Mimi-reconstruction check with a 10-minute bound. Consume that artifact; do not duplicate it. Require source path/start/duration, source and output hashes, sample rates, finite/silent/clipped checks, and original/reconstructed audio. Follow with identical Breeze transcription settings on both only when the GPU is free. If originals transcribe poorly, generated-audio Breeze scores are not a calibrated quality gate. Reconstruction that preserves these few clips supports retaining Mimi; it does not establish all-speaker or all-phoneme coverage.

## 4. ADI release and GPU weak labels

Use Sol's prepared strict eligible training release, with new speaker-disjoint development speakers. Preserve most strict train; do not globally exclude all speakers found in official test. Keep official test outside training and report evaluation strata by seen/unseen speaker and text. Require train/dev sentence and waveform leakage checks and a split manifest. Preserve original native Taiwanese transcript separately from Breeze's Mandarin semantic weak-label candidate and its source/model/revision. These fields are not interchangeable, and transcription into Chinese is not a verified translation.

Current adapter CLI is `python -m training.scripts.prepare_adi_taigi transcribe RELEASE OUTPUT --device cuda:0 --asr-batch-size 1..4`; confirm Sol's final flags before launch. Start a small bounded batch, measure clips/minute, audio-hours/minute, GPU memory and output eligibility. Choose batch size 1–4 from that measurement. Then continue under a deadline-bounded cgroup with incremental resumable output. Freeze a useful completed subset early enough to leave actual training and evaluation time; do not wait for exhaustive ASR if it would consume the evening. Keep quarantined uncertain rows out of the admitted training manifest.

## 5. Fresh adaptation and conditional text-control experiment

The existing `fork_run` allows only `run_dir` and `max_steps` changes and verifies inherited input identities. It cannot authorize replacing data or tokenizer. Build a new branch with explicit new release/tokenizer identity, retained base/checkpoint hashes, source code identity, and a declared initialization policy. Do not weaken identity verification or point the old run at new files. If retaining the old tokenizer and warm-starting step 3000, load weights intentionally with a fresh optimizer and record this as adaptation. If replacing tokenizer, resetting/remapping text embeddings is a separate recipe; resetting all embeddings while calling it an ordinary continuation is misleading. The quality comparison should include the English-teacher initialization if the old branch still ignores text.

First obtain one bounded 32–64-pair text-control diagnostic from traceable, short, preferably single-speaker examples. Keep audio/text pairs fixed, record sample hashes, and test free generation of trained sentences plus matched changed text (numbers/negation/names), with the same seed and no audio prompt. A model that changes output with text has sensitivity, not necessarily correctness. A model that reproduces trained content but fails unseen text has a generalization problem. Failure even on trained examples warrants one controlled recipe adjustment before scaling. Native Taiwanese labels are acceptable for this auxiliary diagnosis, but success does not satisfy the Mandarin-input product goal. Bound this diagnostic to roughly one hour; preserve results even if inconclusive.

Benchmark throughput before selecting the fresh ADI training batch. Candidate physical batches are 4 and 8 with accumulation 16 and 8 respectively (effective batch 64); optionally 16×4 only after measured headroom. Use FBM 4 for the official-style candidate, and retain the existing 1×8, FBM 1 pilot as a recorded baseline. FBM multiplies flow-loss samples per backbone position, not the number of distinct audio/text examples. The supposed 16 GB recipe requirement is not a verified bound on this 3090.

For each benchmark, fix clip-duration distribution, seed, data order, validation frequency, dtype, and optimizer; warm up and measure at least a small stable window within a 5–10 minute cap. Log examples/s, audio-seconds/s, optimizer updates/s, wall time, peak allocated/reserved CUDA memory and observed device memory. Require headroom below 24 GB and no competing GPU process. On OOM, end that isolated unit, confirm all GPU children exit, then retry a smaller batch in a fresh directory. Never change the live old config. Benchmark work should total no more than about 30 minutes.

Only after text control and memory/throughput are understood, compare LR `3e-5` and `2e-4` on identical initialization/data/batch/FBM with fresh optimizer state. A simultaneous batch+FBM+LR+dropout change is a recipe comparison, not evidence that LR caused the result. Align warmup by sample exposure, not raw step count. Compare at equal audio/text examples (and record total audio duration and flow-loss sample multiplier), plus equal-time checkpoints for delivery practicality. For example, 64-example updates give eight times the example exposure of the current effective-batch-8 update; 3000 versus 3000 updates is not a matched comparison. Do not choose another arbitrary long endpoint before measuring available throughput.

## 6. Time, publication, and evidence

Local order: preserve 3000 → fixed evaluation → weak-label throughput/sample release → text control and bounded recipe measurements → best supported fresh adaptation → final inference/publication checks. CPU preparation and Sol's reconstruction can overlap existing training; jobs on the same GPU cannot. Maintain one owner per GPU. With the later H100 authorization, a separately verified cloud worker may run ADI ASR in parallel with local step 3000, avoiding that scheduling delay. Require evidence that the old instance stopped, current spend, exact new hourly/storage cost, remaining budget, and an independently enforced shutdown deadline before admission. Local GPU use does not prove total spend remains below US$150. The primary agent is assigning cloud readiness/cost verification separately.

By 20:30 stop admitting new experiments or broadening ASR. Reserve 20:30–21:30 for final free generation, public inference entry-point checks on ordinary Traditional Mandarin including long punctuation-heavy input, packaging, and private HF upload verification. Keep 21:30–22:00 as recovery margin. Every new persistent unit must use a runtime bounded by the absolute deadline minus its cleanup reserve, not merely an unbounded child timeout. Deadline epoch is `1793023200`.

Package the chosen checkpoint, tokenizer/normalizer, config, exact data/split/weak-label provenance, initialization/recipe, fixed samples, and honest quality limits. Verify the private repository and uploaded revision/artifact hashes. A failed quality gate still permits delivery of a clearly labeled experimental artifact; it does not permit a claim that correct Mandarin-to-Taiwanese synthesis is solved. Teacher quality must precede six-layer distillation.

Confidence: high for the code-derived no-stop-flag finding, main-only suspension boundary, validator paths, and checkpoint/evaluation commands; medium for relative experiment priorities; unmeasured for available throughput, ASR semantic reliability, recipe quality, and the 24 GB memory envelope. This bounded operational plan has been checked against the live unit properties and source; separate document-review agent dispatch was not performed. Execution and final verification remain with the primary agent.
