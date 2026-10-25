# Continuation recovery review

2026-10-25T23:33:50.588302+00:00 — GPT-6 Astra. Verdict: PASS for the bounded four-file source change; actual training admission remains conditional on the completed step-200 evaluation.

Independent SSH read-only inspection of the actual r2 baseline found six decoder-too-short failures (7,5,5,5,5,5 frames), six emitted audio outputs with measured ASR consistency scores (0,0,0,1/9,1/7,0), and two frame caps. There are no silent baseline rows. The evaluator intentionally never decodes or transcribes the short outputs. Treating these known task failures as explicitly labelled zero task-consistency scores is defensible; claiming measured ASR scores for them would not be.

The reviewed correction is appropriately narrow: only the exact too_few_frames_for_decoder status with integer frames0..7, finite nonnegative consistent-zero duration, and absent audio/hash/ASR/score keys gets a derived zero. Normal audio still needs its verified artifact plus actual string ASR and finite numeric score in[0,1]. Unknown, nonfinite, silent, empty and malformed results remain incomparable and fail the gate. Summaries expose score sources and a failure-adjusted field, while retaining the original evaluation bytes and measured scores. Existing health/cap/repetition, coverage, loss, deadline and later-stage improvement thresholds remain unchanged. The regression reproduces the observed six-plus-six baseline and rejects missing measurements/contradictions. Sol reports37 focused tests passing; no duplicate suite was run for this review.

Actual validation loss is .5278602999448776 at step0 and .5000719985365868 at step200, a5.27% decrease. This satisfies the loss bound but is not sufficient evidence to continue. At step200 require complete fixed-prompt diagnostics, at least six healthy outputs, no more than two frame caps and no repetition regression, plus finite training/checkpoint state. If those hold, one bounded bridge to500 is warranted by limited data coverage; it is not evidence of usable Taiwanese. Later500→1000→2000 still requires the existing joint validation/text-improvement gates. Do not spend the full allowance merely because it is available.

Publisher review also passes: the optional relative selection path preserves the same backup-receipt root and five artifact hash checks, rejects absolute/parent traversal/outside-root and final-component symlink paths, and retains the old default. The actual intended continuation-v2 path uses ordinary directories. Internal directory aliases that remain within the verified root are not rejected by this implementation; this does not expand the permitted artifact boundary.

Keep the entire r2 root mirrored. Let the old umbrella terminate and retain its step200 export and selected.json. Start the revised controller with a fresh continuation-v2 directory and explicit deployment receipt; point the publisher's alternate selector there, with a fresh delivery output. Do not overwrite old selections, restart the parent training job, narrow backup to only the continuation subtree, or rearm automatic stop/publication against stale unit/job/selection state. Root owns the systemd/backup transition and unchanged22:00 cutoff/budget clock.

Reviewed file hashes:
- training/scripts/continue_taigi_training.py: `6dfb19789456d4d99f702e8d46ec22cb7241874a5e9ea8beb63acf2887ee5811`
- tests/test_taigi_continuation.py: `e6a6b79b74960a4d1a98dffc02924ab1276f5dab72f3403f1a4b57d019c298ab`
- training/scripts/publish_taigi_when_ready.py: `0deaa0d36f82e97dd122f79a069183c705f4b5055da504b6e6c5cd1f80a0a9c9`
- tests/test_taigi_publish_ready.py: `066e3fc5a4ae2617aa00f4f83a3cbb9fea8986d31a6355579e7f884cabbe8aa7`

Self-check: independent source/tests inspection and SSH reads only; no GPU, runtime/source edits, Git mutation or delegation. Only this review artifact written. No admission or quality claim before actual step200 audio diagnostics.

## Actual step-200 outcome — do not continue unchanged

Independent read-only SSH inspection now confirms `final/report.json` is complete:12 emitted audio outputs,7 frame caps (baseline2), and repeated unrelated ASR outputs including “你給我回來” on several distinct prompts. The gate must refuse generation_regressed. Step200 is an experimental archival candidate, not usable Chinese-to-Taiwanese speech. Small character-overlap gains from generic unrelated utterances are not semantic success.

The EOS terminal-position fix remains present and correct on source inspection. No new implementation defect causing these caps has been demonstrated. Teacher-forced validation EOS loss decreased from .1138998 to .0314758; flow loss decreased from .5164703 to .4969244 and was already .4961145 at50. These aggregate losses do not measure terminal recall during free-running generation. The training manifest has22,606 rows, median2.14s and mean2.84666s;200updates×8 nominal examples is only .07078 corpus passes. One EOS position per ceil(duration×12.5)+1 positions gives an approximate2.70% positive-label fraction. A new random4096-entry text table plus short exposure and imbalanced EOS supervision are plausible causes of weak conditioning/calibration; these observations do not establish a causal diagnosis or justify a new loss weight.

A concrete evaluation-contract difference is confirmed: training/modules/model.py generate defaults to eos_threshold=-1.0, and the current evaluator does not override it; TTSModel inference uses the runtime config default, ultimately DEFAULT_EOS_THRESHOLD=-4.0 when the24-layer config omits an override. This is not proof that the threshold explains failed text adherence. The repository's training README also explicitly warns that more negative thresholds trade cap failures for premature/empty output on undertrained models.

The next justified experiment is **evaluation only**, with no optimizer updates: pin the raw step100 checkpoint (lowest observed validation loss .4990983) and raw step200; hold prompts, seeds, temperature and CFG fixed; compare threshold-1 with the actual runtime threshold-4, optionally a single middle-2.5 setting if needed. Retain old diagnostics separately and report threshold explicitly. Assess caps, too-short/silent outputs, actual prompt-specific content and the existing paired distinctions (negation, counts, destinations), not duration or character overlap alone. Allow a hard10–15minute compute envelope for this targeted question. If neither checkpoint produces meaningful prompt-specific content, stop; do not launch500 or tune thresholds until a superficial gate passes. A further training recipe change would then need separately justified evidence, not an automatic continuation.

Proceed with verified backup and stopping the idle VM. The small evaluation experiment can be scheduled from the backed-up checkpoint on local CPU or explicitly restarted bounded compute; retaining an idle billable GPU is unnecessary. No training/GPU/evaluation jobs were launched by this review, and no runtime state was changed.

### Correction and local-only follow-up strategy

The preceding paragraph's implication of a **current delivered-package threshold mismatch is withdrawn**. Source inspection of `training/scripts/publish_taigi_model.py` confirms its shipped `inference.py` explicitly passes `eos_threshold=-1`, matching the training evaluator. The general library default is-4, but it is overridden in this package. Comparing thresholds is therefore a calibration hypothesis, not a proven bug fix or a reason to invalidate the measured7/12cap regression.

No new paid experiment is recommended now. Allow the already rearmed verified backup/publication/stop lifecycle to finish. Retain the200 candidate as explicitly experimental. After backup, a local CPU-only diagnostic can use the existing evaluator unchanged through an isolated process-level method wrapper, recording the altered threshold in a separate receipt. Do not edit the evaluation report to imply the old default protocol was used. Example for one fresh evaluation directory (execute only when root schedules the diagnostic):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=30s 900s /mnt/HDD2/proj_tts/.venvs/taigi-pipeline/bin/python - <<'PY'
import argparse
import json
from pathlib import Path
from unittest.mock import patch
from training.modules.model import TrainableTTS
from training.taigi.evaluate import evaluate

job = Path('/mnt/HDD2/proj_tts/pocket-tts/data/taigi-full-corpus/production/recovery-20261026-r2/job-200')
# Keep these diagnostics outside the already verified production tree.
out = Path('/mnt/HDD2/proj_tts/pocket-tts/data/taigi-full-corpus/local-diagnostics/step100-threshold-minus4')
threshold = -4.0
original = TrainableTTS.generate

def with_threshold(self, *args, **kwargs):
    kwargs['eos_threshold'] = threshold
    return original(self, *args, **kwargs)

out.parent.mkdir(parents=True, exist_ok=True)
receipt = out.with_suffix('.protocol.json')
with receipt.open('x') as stream:
    json.dump({'eos_threshold': threshold, 'device': 'cpu', 'checkpoint_step': 100,
               'seed': 17, 'max_frames': 125, 'asr': False,
               'meaning': 'termination calibration hypothesis; no quality claim'}, stream)
with patch.object(TrainableTTS, 'generate', with_threshold):
    evaluate(argparse.Namespace(
        config=job/'training.yaml', prompts=job/'prompts.jsonl', output=out,
        checkpoint=job/'training/checkpoint_00000100.pt', device='cpu',
        max_frames=125, seed=17, asr=False))
PY
```

Run from the matching worktree with the verified checkpoint/config/tokenizer and existing local model caches. The example deliberately disables ASR for the first cheap termination question; it cannot establish text-consistency improvement. Use the same wrapper at-1 for step100 as its comparison, then200 only if needed; the original200/-1 result already exists. Listen to fixed paired prompts before deciding whether a more expensive ASR pass is warranted. If lower caps are replaced by early cutoff or the same unrelated utterances, reject the hypothesis and stop. A timeout is incomplete evidence, not a pass. These commands are a plan only and were not executed in this review.
