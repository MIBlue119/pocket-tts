#!/bin/bash
set -euo pipefail
cd /mnt/HDD2/proj_tts/pocket-tts/.worktrees/taigi-batch-weaklabels
pyrun=/mnt/HDD2/proj_tts/.venvs/taigi-pipeline/bin/python
job=/mnt/HDD2/proj_tts/pocket-tts/data/taigi-adi-20261026/semantic-control-20261026
stage=$1
case "$stage" in 400|600|800) ;; *) exit 2 ;; esac
export HF_HOME=/mnt/HDD2/proj_bark/pretrained_model/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. PYTHONUNBUFFERED=1
"$pyrun" - "$stage" <<'PY'
import json,sys
from pathlib import Path
from training.scripts.continue_taigi_training import verify_pair,verify_parent
from training.scripts.run_experiment import run_lock
step=int(sys.argv[1]);p=Path('/mnt/HDD2/proj_tts/pocket-tts/data/taigi-adi-20261026/semantic-control-20261026')/f'step-{step}'
config,identity=verify_parent(p/'training.yaml');r=Path(config['run_dir'])
a=max(r.glob('attempts/*.json'),key=lambda x:x.stat().st_mtime);x=json.loads(a.read_text())
assert x['state']=='succeeded' and type(x['exit_code']) is int and x['exit_code']==0 and type(x['child_exit_code']) is int and x['child_exit_code']==0
assert x['ended_at']>=x['started_at'] and x['identity']==identity
with run_lock(r):
    pair=verify_pair(r)
    assert pair[2]==step
print('Verified durable stage completion',step,a.name,flush=True)
PY
previous=$((stage-100))
printf -v previous_checkpoint '%08d' "$previous"
printf -v checkpoint '%08d' "$stage"
"$pyrun" -m training.taigi.evaluate "$job/step-$stage/training.yaml" "$job/seen-prompts.jsonl" "$job/eval$previous-seen" --checkpoint "$job/step-$stage/training/checkpoint_$previous_checkpoint.pt" --device cuda:0 --max-frames 100 --seed 17 --cfg-coef 2.0 --asr
for category in seen reserved heldout; do
"$pyrun" -m training.taigi.evaluate "$job/step-$stage/training.yaml" "$job/$category-prompts.jsonl" "$job/eval$stage-$category" --checkpoint "$job/step-$stage/training/checkpoint_$checkpoint.pt" --device cuda:0 --max-frames 100 --seed 17 --cfg-coef 2.0 --asr
done
