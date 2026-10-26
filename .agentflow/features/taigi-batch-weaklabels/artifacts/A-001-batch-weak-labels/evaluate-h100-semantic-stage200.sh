#!/bin/bash
set -euo pipefail
cd /mnt/HDD2/proj_tts/pocket-tts/.worktrees/taigi-batch-weaklabels
pyrun=/mnt/HDD2/proj_tts/.venvs/taigi-pipeline/bin/python
job=/mnt/HDD2/proj_tts/pocket-tts/data/taigi-adi-20261026/semantic-control-20261026
export HF_HOME=/mnt/HDD2/proj_bark/pretrained_model/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. PYTHONUNBUFFERED=1
"$pyrun" - <<'PY'
import json
from pathlib import Path
from training.scripts.continue_taigi_training import verify_pair,verify_parent
from training.scripts.run_experiment import run_lock
p=Path('/mnt/HDD2/proj_tts/pocket-tts/data/taigi-adi-20261026/semantic-control-20261026')
config,identity=verify_parent(p/'stage-200.yaml')
r=Path(config['run_dir']); a=max(r.glob('attempts/*.json'),key=lambda x:x.stat().st_mtime);x=json.loads(a.read_text())
assert x['state']=='succeeded' and type(x['exit_code']) is int and x['exit_code']==0 and type(x['child_exit_code']) is int and x['child_exit_code']==0
assert x['ended_at']>=x['started_at'] and x['identity']==identity
with run_lock(r):
    pair=verify_pair(r)
    assert pair[2]==200
print('Verified durable step200 completion',a.name,flush=True)
PY
"$pyrun" -m training.taigi.evaluate "$job/stage-200.yaml" "$job/seen-prompts.jsonl" "$job/eval100-seen" --checkpoint "$job/training-200/checkpoint_00000100.pt" --device cuda:0 --max-frames 100 --seed 17 --cfg-coef 2.0 --asr
for category in seen reserved heldout; do
"$pyrun" -m training.taigi.evaluate "$job/stage-200.yaml" "$job/$category-prompts.jsonl" "$job/eval200-$category" --checkpoint "$job/training-200/checkpoint_00000200.pt" --device cuda:0 --max-frames 100 --seed 17 --cfg-coef 2.0 --asr
done
