#!/bin/bash
set -euo pipefail
/bin/bash /home/pilot/evaluate-semantic-stage.sh 800
cd /mnt/HDD2/proj_tts/pocket-tts/.worktrees/taigi-batch-weaklabels
pyrun=/mnt/HDD2/proj_tts/.venvs/taigi-pipeline/bin/python
job=/mnt/HDD2/proj_tts/pocket-tts/data/taigi-adi-20261026/semantic-control-20261026
export HF_HOME=/mnt/HDD2/proj_bark/pretrained_model/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. PYTHONUNBUFFERED=1
for category in general supplemental; do
"$pyrun" -m training.taigi.evaluate "$job/step-800/training.yaml" "$job/$category-prompts.jsonl" "$job/eval800-$category" --checkpoint "$job/step-800/training/checkpoint_00000800.pt" --device cuda:0 --max-frames 125 --seed 17 --cfg-coef 2.0 --asr
done
