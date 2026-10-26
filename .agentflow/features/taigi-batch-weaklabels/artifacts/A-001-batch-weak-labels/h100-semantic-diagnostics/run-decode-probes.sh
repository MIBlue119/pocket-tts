#!/bin/bash
set -euo pipefail
cd /mnt/HDD2/proj_tts/pocket-tts/.worktrees/taigi-batch-weaklabels
pyrun=/mnt/HDD2/proj_tts/.venvs/taigi-pipeline/bin/python
job=/mnt/HDD2/proj_tts/pocket-tts/data/taigi-adi-20261026/semantic-control-20261026
export HF_HOME=/mnt/HDD2/proj_bark/pretrained_model/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. PYTHONUNBUFFERED=1
for steps in 1 4; do
"$pyrun" "$job/evaluate_decode_probe.py" --config "$job/step-800/training.yaml" --checkpoint "$job/step-800/training/checkpoint_00000800.pt" --prompts "$job/seen-prompts.jsonl" --output "$job/probe3-cfg1-steps$steps" --decode-steps "$steps"
done
