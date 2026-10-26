#!/bin/bash
set -euo pipefail
cd /mnt/HDD2/proj_tts/pocket-tts/.worktrees/taigi-batch-weaklabels
pyrun=/mnt/HDD2/proj_tts/.venvs/taigi-pipeline/bin/python
job=/mnt/HDD2/proj_tts/pocket-tts/data/taigi-adi-20261026/semantic-control-20261026
export HF_HOME=/mnt/HDD2/proj_bark/pretrained_model/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. PYTHONUNBUFFERED=1
for category in heldout reserved; do
"$pyrun" "$job/evaluate_decode_probe.py" --config "$job/step-800/training.yaml" --checkpoint "$job/step-800/training/checkpoint_00000800.pt" --prompts "$job/$category-prompts.jsonl" --output "$job/lr-stability/baseline800-$category" --decode-steps 4
done
