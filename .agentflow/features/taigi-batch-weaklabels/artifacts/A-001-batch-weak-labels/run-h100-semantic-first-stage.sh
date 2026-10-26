#!/bin/bash
set -euo pipefail
cd /mnt/HDD2/proj_tts/pocket-tts/.worktrees/taigi-batch-weaklabels
pyrun=/mnt/HDD2/proj_tts/.venvs/taigi-pipeline/bin/python
job=/mnt/HDD2/proj_tts/pocket-tts/data/taigi-adi-20261026/semantic-control-20261026
export HF_HOME=/mnt/HDD2/proj_bark/pretrained_model/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. PYTHONUNBUFFERED=1
"$pyrun" -m training.taigi.evaluate "$job/stage-200.yaml" "$job/seen-prompts.jsonl" "$job/baseline0-seen" --device cuda:0 --max-frames 100 --seed 17 --cfg-coef 2.0 --asr
"$pyrun" -m training.scripts.run_experiment run "$job/stage-200.yaml" --max-seconds 5400 --grace-seconds 120 --wandb offline --provenance /mnt/HDD2/proj_tts/pocket-tts/data/taigi-full-corpus/production/recovery-20261026-r2/job-200/tokenizer.json --provenance "$job/provenance.json"
