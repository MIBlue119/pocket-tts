#!/usr/bin/env bash
set -euo pipefail
repo_dir="${1:-$HOME/pocket-tts}"
cd "$repo_dir"
mkdir -p data/taigi-pilot/cloud-logs
exec > >(tee -a data/taigi-pilot/cloud-logs/setup.log) 2>&1
date --utc --iso-8601=seconds
nvidia-smi
if ! command -v uv >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh -o data/taigi-pilot/cloud-logs/uv-install.sh
  sh data/taigi-pilot/cloud-logs/uv-install.sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version
if [[ ! -x .venv/bin/python ]]; then
  uv --no-config venv --python 3.12 .venv
fi
uv --no-config pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 'torch==2.9.1+cu128' 'torchaudio==2.9.1+cu128'
printf '%s\n' 'torch==2.9.1+cu128' 'torchaudio==2.9.1+cu128' > data/taigi-pilot/cloud-logs/cuda-constraints.txt
uv --no-config pip install --python .venv/bin/python --constraints data/taigi-pilot/cloud-logs/cuda-constraints.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match -e . sphn pyyaml soundfile tqdm jiwer
.venv/bin/python - <<'PY'
import importlib
import json
import torch
import torchaudio
from pathlib import Path

assert torch.__version__ == '2.9.1+cu128', torch.__version__
assert torchaudio.__version__ == '2.9.1+cu128', torchaudio.__version__
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
for name in ('pocket_tts', 'sphn', 'yaml', 'soundfile', 'sentencepiece', 'tokenizers', 'pydantic', 'typer', 'einops', 'safetensors', 'scipy', 'requests', 'huggingface_hub', 'tqdm', 'jiwer', 'training.train'):
    importlib.import_module(name)
torch.manual_seed(0)
a = torch.randn(128, 128, device='cuda')
b = torch.randn(128, 128, device='cuda')
c = a @ b
torch.cuda.synchronize()
torch.testing.assert_close(c.cpu(), a.cpu() @ b.cpu(), rtol=1e-3, atol=1e-3)
record = {'torch': torch.__version__, 'torchaudio': torchaudio.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(), 'memory_bytes': torch.cuda.get_device_properties(0).total_memory, 'matmul_verified': True}
Path('data/taigi-pilot/cloud-logs/gpu-verified.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record))
PY
uv --no-config pip freeze --python .venv/bin/python > data/taigi-pilot/cloud-logs/requirements-resolved.txt
