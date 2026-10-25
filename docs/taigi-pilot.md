# Chinese text to Taiwanese speech pilot

This pilot verifies execution on real audio. Its 32 source clips and 20 training
steps do not establish synthesis quality. Breeze produces Chinese semantic weak
labels; native Taiwanese orthography and word alignment are not prerequisites.

## Environment

Use an isolated Python environment. The tested local GPU runtime uses Python 3.11,
PyTorch and torchaudio 2.9.1 with CUDA 12.8, and Transformers 4.57.1. Keep torch and
torchaudio builds matched. Install the repository and `taigi` dependency group;
FFmpeg is required for optional YouTube acquisition. The training package runs from
the repository checkout.

```sh
uv venv --python 3.11 .venv-train
uv pip install --python .venv-train/bin/python torch==2.9.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-train/bin/python -e . --group taigi
```

Use `.venv-train/bin/python` for the commands below. Authenticate Hugging Face and
W&B through their local credential stores. The 24-layer teacher is gated; access
must cover its pinned model revision. Never commit tokens or copy credential stores
to the training VM.

## Real-data smoke

```sh
python -m training.scripts.acquire_taigi_pilot --limit 32
python -m training.taigi prepare data/taigi-pilot/sources.jsonl \
  data/taigi-pilot/prepared --train-only
USE_TF=0 USE_FLAX=0 python -m training.taigi transcribe \
  data/taigi-pilot/prepared/candidates.jsonl data/taigi-pilot/releases --device cuda:0
```

The last command prints the immutable release directory. Copy its `train.jsonl` to
`data/taigi-pilot/train.jsonl`; the contained paths continue to reference the
release's verified audio copies. Preserve `metadata.jsonl` and `report.json`.
The acquisition records the dataset revision, original split and TRAIL-0.1 license.
Audio language remains source-declared and labels remain unverified.

```sh
python -m training.scripts.train_tokenizer data/taigi-pilot/tokenizer \
  data/taigi-pilot/train.jsonl --vocab-size 512 --model-type bpe
python -m training.scripts.convert_tokenizer data/taigi-pilot/tokenizer.model \
  --out data/taigi-pilot/tokenizer.json --texts data/taigi-pilot/train.jsonl
python -m training.scripts.run_experiment run training/configs/taigi_smoke.yaml \
  --max-seconds 35 --grace-seconds 120 --wandb online \
  --entity weirenlan --project pocket-tts-taigi \
  --provenance data/taigi-pilot/tokenizer.json
```

Add the printed release's `report.json` as another `--provenance` argument. A timed
attempt returns 124 even when the trainer checkpoints successfully. Repeat the
same config and provenance arguments with a larger time limit to verify restoration.
Config or input changes require a new run directory. The limit covers execution,
followed by the explicit checkpoint grace period and bounded tracker cleanup.

The smoke config resets the text embedding and fine-tunes the full pretrained
24-layer flow model. It uses real zero-length voice conditioning, full sentence
targets, and frozen Mimi audio coding. The 512-token vocabulary and batch size one
are smoke settings. A subsequent data-scale pilot needs a broader training-only
tokenizer, speaker/source-held-out evaluation, and an effective batch around 64.

## MLOps and cloud

Git stores code, recipes and reports. Content-addressed releases store audio and
weak-label provenance. Each run stores configuration and input hashes, Git revision,
attempt state, JSONL metrics, model and optimizer checkpoints. W&B receives numeric
metrics and hashes; audio, transcripts and weights stay on disk. TensorBoard is
optional and unnecessary for this first pipeline.

Nebius jobs use a dedicated VM and disk, a bounded launcher, and an external
watchdog that calls the Nebius stop API. Stopping Python or shutting down the guest
does not establish that cloud billing stopped. Verify provider state, retrieve and
checksum artifacts, then remove temporary pilot resources. Retained disks continue
to cost money. The first pilot has an authorized US$50 total ceiling; its planned
four-hour RTX PRO 6000 plus 150 GiB disk cost is about US$7.26 before tax at the
[rates checked on 2026-10-25](https://nebius.com/prices).

On an already provisioned CUDA VM, `bash training/scripts/setup_nebius_pilot.sh`
installs the tested GPU environment, checks the actual training entrypoint imports,
and verifies a CUDA operation against CPU output. It records the resolved packages
and GPU result under the ignored `data/taigi-pilot/cloud-logs/` directory. Provisioning
and the provider stop watchdog must be configured separately before starting a job.

Use the 24-layer model to establish a stronger teacher before distilling a smaller
Pocket TTS student. Expand data only after split integrity, restartability and
automated evaluation are working. ASR agreement from the same model that generated
the labels is a consistency measure, not independent pronunciation evidence.
