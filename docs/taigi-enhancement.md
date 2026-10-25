# Opt-in Taigi enhancement feasibility

`python -m training.taigi.enhance` produces an enhanced **candidate**, with a JSON provenance sidecar. It does not label the audio, change existing releases, or declare improved quality. Original audio is retained. Breeze comparisons should use exactly the same source interval for raw and enhanced variants.

The adapter pins [NVIDIA RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/tree/31f82f2efad1095034492f236e8dac9239ba713b) to revision `31f82f2efad1095034492f236e8dac9239ba713b` (model card license: OpenMDW-1.1). It imports model modules only, loads local JSON config and safetensors strictly, and avoids upstream inference scripts' unpinned Hub requests. CPU assembly uses 50% overlapping windows; only one window enters CUDA at a time. Native supported sample rates are preserved. No bandwidth extension is used.

Use a separate CUDA environment with compatible `torch`, `mamba-ssm`, `numpy`, `einops`, `huggingface_hub`, `safetensors`, and `soundfile`. A preexisting Mamba extension may have the wrong PyTorch ABI. Confirm a tiny actual CUDA inference before claiming compatibility. The model card describes Ampere/A100; other GPUs require an actual compatibility check. Do not change the teacher environment to install these dependencies.

```bash
python -m training.taigi.enhance raw.wav candidate.wav \
  --start 0 --duration 10 --chunk-seconds 5 --device cuda:0 \
  --download-model
```

`--download-model` permits fetching only the pinned model/config/code/license. Without it, the snapshot must already be cached. `--cache-dir` selects the Hugging Face Hub cache root. The command reads at most 600 seconds from the source (default 30); chunks must be 0.5–15 seconds. Output is float32 WAV to avoid hidden integer clipping. Existing output or metadata paths cause failure rather than replacement. Provenance records the original file hash, source interval, mono PCM clip hash, enhanced hash, model revision/artifact hashes, adapter hash, and clipping measurements. Downmixing is an explicit mean across channels. Do not run on ongoing mutable source files.

Before promoting candidates, automatically quarantine nonfinite/invalid output, clipping, silence, repetitive ASR, unexpected text-duration ratios, and overlapping or wrong-language speech. The present adapter rejects invalid samples and records clipping; downstream quality filtering still needs integration. Enhancement may suppress secondary speakers or change words/tones. Raw/enhanced ASR agreement is diagnostic, not ground truth. Keep candidate quality explicitly unverified; listening can add evidence but is not required to run feasibility experiments. Chinese Breeze weak labels remain Chinese semantic inputs paired with Taigi audio, not Taigi verbatim transcripts.

## Next Nebius run

Freeze release A for the trainer. Run one bounded CPU acquisition/decoding producer from explicit video URLs, with total duration/bytes, disk reserve, and retries capped. Add durable stage queues and content/config/model-addressed caches: acquired, segmented, enhanced, transcribed, quality-reviewed, release-ready, quarantine. Preserve original source/group identity across variants and across releases.

Begin with a single inference lease shared by RE-USE and Breeze, batch 1, alongside training. Separate worker processes permit real memory release. Measure actual teacher production batch peaks through warmup, validation, checkpointing, and latent precompute; a small pilot's memory usage cannot establish this envelope. Proposed concurrency gates: no OOM, median warmed training-step slowdown at most 10%, and free memory at least the greater of 12 GiB or 20% of device capacity. Compare teacher alone, teacher plus RE-USE, and teacher plus Breeze for 50–100 representative steps. Pause sidecar admission when a gate fails and unload after the bounded in-flight job. Pause GPU preprocessing during precompute or other unmeasured peaks; CPU acquisition can continue within quota. Model file size does not predict inference peak memory.

After candidate checks, freeze release B. Continue from a verified checkpoint in a new run directory with a new run identity, parent checkpoint hash, release hash, and explicit optimizer/scheduler policy. Keep heldout groups fixed; never hot-edit the active training manifest. Existing trainer restoration loads the latest checkpoint inside its run directory, including its step, optimizer and EMA: set the new cumulative step target accordingly. All work shares the next authorized instance and budget cutoff; this plan does not provision another GPU.
