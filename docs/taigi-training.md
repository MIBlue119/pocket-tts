# Run a bounded Chinese-to-Taigi training pilot

Use complete semantic pairs of Chinese input text and Taiwanese audio. In the training config, set `data.sampling_mode: sentence` and `data.precompute: false`. Keep the complete utterance as the target. See the [research report](research/2026-10-25-direct-zh-to-taigi-pocket-tts-research.md) for the task definition and remaining quality questions.

Prepare the data manifests and the model config before allocating GPU time. Record source rights, segmentation, label origin such as `asr_pseudo`, and filtering decisions in a provenance file. Split evaluation data by source and speaker before fitting. A low training loss alone does not establish intelligibility or semantic correctness.

Run the launcher from the repository root in the training environment. It invokes `python -m training.train` with the same interpreter.

```bash
python -m training.scripts.run_experiment run training/configs/taigi-pilot.yaml \
  --max-seconds 1800 --grace-seconds 120 \
  --provenance data/taigi/provenance.json \
  --wandb offline --entity weirenlan --project pocket-tts-taigi
```

Use your actual config and provenance paths. `run_dir` and data/model paths inside the YAML resolve from the repository root, matching the child trainer. The config and `--provenance` arguments resolve from the invoking shell. Set a finite training `max_steps` in the YAML as well. The launcher requires a positive finite wall-clock limit. That limit starts when the child launches. Shutdown can take another `--grace-seconds`, plus at most three seconds for W&B cleanup. Config hashing occurs before the training timer starts.

At the deadline, the launcher sends SIGTERM to the entire training process group. The trainer can finish its current step and save a checkpoint. After the grace period the launcher sends SIGKILL. Set the grace period long enough to write the optimizer checkpoint on your storage. A timeout exits with code 124 even if the child saves successfully and exits zero. An external SIGTERM or Ctrl-C exits with code 130. Ordinary child failures retain their exit code. A signal death maps to 128 plus the signal number.

The command prints the run manifest and attempt paths. Inspect these local artifacts.

- `run_manifest.json` binds the exact YAML bytes to SHA-256 hashes of train and validation manifests, supplied provenance files, and a local model config when present.
- `attempts/<id>.json` records the full Git SHA, dirty-worktree flag, identity, start/end timestamps, checkpoint selected for resume, child exit code, and final state.
- `progress.jsonl` retains the trainer's append-only event history.
- `trainer.log` retains child stdout and stderr across attempts. It stays local and may contain paths or trainer diagnostics.

Re-run the identical command to resume. Changes to the YAML, input manifests, or supplied provenance reject resume. This intentionally includes whitespace changes to the YAML. Use a new run directory for a different experiment. The latest checkpoint must have a nonempty matching optimizer sidecar. The launcher refuses pre-existing training output without its run manifest. It does not deserialize checkpoint payloads, hash every referenced audio file, or prevent other programs from modifying inputs during training. Keep versioned input assets immutable. A dirty-worktree flag is an audit warning, not a source snapshot.

A process lock excludes simultaneous launchers and snapshots in the same run directory. Always launch this experiment through the wrapper. A direct invocation of `training.train` does not acquire that lock. If the machine or launcher dies abruptly, an attempt may remain `running`; inspect the actual process before restarting. The OS releases the lock when its owner exits.

## Track progress with W&B

The default `--wandb disabled` never imports W&B. Install the project's optional tracking dependency in the training environment to use `offline` or `online`. Authenticate through the environment or W&B login. Do not put API keys in the YAML or command arguments. Confirm that `weirenlan/pocket-tts-taigi` is private before using online mode.

The logger sends allowlisted numeric training/validation metrics, trainer steps, config/input hashes, and the Git SHA. It disables console capture, source-code capture, Git capture, automatic metadata collection, and system metrics. It does not call artifact, audio, table, model, or transcript upload APIs. Each launch uses a fresh W&B attempt ID so a resumed trainer step cannot roll back a previous dashboard run.

W&B runs in a separate process. SDK import, authentication, or network failure cannot fail training. A blocked logger is terminated after training exits. Its bounded queue can drop dashboard updates under pressure; `progress.jsonl` is the authoritative record. Offline mode stores W&B files locally and does not synchronize them automatically. Dashboard delivery is best effort, so verify the dashboard separately before relying on remote monitoring.

## Create a complete local checkpoint bundle

Stop the launcher, then copy the latest model/optimizer pair while holding the run lock.

```bash
python -m training.scripts.run_experiment snapshot runs/taigi-pilot \
  snapshots/taigi-pilot-step-001000
```

The destination must not exist. The snapshot copies the latest `checkpoint_*.pt`, matching `optim_*.pt`, run manifest, attempt manifests, and available progress log. It checks source and copied SHA-256 hashes and writes `checksums.json`. It writes `READY` last. A failed copy leaves an incomplete directory without `READY`. Treat only a bundle with `READY` and matching checksums as complete. Configs, immutable source data, tokenizer/model assets, and the recorded Git revision must also be retained for reproduction. This is a manual snapshot after stopping training, not a live uploader.

The launcher allocates no cloud resources and enforces no monetary budget. For the authorized US$50 pilot, choose a provider price and reserve storage and shutdown costs before converting the remaining compute budget to `--max-seconds`. Arrange provider-side instance shutdown independently. Exiting training does not stop billing for a running VM.
