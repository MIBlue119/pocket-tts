---
title: Bounded Mimi reconstruction diagnostic for Taiwanese Hokkien inputs
date: 2026-10-26
status: measured
---

# Bounded Mimi reconstruction diagnostic for Taiwanese Hokkien inputs

## Result

The pinned `english_2026-04_24l` Mimi codec passed the **structural reconstruction gate** on this
six-clip CPU-only probe: all decoded samples were finite, aligned reconstructions retained the full
input sample count, and the untrimmed decoder output added only 20–72 ms from codec-frame padding.
Reconstructed RMS was 0.847–0.929 times input RMS (mean 0.895).

One cached Breeze-ASR-26 comparison returned the identical Mandarin-character sequence for the
original and reconstruction: `之前好像沒有那個東西` (character-sequence similarity 1.0). This is
same-model consistency on one clip, not an independent assessment of transcript correctness,
Hokkien pronunciation, tone/sandhi, naturalness, or broad codec suitability. No person listened to
the clips.

This bounded result does not support changing or fine-tuning Mimi before the text-conditioning and
weak-label controls. It also does not prove that the codec preserves every Hokkien contrast.

## Inputs and measurements

The run reused the exact codec config and combined checkpoint named by
`data/taigi-full-corpus/local-longer-20261026/step-1000/training.yaml`. Four clips were complete,
untrimmed audio members from the actual first ADI tar and had explicit `locale: nan-tw`. Two were
exact short intervals from the active YouTube training manifest; their language remains
source-declared/weak-label provenance rather than independently verified.

All sources were converted to mono and resampled with polyphase resampling to the codec's 24 kHz
rate. The original source file/member, float 24 kHz input, full decoded waveform, and input-length
aligned decoded waveform were retained in the artifact directory.

| Input | Provenance | Seconds | Latent frames | Full decode extra | Finite | RMS ratio |
|---|---|---:|---:|---:|---:|---:|
| `train-000000001`, 食薰 | ADI first tar, explicit `nan-tw` | 2.448 | 31 | 32 ms | 1.0 | 0.921 |
| `train-000000008`, 銅錢仔 | ADI first tar, explicit `nan-tw` | 2.088 | 27 | 72 ms | 1.0 | 0.902 |
| `train-000000019`, 斷糧 | ADI first tar, explicit `nan-tw` | 2.196 | 28 | 44 ms | 1.0 | 0.929 |
| `train-000000052`, 夭壽喔 | ADI first tar, explicit `nan-tw` | 2.916 | 37 | 44 ms | 1.0 | 0.910 |
| YouTube train row 0 | active manifest interval, weak label | 1.800 | 23 | 40 ms | 1.0 | 0.847 |
| YouTube train row 7 | active manifest interval, weak label | 1.820 | 23 | 20 ms | 1.0 | 0.859 |

The four ADI source MP3s were 32 kHz. The two YouTube source WAVs were 48 kHz. Every Mimi latent had
the configured continuous dimension 32 at 12.5 frames/second; the encoder internally ran at 200
frames/second. The six encode/decode calls used one CPU thread and took 2.82 seconds combined; the
complete codec script, including input extraction and model loading, took 4.77 seconds.

## Breeze consistency probe

The comparison used the already cached `MediaTek-Research/Breeze-ASR-26` revision
`7b992682e7f5ceedd0a41ebec240f01ba469d19e`, CPU only, with the repository's pinned
`breeze-resample/v2` adapter resampling both 24 kHz files to 16 kHz. It loaded offline and compared
YouTube train row 0:

| Signal | Breeze output |
|---|---|
| Original 24 kHz interval | `之前好像沒有那個東西` |
| Mimi reconstruction, aligned to input length | `之前好像沒有那個東西` |

The pair took 145.6 seconds including cached model loading and two CPU transcriptions. The command
needed `USE_TF=0` because an unrelated system TensorFlow/NumPy binary mismatch otherwise prevented
the Transformers pipeline import. No package or model was downloaded or modified.

## Reproducibility and boundaries

- Artifact root:
  `/mnt/HDD2/proj_tts/pocket-tts/data/taigi-adi-20261026/diagnostics/mimi-reconstruction-v1`
- Machine-readable codec measurements: `metrics.json`
- Machine-readable ASR comparison: `asr_consistency.json`
- Temporary bounded scripts: `run_diagnostic.py`, `run_asr_consistency.py`
- Codec checkpoint: cached Kyutai `english_2026-04_24l` revision
  `492522650173a0653b7575cdc25ae09810e5d741`
- Device: CPU; `CUDA_VISIBLE_DEVICES=''`; Hugging Face and Transformers offline modes enabled.
- No live training process, checkpoint, adapter source file, dataset manifest, or GPU state was
  changed. The generated diagnostics are under the ignored data tree; this report is the only
  repository document added by the probe.

The sample is deliberately too small for a pronunciation-quality conclusion. RMS and finiteness
catch gross numerical, duration, and level failures only. Breeze generated the current weak labels
and therefore cannot independently validate their meaning; matching its own output across the codec
only shows that this one recognizer-visible sequence survived this one reconstruction.
