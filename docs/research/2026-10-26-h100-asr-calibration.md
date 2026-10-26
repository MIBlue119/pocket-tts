---
title: H100 source-ASR calibration for the bounded Taigi semantic control
date: 2026-10-26
status: measured
---

# H100 source-ASR calibration for the bounded Taigi semantic control

## Result

One offline, cached Breeze-ASR-26 GPU pass transcribed the 31 predeclared source recordings: eight
seen mappings, 16 same-speaker reserved controls, and seven prepared speaker-held-out controls.
Every category had a distinct raw and normalized ASR output for every source (8/8, 16/16, 7/7).
Therefore the later generation gate has enough non-tied source-ASR references to be discriminable;
this calibration does **not** make those references correct or independently verified.

Mean normalized Mandarin-target `SequenceMatcher` similarity was 0.453 for seen sources, 0.434 for
same-speaker reserved sources, and 0.517 for held-out sources. Across all 31, the mean was 0.458;
15 were at least 0.5 and 12 were zero. These are judge-behavior measurements, not CER, translation
accuracy, pronunciation, or perceptual quality.

The recovered fixed-seed punctuation diagnostic also produced sharply different outputs:

| Input at step 1,000 | Audio SHA-256 | Breeze output |
|---|---|---|
| `你好嗎` | `6daea68855480a1abb1aa244f75ebea11bb1421ba6598a8a6751e23de8695df6` | `阿彌陀佛` followed by repeated `唵` |
| `你好嗎？` | `1acdc2a55bbd4aad24a5959d089345af3f7a4902c1cd9d47c48802e958d5961a` | `你這條白鞋 配這條黑鞋好嗎` |

Both files were 5.04-second, 24 kHz, mono float WAVs. The original evaluation record fixes the same
checkpoint, seed 17, temperature 0.3, CFG 1, zero voice prompt and 63 generated frames; the terminal
full-width question mark is the recorded input difference. Thus punctuation changed this model's
fixed-seed generation path. It does not show that the second waveform is correct Taigi, natural, or
the same underlying utterance with punctuation added.

## Source calibration rows

The dictionary target/reference columns below are immutable proof carried from the existing mapping
files. Breeze output was not used to relabel them. Similarity is computed after removing punctuation
and whitespace and lowercasing Latin text.

| Slice | Mandarin target | Native reference | Source Breeze output | Similarity |
|---|---|---|---|---:|
| seen | 一百塊錢 | 一百箍 | 一百塊 | 0.857 |
| seen | 手剎車 | 手禁 | 手帕 | 0.400 |
| seen | 跌倒 | 跋倒 | 跌倒 | 1.000 |
| seen | 一串香蕉 | 一枇弓蕉 | 一筆香蕉 | 0.750 |
| seen | 叫車過來 | 喊車 | 禪車 | 0.333 |
| seen | 公雞 | 雞公 | 整天 | 0.000 |
| seen | 腋下 | 胳下胴 | 國的人 | 0.000 |
| seen | 眼睛看不清 | 目睭花 | 眼花 | 0.286 |
| same-speaker reserved | 飛機 | 飛行機 | 飛行機 | 0.800 |
| same-speaker reserved | 海豚 | 海豬仔 | 還讀啊 | 0.000 |
| same-speaker reserved | 男孩子 | 查埔囡仔 | 男孩 | 0.800 |
| same-speaker reserved | 廁所 | 便所 | 扁壽 | 0.000 |
| same-speaker reserved | 有膽就不要逃跑！ | 好膽莫走！ | 有膽別跑 | 0.545 |
| same-speaker reserved | 不好意思 | 歹勢 | 抱歉 | 0.000 |
| same-speaker reserved | 頭髮 | 頭毛 | 頭髮 | 1.000 |
| same-speaker reserved | 茶葉 | 茶米 | 茶葉 | 1.000 |
| same-speaker reserved | 蝴蝶 | 蝶仔 | 媽 | 0.000 |
| same-speaker reserved | 牙籤 | 齒戳仔 | 蓋塗鴉 | 0.000 |
| same-speaker reserved | 小孩子 | 囡仔 | 孩子 | 0.800 |
| same-speaker reserved | 屁股 | 尻川 | 屁股 | 1.000 |
| same-speaker reserved | 你不要再哭了！ | 你莫閣哭矣！ | 你不要再哭了 | 1.000 |
| same-speaker reserved | 一起 | 鬥陣 | 相處 | 0.000 |
| same-speaker reserved | 鯨魚 | 海翁 | 海洋 | 0.000 |
| same-speaker reserved | 菠菜 | 菠薐仔 | 白鷺鷥 | 0.000 |
| speaker-held-out | 剁雞肉 | 鏨雞肉 | 斬雞肉 | 0.667 |
| speaker-held-out | 你說什麼？ | 你講啥？ | 喂 你說什麼 | 0.889 |
| speaker-held-out | 肥皂 | 油胰 | 友誼 | 0.000 |
| speaker-held-out | 有錢人 | 好額人 | 有錢人 | 1.000 |
| speaker-held-out | 你運氣真好耶！ | 你真好運呢！ | 你真幸運 | 0.400 |
| speaker-held-out | 下午 | 下晝 | 下午 下午 | 0.667 |
| speaker-held-out | 吃午餐 | 食晝 | 家道 | 0.000 |

Notably, a unique source-ASR string is not necessarily a useful Mandarin semantic reference:
examples such as `整天`, `國的人`, `媽`, and `白鷺鷥` are distinct but poorly matched to their
dictionary targets. Later generation evaluation must preserve the raw source strings and dictionary
targets as separate axes instead of silently treating Breeze output as truth.

## Selection proof and execution

The selection joined each dictionary mapping to a prepared candidate by **both** full original clip
path and sentence ID. It then required the exact top-speaker group for the 8+16 rows and prepared
`valid` plus a different speaker group for the seven held-out rows. Each cloud audio file was checked
against the SHA-256 recorded in the preparation receipt before inference. All requested rows resolved
uniquely; none were substituted or omitted.

Artifacts are under
`/mnt/HDD2/proj_tts/pocket-tts/data/taigi-adi-20261026/diagnostics/h100-source-asr-v1` locally and
on the H100 VM:

- `selection.jsonl`: 31 immutable dictionary/source joins with proof and audio hashes.
- `asr-calibration.json`: raw/normalized ASR, per-row similarities and category summaries.
- `build_selection.py` and `run_gpu_asr.py`: bounded reproducibility scripts.
- `prompt-00.wav` and `prompt-01.wav`: remote copies with verified source hashes.

The single `batch_size=4` pass used cached `MediaTek-Research/Breeze-ASR-26` revision
`7b992682e7f5ceedd0a41ebec240f01ba469d19e`, the repository-equivalent 16 kHz polyphase adapter,
offline mode, and H100 `cuda:0`. Payload time was 58.05 seconds; the enclosing systemd unit completed
successfully in 61.4 seconds under `RuntimeMaxSec=360` and `KillMode=control-group`. A post-run
`nvidia-smi --query-compute-apps` was empty before GPU handoff.

No mapping, dataset, training configuration, model, running service, budget setting, token, or VM
state was changed. The pre-existing CPU ASR service was not inspected, stopped, or mutated. No Git
commit was created.
