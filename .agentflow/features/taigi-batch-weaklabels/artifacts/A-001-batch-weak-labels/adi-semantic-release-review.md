# Independent ADI semantic release and adaptation review

Verdict: **PASS for the bounded builder/fork implementation and local CPU integration.** No unresolved code findings in the reviewed scope. The actual cloud release build and its trainer-manifest reads remain the parent agent's execution gate; this review did not start training or approve model quality.

Reviewed commit: `dcff9f6b6eb009ad76b38ad4534ca709bae866e4`. Direct `git show` byte comparisons confirm that all four committed files match the reviewed working files.

| File | SHA-256 |
|---|---|
| `training/scripts/build_adi_semantic_release.py` | `051215328f80e3350e50d12721f8635257133af9f9d38a78a3b38b2bc6c3975b` |
| `training/scripts/fork_taigi_adaptation.py` | `92185bf4727ec74a8690db458d9021d23b514c801d1caf319b229c631e0dec95` |
| `tests/test_adi_semantic_release.py` | `4cd1d5cf345c6ea10e42e6fe4774a4a74f344febe41fa5a620a99d764754c840` |
| `tests/test_taigi_adaptation.py` | `b44be94151328ee28e8413a3d1e06a98d826fe57825f5f44f03abeb374191684` |

## Scope and intended behavior

Reviewed the two scripts, their focused tests, the adaptation design, the semantic-label strategy, prepared-release serialization, trainer manifest/audio readers, checkpoint restore, and launcher identity boundaries. The intended task is Mandarin semantic conditioning for Taigi speech, using exact official MOE parallel cells as weak labels; original Taigi annotations remain separate evidence. The existing Breeze schema keeps its distinct provenance validation.

No implementation changes were made by this reviewer. Review feedback was integrated by the implementation owner before the final freeze.

## Direct verification

- Ran `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /mnt/HDD2/proj_tts/.venvs/taigi-pipeline/bin/python -m pytest -q tests/test_adi_semantic_release.py tests/test_taigi_adaptation.py`: **24 passed in 4.71s** on the final hashes.
- Independently parsed the preserved official ODS `content.xml`, including repeated cells and numeric `office:value`. All **1,803 references across 805 mappings** matched the actual sheet, one-based row, native cell, Mandarin cell, and stored source-row fields; zero mismatches. Also exercised the builder's final ODS validator on all 805 real mappings successfully.
- Built a real-WAV fixture with absolute prepared audio paths, added a distinct recording of an existing Mandarin target, then forked a sentence-mode parent and iterated the actual `training.dataloader.DataLoader` on both copied child manifests. Train audio shape was `[3, 1, 24000]`; valid was `[1, 1, 24000]`. Complete expected text tokens matched. Both distinct same-target recordings were retained.
- Hashed every existing fixture file before the fork and compared after it: parent, prepared input, and release bytes were unchanged. Checkpoint and optimizer copies were byte-identical. The focused tests also check EMA contents, tokenizer/model continuity, the exact config-change allowlist, lineage identity, and sentence-sampling/precompute rejection before child creation.
- Checked published audio files directly: regular internal copies, no symlinks, no write permission bits. Mutated one generated WAV's payload and confirmed that release verification rejected it with an audio-hash mismatch.
- Independently joined the real frozen mapping file to the local cloud-prepared metadata receipt, checking exact sentence IDs and native reference values. Accounting was **1,012 raw clips − 19 absent from frozen prepared candidates − 10 diagnostic-target alias clips − 37 cross-split target clips = 946 retained clips: 867 train / 79 valid**. The retained metadata is disjoint across splits by original client ID, native reference, normalized Mandarin target, and audio SHA-256. This was a metadata audit, not a local read of the cloud audio.

## Findings resolved before final freeze

1. Prepared candidates contain absolute audio paths while the prepared artifact hash table uses relative paths. The builder now converts the candidate path to the correct artifact key.
2. Relative release audio paths failed actual audio reads after manifests were copied into the child. Generated manifests now carry absolute in-release audio paths; the full build/fork/data-loader check passes.
3. Self-consistent embedded dictionary references previously passed without agreement with the original ODS. The builder now parses the preserved source and checks the cited cells; an out-of-range positive row is rejected.
4. Nine native mappings shared Mandarin targets with diagnostic controls, affecting ten prepared clips. Those clips are now explicitly excluded from both train and valid, and their count is recorded.

The implementation intentionally treats raw mappings missing from the already filtered prepared release as counted exclusions. It does not restore those recordings or change prepared split assignments. The design's missing-source wording was reported to the implementation owner for alignment with this approved behavior.

## Input identities

- Official `kautian.ods`: `4a73eb842e6206d7bb21f2b2e16f5ed70cebe075b8c54b9d448e95d7f1ebe7ed`.
- `weak-admitted-mappings.jsonl`: `5597013c13cb6513df501265f8e8f49916ccf7ae87f9f180e0ff63a3a7c3ef61`.
- `diagnostic-translations.jsonl`: `3570cc1e6202c5364d15ef5ba27077eac9813f43724a150318606f28c8c1e40d`.
- Prepared receipt `ef8b74e578f2d45d1fb2/report.json`: `46d632e556cb5cdcf6d7666102bcd14eaef075aa4e6e8c1ca259e6d5d31c663a`.

## Limits

No SSH, GPU work, model inference, cloud service changes, or full real-audio cloud release build were performed by this reviewer. Tiny real Torch checkpoint fixtures establish copying and continuity behavior, not production checkpoint fitness. Client IDs are speaker proxies; diagnostic cases are not automatically speaker-held-out. Exact dictionary evidence is not human or acoustic semantic verification. CC BY-ND 3.0 TW evidence remains preserved; no unrestricted-publication or downstream license conclusion is made. This review supports preparing the bounded weak-label experiment, not a claim that open-domain Mandarin-to-Taigi synthesis works.
