# ADI source review — 2026-10-26

**PASS for the frozen adapter's CPU preparation and ASR candidate generation. No open release-blocking source finding. This is not a training-data or linguistic-quality PASS.** This is a bounded independent source/test review. Root still verifies the actual cloud release and admits labels through separate quality checks before training.

Reviewer: delegated Astra review agent. No source, test, Git, cloud-resource, or GPU changes made by this reviewer. The only repository write is this receipt. The implementation agent completed the fixes described below before the final independent tests.

## Reviewed identity

Worktree: `/mnt/HDD2/proj_tts/pocket-tts/.worktrees/taigi-batch-weaklabels`.

Git HEAD: `43a2185eacc3594de995b0131c20a87c4c88777e`. The reviewed adapter and tests are untracked working-tree files; HEAD alone does not identify them. Their hashes were checked again after the tests:

| File | SHA-256 |
| --- | --- |
| `training/scripts/prepare_adi_taigi.py` | `2215ec88314cdf5e001bfd73aede715b248187930cbcf34b4574f9e99867a842` |
| `tests/test_taigi_adi_prepare.py` | `e4a06efeea2360422c7841930404cb5601af87433e0c75f1ade74d89aeb1eac1` |

Supporting reads covered `training/taigi/{corpus_worker,pipeline,transcriber,merge}.py`, the release lock/writer helpers, and existing corpus-worker tests. This receipt does not review the separate adaptation/fork implementation or the whole branch.

## Actual verification

Independent focused test command, run only after the implementation agent's freeze signal:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_taigi_adi_prepare.py tests/test_taigi_corpus_worker.py -q
```

Result: **28 passed in 15.59s**, exit 0. The first attempt without the environment variable stopped before test collection because the installed `pytest-xdist` plugin could not import `execnet`. Disabling unrelated plugin autoload required no dependency changes; these tests use built-in pytest fixtures and ran serially.

The tests exercise native reference preservation, corrupted romanization preservation, strict/conflicting/inferred locale handling, unsafe archive rejection, duration rejection, official-test audio exclusion, speaker isolation, cross-dev/train native-text exclusion, within-train repeated text on distinct recordings, release corruption rejection, and deterministic one-versus-four-worker selection/split/PCM identities. The real cached-transcriber wrapper with a mocked Breeze backend exercises batch prefill, resume without additional model loads, weak/native label separation, and rejection of a tampered completed training manifest. Existing worker tests also cover interrupted batches, cache corruption, candidate hash rejection, ordering, and resumed cache reuse.

Separate read-only checks against actual local raw data at `/mnt/HDD2/proj_tts/pocket-tts/data/taigi-adi-20261026/raw` passed:

- Inventory identifies revision `ffcc36b9afe31ec50f7742c0c4c614f1a5b40151`, seven train tars and one test tar. All eight actual archives match the inventory's sizes and LFS SHA-256 values.
- All **31,482 train rows and 323 test rows** agree between tar JSON metadata and literal-tab TSV on normalized path, exact native sentence, case-folded locale, exact client ID, and exact sentence ID. Zero extra or missing identities. No audio decode or full preparation was performed for this audit.
- Actual `LICENSE`, `train.tsv`, and `test.tsv` sizes and Git blob IDs match the pinned inventory. Their SHA-256 values are respectively `172187370351274e50051d8a8f21004ea8240750bbbf7381cdcc30ac9e51072e`, `2e2771a5a9e87c8bc731e9fb7c83b20c0f947820570eadeeece092afed5739cf`, and `4c68aee6dc25da7acdd5c373e04c4d8cbec1eb815b1282e82946d6b43f6a07b0`.

## Findings resolved before the freeze

1. **Resume integrity:** the initial ADI ASR wrapper could reuse an existing weak-label release after checking only its audio, leaving changed label manifests undetected. The frozen version records and checks all five output manifest/report hashes, freezes dependency code identities, and validates existing release metadata, compact manifests, and audio before ASR. The independent test run includes a regression that alters `train.jsonl` and verifies refusal before another model load.
2. **Identity reconciliation:** the initial TSV/archive comparison omitted speaker and sentence IDs, although splitting depends on both. The frozen comparison includes them. The separate audit confirms the actual pinned source has no identity mismatch.
3. **Known excessive text deduplication:** the initial implementation dropped repeated text within training. The frozen code only applies native sentence/text exclusion from development to training, while globally deduplicating actual audio with development precedence. The regression retains two distinct training recordings with identical native text.

## Requirement assessment and limits

- The eligible source cohort requires explicit `nan-tw` and a matching nan path. Blank locale stays quarantined; Mandarin/conflicting paths are excluded. No inference from Han text makes a row eligible.
- Native JSON/TXT bytes and their hashes remain provenance. The optional trailing romanization display split preserves the original corrupted annotation; it does not guess lost characters or manufacture a Mandarin target.
- Archive members are inspected without filesystem extraction; unsafe/duplicate members and missing pairs fail. Text and audio reads are bounded. Audio must decode to nonempty finite data, satisfy the original 0.8–20 s duration, and remain finite after whole-clip mono 24 kHz conversion.
- Official-test paths and both compressed/decoded audio hashes exclude matching training recordings. Test speakers and sentences may remain in training under the approved revised policy, with explicit seen/unseen diagnostic buckets. The rejected global official-test-speaker exclusion is absent.
- Speaker groups are deterministically assigned to development targeting 10% of pre-dedup eligible duration. Development sentence/text/audio takes precedence over training. **The final retained development fraction can change after training overlap exclusions**; the actual release report's hours and counts must be used, rather than claiming an exact final 90/10 ratio.
- Four threads write distinct source-derived audio names; `pool.map` preserves source ordering for downstream split and filtering. The one/four-worker test compares content selection, split, and PCM hashes. Elapsed-time/progress/report fields are intentionally not byte-identical across runs.
- Prepared releases have content-derived identities, artifact hash verification, and atomic publication. CPU preparation itself is not resumable after an interrupted `.partial` directory; ASR caches are resumable. No full CPU preparation was duplicated locally.
- Breeze remains pinned to its model revision and `breeze-resample/v2` adapter; input is actually resampled to 16 kHz, and cache keys include audio/interval/model/adapter/batch/code identity. Batch four uses one resident backend. Outputs remain `asr_pseudo`, `unverified_weak_label`, with `quality_verified: false`; automatic nonempty/repetition screening is not semantic truth or spoken-language verification.
- The exact TRAIL license is copied into the prepared release and its hash is propagated in provenance. Distribution packaging and legal conclusions are outside this code review.
- **Absolute paths are a deployment constraint:** prepared candidates and published weak-label manifests contain absolute audio paths. Root must prepare at the final cloud path or preserve identical paths when transferring releases. `verify_release` is not a manifest relocation tool. This reviewer did not inspect cloud-produced manifests because full preparation was not complete at review time.
- The actual whole-corpus decoded count/hours, final train/dev overlap statistics, cloud codec/runtime behavior, GPU ASR throughput, tokenizer parity, independent label quality, and exact overlap with YouTube remain runtime/downstream checks. Unit audio fixtures are synthetic WAV payloads, so they do not independently establish MP3 decoder behavior on the H100 host.

## Material quality gap from the cloud preflight

At review completion, root reported an actual four-clip H100 Breeze preflight: `食薰 → 抽菸`, `銅錢仔 → 送錢`, `斷糧 → 大娘`, and `夭壽喔 → 要命喔`. This reviewer did not run or independently inspect that GPU preflight. Root identifies the middle two as incorrect semantic labels, while the other two are good or plausible. These observations are enough to reject automatic admission of every nonempty, nonrepetitive output as training data; they are not a statistically estimated error rate.

The current adapter accurately marks its outputs as unverified candidates, but does not provide the independent semantic cross-check needed to admit them. Root owns the next automated native-reference/Mandarin semantic validation stage, retaining uncertain pairs in quarantine. No request for owner listening or manual labeling is introduced by this review.

This PASS applies to CPU preparation and ASR candidate generation only. It does not authorize treating those candidates as verified training pairs or claim that training quality has been established.
