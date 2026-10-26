# ADI native annotation → Mandarin semantic conditioning: bounded plan and artifacts

Date: 2026-10-26. Text-only investigation; no GPU work, audio listening, source-code change, or commit. This is a weak-label pilot plan, not evidence that ordinary Mandarin→Taigi TTS works.

## Concrete result

The official MOE dictionary has actual parallel columns. Use exact native-label matches against those columns before attempting ASR-derived translation. Do not feed whole dictionary definitions to a TTS utterance as if they were its translation. The checked-in ADI sentence is Taigi orthography, not automatically Mandarin text.

Downloaded the current [official dictionary ODS](https://sutian.moe.edu.tw/media/senn/ods/kautian.ods), linked from the [official resource page](https://sutian.moe.edu.tw/zh-hant/siongkuantsuguan/). File size 4,473,679 bytes; SHA-256 `4a73eb842e6206d7bb21f2b2e16f5ed70cebe075b8c54b9d448e95d7f1ebe7ed`. Its `詞彙比較` sheet contains `漢字` and `華語詞目`; its `例句` sheet contains `漢字`, `羅馬字`, and `華語`. The download contains 12,271 comparison rows and 17,907 example rows. Some example Mandarin fields are explanatory glosses, and some short comparison targets omit meaning or are dialect-dependent. A direct cell match is evidence, not a correctness guarantee.

The [official copyright statement](https://sutian.moe.edu.tw/zh-hant/piantsip/pankhuan-singbing/) declares text/audio CC BY-ND 3.0 Taiwan. Preserve the original unchanged ODS, attribution, and source references for this private internal research; these notes do not resolve downstream model/dataset licensing. Do not publish the modified mapping as an unrestricted dictionary. Prefer this official download over historical third-party Mandarin cross-reference scrapes whose provenance differs.

## Measured ADI coverage

Input revision: `ffcc36b9afe31ec50f7742c0c4c614f1a5b40151`. Parsed actual-tab TSV (the header contains tab bytes; if future exports use literal backslash-t, detect that explicitly). Native key removes only a final full-width parenthesized annotation and surrounding whitespace. No fuzzy match, simplified/traditional conversion, character repair, or pronunciation reconstruction.

| Measurement | Unique native phrases | Strict train clips |
|---|---:|---:|
| All explicit `nan-tw` train data | 18,153 | 23,158 |
| Native text itself contains literal `?` | 624 | 859 |
| Exact comparison-sheet match | 574 | 719 |
| Exact example-sheet match | 690 | 866 |
| Union of exact parallel matches | 1,187 | 1,494 |
| Final mechanically admitted weak subset, after diagnostic holdout | 805 | 1,012 |

There are 19,246 distinct original full annotation strings before removing the trailing romanization. This is not an 18k-phrase task that can safely be solved by guessing from a few ASR examples. Romanization corruption and native-text corruption are separate: native `?` rows must be quarantined even if audio appears usable.

## Artifacts ready for the data pipeline

All data artifacts are under `/mnt/HDD2/proj_tts/pocket-tts/data/taigi-adi-20261026/semantic-labels/` (ignored data, outside the worktree):

- `kautian.ods`: unchanged official source.
- `moe-parsed.json`: local parsed sheets. Numeric ODS cell values read from `office:value` when text is empty; repeated columns handled.
- `unique-native-inventory.jsonl`: all 18,153 native keys with counts and coverage/quarantine status.
- `all-exact-parallel-mappings.jsonl`: all 1,187 matched native phrases, candidate targets, exact sheet/ODS row/full source record, license and SHA, ADI clip paths/sentence IDs/client IDs, and reasons for quarantine.
- `weak-admitted-mappings.jsonl`: 805 phrases / 1,012 clips after mechanical filters; each remains `dictionary_parallel_unverified`, `human_verified=false`, `audio_verified=false`. This is a bounded weak training candidate set, not gold.
- `diagnostic-translations.jsonl`: 64 machine-screened semantic controls spanning 80 ADI clips; 63 official-supported mappings and one explicitly ambiguous machine-only `食飽未 → 吃飽了嗎` control. All carry `machine_translation_unverified`; no acoustic truth claim. The 63 dictionary-matched control native keys are excluded from the admitted training file.
- `summary.json`: exact counts, exclusions, greeting coverage and provenance.
- `build_diagnostic_artifacts.py`: reproduces mapping/filter artifacts from the preserved parsed source and exact-candidate cache; no model invocation, training, or audio processing.

### Admission boundary

Automatic quarantine reasons include: more than one distinct Mandarin target across either sheet (including unnormalized orthographic variants); corrupt native; one-character native without context; multiple senses/readings in exact dictionary headword matches; target too long relative to native; explanatory/alternative markers; overlap with strict official test native keys; and diagnostic-control native keys. Fourteen additional suspect short mappings were quarantined after machine spot-checking (e.g. a food name losing its food suffix, a disease mapped to a broader disease category, or a face-washing phrase mapped to hand/foot washing).

These filters can miss short definitions, archaic Mandarin, regional senses, and annotation/audio mismatches. They do not justify calling the 805 phrases unambiguous in an absolute linguistic sense. The dictionary supplies independent semantic evidence; Breeze ASR must remain a diagnostic view and must not overwrite native annotation meaning. The observed `銅錢仔 → 送錢` and `斷糧 → 大娘` ASR candidates conflict with the native labels and are not admitted by this lexicon route. Four prior observations are not a population error estimate.

For `食薰`, [MOE entry 5686](https://sutian.moe.edu.tw/zh-hant/su/5686/) directly supports the short smoking equivalent. The control uses `抽菸` with its machine normalization explicitly recorded. For ambiguous `穩當`, multiple official senses prevent blind single-target admission. For `夭壽喔`, context-dependent exclamation meaning cannot be replaced by the dictionary's literal early-death definition; quarantine unless a separate exact utterance translation supplies the intended sense.

## Today-sized implementation

1. Join `weak-admitted-mappings.jsonl` to prepared strict `nan-tw` source rows by normalized original clip path plus sentence ID; preserve original annotation and original split. Require exactly one matching native key and one Mandarin candidate. Never train from a join based only on row order or shortened clip basename.
2. Keep this as an explicitly separate weak-semantic dataset/run. The original Mandarin target must be the unchanged selected parallel cell; no raw definition prose. Log source SHA/table/row and `dictionary_parallel_unverified` per example. Complete audio-integrity checks already required by the existing pipeline.
3. Reserve all 64 control native keys and all their Mandarin aliases. The artifact mechanically excludes identical native keys from the weak train subset; **speaker disjointness and Mandarin-target alias disjointness are not yet implemented by the artifact**. Use the stored client IDs to build a separate speaker-held-out slice, report its size, and report whether it still has useful coverage. Do not silently claim speaker independence: official ADI train/test already overlap speakers and sentence IDs.
4. Run a small same-settings weak-semantic pilot against the current baseline. Evaluate the same Mandarin inputs and corresponding native expected content, including controls omitted from training. ASR consistency alone cannot establish intelligibility/meaning. Show separate memorized-phrase, held-out-phrase, and held-out-speaker outcomes; do not merge them into one success claim.
5. Native-Han→Taigi training can be an auxiliary pronunciation/acoustic adaptation stage using intact native annotations and explicit task conditioning or a separate checkpoint. It does not satisfy Mandarin→Taigi semantics, and mixing untagged native-Han and Mandarin labels can create input ambiguity. Whether this stage improves the semantic target needs an ablation against direct semantic fine-tuning; do not assume benefit from dataset size.
6. Leave unmatched phrases as unmatched today. Translate each remaining unique intact native phrase once only when an actually available text model/API is confirmed. Supply native text, original romanization as corrupt evidence, dictionary sense candidates, and a request for a short natural Traditional Mandarin utterance plus ambiguity flags. Independent translation/disagreement screening remains weak evidence. Do not invent a paid API, repair missing native characters by guessing, or use model confidence as accuracy.

## Greeting coverage limits

Strict train has one clip each for `你好`, `食飽未`, `食飽矣`, `食飽啊`, `最近好無`, `按呢好無`, `歹勢`, `多謝你`, and `多謝你！`. There is no exact `你好嗎` or `早安` in the strict native inventory. Other substring hits include longer thanks/support statements, a request containing `好無`, and proverbs containing `食飽`; those are not interchangeable greeting examples. `多謝你！→謝謝你！` is exact official parallel evidence; `食飽未` can be a literal eating question or a social greeting, and the latter does not license substituting any arbitrary Mandarin greeting.

This small phrase/word-heavy coverage is sufficient for a controlled diagnostic, not proof of open-domain Mandarin sentence translation. The existing short YouTube segments do not remove that semantic coverage gap.

## Verification performed

Read official ODS and inspected its schema, counted every strict TSV row, read all emitted JSONL records, asserted all 64 controls exist in ADI, asserted exact parallel control targets are unique except explicitly handled spelling normalization, spot-checked weak admitted pairs, and verified native control/test exclusions in the emitted training file. No audio or acoustic verification was performed.
