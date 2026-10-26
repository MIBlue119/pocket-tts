# Pocket TTS 非英語：文字表示、資料與評測證據

產品主目標仍為一般繁體華語輸入→正確、自然台語語音；Mandarin／Japanese 是研究比較，不是新增三語模型交付。私人 Hugging Face、總預算 US$100 與既定 22:00 截止沿用主代理控制；本報告不授權額外訓練支出。

日期：2026-10-26。pstack Investigation／how 的 bounded language/data explorer。檢查工作樹 `.worktrees/taigi-batch-weaklabels`，HEAD `64e7ecd24d7d17790e8221d5bdca6c661c0e6067`。只研究及寫本文件，未啟動模型、GPU、訓練或下載語音。下列「事實」來自直接檢查；「推論／建議」尚未經本專案實驗驗證。模型初始化與最佳化細節由另一 explorer 調查。

## Components Found

| 元件 | 檔案／來源 | 功能及本次發現 |
|---|---|---|
| `LUTConditioner`、tokenizer | `pocket_tts/modules/text_conditioner.py:114` | token ID 查 embedding；不是預訓練中文理解器。詞表大小受 assert 約束；相同大小不表示 ID 語意相同。 |
| tokenizer trainer | `training/scripts/train_tokenizer.py:54` | 從 `transcript` 或純文字訓練 SentencePiece；預設 4,000、identity normalization、byte fallback；實際任務 4,096 是另行設定，並非通用最佳值。 |
| text frontend | `pocket_tts/models/text_chunking.py:15` | 推論前正規化、英文空白詞數 heuristic 與 ASCII 標點切句；中文／日文需驗證。 |
| Taigi source schema | `training/taigi/schema.py:9` | `language=nan-tw` 與 `source_declared` 是宣告，不是音訊語種辨識結果。 |
| diagnostic evaluator | `training/taigi/evaluate.py:58` | 固定 seed 直接 token 化、自由生成、音訊健全性、可選 Breeze；ASR 指標是 SequenceMatcher similarity，並非 CER。 |
| ADI 原始資料 | 主 checkout `data/taigi-adi-20261026/raw/{audit.json,LICENSE}` | 原始八個 tar 已由主流程下載／稽核；本 explorer 讀取 audit 與授權正文，未重新解碼音訊。 |
| 既有產品契約 | `docs/research/2026-10-25-direct-zh-to-taigi-pocket-tts-research.md` | 一般華語語意文字直接生成台語；台文／台羅不是必要中間表示。 |

## Flow

1. **先指定任務**：`input_text` 的語言／書寫法與 `target_audio` 的語言分開。繁體字是文字形式，不是輸出語言的識別器。
2. **來源配對**：選定完整語句的音訊與同範圍文字，保留來源文字、語種宣告、標籤來源及版本。ADI 原生台文不直接覆寫成華語語意。
3. **正規化與 tokenizer**：選一套契約後，只用 train 文本建立或評估詞表；固定 tokenizer hash、normalizer、embedding 對應。dev/test 不參與詞表選樣。
4. **模型條件**：`LUTConditioner.prepare/forward` 將 ID 轉成條件向量；訓練 conditioner 再與音訊序列組裝。台語聲學目標來自錄音，不需要每個音節先轉成台羅。
5. **自由生成與評測**：`evaluate()` 直接調用 tokenizer 和 `model.generate`；公開 `TTSModel` 路徑另經 `prepare_text_prompt/split_into_best_sentences`。兩條路徑都須驗證，不能以短句 evaluator 通過推論長文也正確。

## Non-Obvious Things

### 1. 三種產品是不同的學習任務

| 使用者輸入 → 目標 | 核心挑戰 | 最小可信資料契約 |
|---|---|---|
| 一般繁體華語 → 台語 | 跨語言詞彙／句法選擇，加上台語發音 | 完整華語語意＋相符台語錄音；來源字幕／腳本／弱標註均可，但分證據等級 |
| 繁體華語 → 華語 | 中文閱讀、語音、多音字；若要台灣口音，還要該口音資料 | 華語逐字文字＋華語錄音；字形轉換不會將錄音口音轉成台灣口音 |
| 日文漢字假名 → 日語 | 漢字讀音、數量詞、長短音、促音及音高重音 | 日文文字＋日語錄音；讀音或音素前端可作對照，不是預設必要條件 |

**事實**：ACL 2023 §3.2.3 的 T2UT 明確用 Mandarin characters 預測 Hokkien normalized units；§4.1.4 使用篩選後約 1,000 小時台語戲劇及華語字幕。這支持直接華語文字→台語的建模可行性，但其 codec、預訓練和資料規模與本專案不同，不能推出 18 小時／1,000 updates 足夠。[原論文](https://aclanthology.org/2023.findings-acl.307.pdf)

**推論**：ADI `食薰` 是台語表達；若要一般華語輸入契約，需另有其華語語意標籤，不能把去掉括號的台文假定為華語。可保留 native-reading acoustic adaptation 對照，或為相同錄音產生中文語意候選，兩者應有不同 task／label 欄位。這是資料對照選擇，不是要求所有資料補台羅。

**最新官方狀態**：Pocket 模型卡列英文、法文、德文、葡文、義文、西文；不可概稱目前 Pocket 僅支援英文。這不等於本專案 pinned teacher 已支援三個目標語言，也沒有證據顯示官方已支援台語、華語或日語。[官方模型卡](https://huggingface.co/kyutai/pocket-tts)

### 2. 零 unknown token 不代表理解中文，也不代表新詞表一定更好

**事實**：SentencePiece byte fallback 能把詞表外字元拆成 UTF-8 bytes。[官方說明](https://github.com/google/sentencepiece/blob/master/python/README.md) 本地 trainer 已啟用此功能。因此只報 `<unk>=0` 是不足的診斷。

**推論／建議**：先做 CPU-only tokenizer audit，再選訓練方案。對固定的 train sample 和完全隔離的測試文字，輸出 round-trip、一字平均 token 數、p50/p95/p99 長度、byte-token 比例、超過實際 context 預算比例；分開漢字、台羅、假名、英數、姓名、標點。不要用字元數當 token 預算。台羅需測組合式／預組式 Unicode 音標；不要把所有音標剝除。日文也不能把清濁音正規化掉。

| 對照 | 可回答 | 不能單獨回答 |
|---|---|---|
| 原 tokenizer＋原 text embeddings | 既有表示能否在目標資料上轉移 | byte 序列短或語音正確與否仍需實測 |
| 原 tokenizer＋reset embeddings | embedding 重置本身的學習成本 | 新詞表的優劣 |
| 新 tokenizer＋reset embeddings（目前方向） | 新表示＋重學文字條件整體結果 | 相對原方案的差異不能全歸因於 tokenization |
| 新 tokenizer＋相同 piece 的 embedding 移植 | 保留共享字元／bytes 是否改善 | 只有 piece、special-token、normalizer 語意相容才可移植；不能按同一 index 盲拷貝 |

先用小範圍兩組主比較，只有結果需要歸因時才補 reset 對照。固定資料、seed、有效音訊曝露量和評測；同時報訓練算力，因為 token 長度改變計算量。不要無上限掃詞表。`4,096` 並無「不足以容納全部常用漢字就一定不能做」的結論，bytes 可補覆蓋；也無「有 bytes 就無須重訓表示」的結論。

對 Mandarin，可先比較原字形和固定繁體正規化；多音字讀音標籤是後續受控對照。對 Japanese，可比較 raw kanji/kana 與固定 G2P／kana；pyopenjtalk 官方提供 G2P、kana、使用者字典及 full-context/accent 選項，可做可重現前端，但不保證名字或任意語境讀音正確。[官方 pyopenjtalk](https://github.com/r9y9/pyopenjtalk) 不可把華語→台語換成 Mandarin pinyin，因為那仍未提供台語對應詞彙。

### 3. 可用資料：有下載不等於有目標任務標籤

| 候選 | 直接確認的內容 | 在本任務的用途與界限 |
|---|---|---|
| 已下載 ADI Hokkien | pinned `ffcc36b9afe31ec50f7742c0c4c614f1a5b40151`；train 23,158 explicit nan-tw／16.566h，另 5,530 blank locale＋nan path／4.348h | strict 組先作可測候選；fallback 分開報。2,794 Mandarin 路徑排除出台語實驗。metadata 不能證明每段都是台語。 |
| TAT／TAT-TTS | 官方研究團隊說明原生台文朗讀、台羅等標註；TTS 四位語者各 10h，需申請授權 | 可做同語言閱讀／聲學對照；沒有現成普通華語標籤就不是直接產品 benchmark。六支麥克風同時錄製不可當六倍獨立內容。 [官方來源](https://sites.google.com/speech.ntut.edu.tw/fsw/home/tat-corpus) |
| AISHELL-3 | 官方 85h、218 Mandarin speakers，漢字及拼音標註，Apache 2.0 | 乾淨 Mandarin 轉移基準；不能據此宣稱台灣華語口音。 [OpenSLR](https://openslr.org/93/) |
| Common Voice Chinese (Taiwan) 27.0 | 官方頁面標示台灣華語朗讀、CC0、ASR intended use，另列不重新託管／分享等使用條件 | 台灣華語候選；尚未下載、估時或做 TTS 品質審查，不能直接把整包算成可用 TTS 時數。 [Mozilla](https://mozilladatacollective.com/datasets/cmu5wkw5i00c6o107jnm3t51u) |
| JSUT | 官方 10h、一位日本女性、48kHz；涵蓋日常漢字讀音、數量詞、外來語等 | 適合日語單聲線學習診斷；文本與音訊條款不同，商用需聯繫；不是開放多語者泛化證明。 [官方來源](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) |

ADI 的公開 viewer 也直接顯示 `T?a-t?n-ki?` 等損壞羅馬字。[資料卡](https://huggingface.co/datasets/adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hokkien) 本地 audit 確認原 tar TXT/JSON 和 TSV 都有 literal `0x3f`；不是字型或 UTF-8 decoder 問題，無法靠重新 decode 還原。可選擇去掉確認為讀音註解的括號內容以建立 native-Han 對照，但需保留原文、規則與排除個案；不能刪掉任意括號，也不能把損壞字母補猜成 gold 台羅。

ADI `LICENSE` 正文是 **TRAIL – TWM Responsible AI License 0.1，2025-09-30**，不是 MIT；檔案 SHA256 `172187370351274e50051d8a8f21004ea8240750bbbf7381cdcc30ac9e51072e`。其 TRAIL-D 區段與整份模型／衍生物條件須保留於來源清冊。本文件只確認正文及版本，不裁決混合資料／權重的所有下游相容性。

### 4. 自動評測可以減少人工，但不能把同一 ASR 當真值

**事實**：Breeze-ASR-26 官方說明輸出台語音訊的 Mandarin character 表示；台語與華語非一對一，所以完美台語辨識也不必有 0 CER。它由約 10,000h 合成台語微調，官方亦提醒真實語音域差異。[Breeze 模型卡](https://huggingface.co/MediaTek-Research/Breeze-ASR-26)

**本地已確認**：`evaluate.py` 的 `character_sequence_similarity` 是 `difflib.SequenceMatcher.ratio()`，不可換名字當 CER。它把非英數字元移除後比較，也未區分語意等價的文字差異。現有檔案明確標示 `quality_verified=False` 與 same-model consistency；應保留此限制。

**建議評測面板**：

- 固定 source/speaker/text groups 後再分 train/dev/test。ADI audit 記錄 supplied train/test 共用 **78 client IDs、117 sentence IDs**，雖然 path overlap=0；此 test 不能直接證明未見語者與未見句子泛化。去重時連 raw/enhanced、同片切段、字幕及音訊近重複一起處理。
- 首先校準 judge：對已知 reference audio 原音與 Mimi round-trip 各做同一組測量。若重建已嚴重損失訊息，先調查 codec／前處理；若 judge 對原音已失敗，生成失敗率要拆成 judge 盲點與模型差異。不同音訊不要以 waveform L2 當自然度。
- 同 prompt、同 seed，只改一個數字／否定／實體的 **contrastive prompts**；檢查生成內容是否隨輸入改變。只計音訊 hash 變化不夠，還需內容 recoverability。另設 shuffled-text/null-text 對照，偵測模型是否忽略文字；這些不是自然度評分。
- Mandarin／Japanese：獨立 ASR 對 transcription 的 CER，再分數字、名字、漏詞、重複；日文另保留 kana／音素表示的閱讀錯誤對照。漢字 CER 可掩蓋同漢字錯讀，不能證明 pitch accent 正確。
- 直接中文→台語：Breeze 回譯的 CER／語意相似度只能作一項代理；數量、否定、時間與專名分開檢查，允許合理華語同義表述。不可拿台文逐字稿直接和華語 ASR 當同一正字系統計 CER。
- 輸出是否台語需獨立證據：聲學 language-ID／其他台語辨識器若未在本域校準，只能報 provisional。Mandarin-compatible ASR 輸出不證明音訊說台語；中文文字 classifier 更無法判定。
- EOS hit-limit、時長異常、靜音、clipping、重複、音色一致性、CPU 延遲分開報；ASR 良好不等於音高／聲調／自然度良好。
- 使用者不需逐段聽打。先用既有來源腳本／平行語音建立自動參考；無可靠獨立 reference 的項目明確標為未驗證。不同 ASR 若共享合成資料或相同底座，也不能只靠票數稱為獨立真值。

### 5. ADI split：保留資料，收窄宣稱，不盲目移除大部分 train

主代理補充的 metadata 統計（本 explorer 未重算）：strict train 有 254 client IDs；若為讓 publisher test 全部 speaker-disjoint 而移除其 78 個重疊 IDs，會移除 19,893／23,158 rows（85.9%），只留 3,265。**建議採主代理提出的替代方案**：

1. 從原 train 固定選一組 dev speaker IDs，目標約 10% rows／hours，同時檢查長度、內容與可用標籤分布；不要只取 10% speaker 數假定等於 10% 音訊。
2. 對 train/dev 移除相同 sentence ID、正規化文字及音訊近重複；整個 speaker group 不跨 train/dev。公布實際剩餘樣本／時數，分割設定與 seed 固定於標註、tokenizer 與調參之前。
3. Publisher test 原始音檔不進 train；如 train 有相同音訊或重上傳，亦排除。將 official test 分成 speaker seen/unseen × sentence seen/unseen 四種 bucket，各報樣本數與結果；若保留相同句子於 train，需如實標示，不能拿 overall 分數宣稱未見內容泛化。
4. Primary dev 是模型選擇資料，不是反覆調參後的獨立最終 test；另留固定的 source-disjoint 普通華語語意→台語評測作產品 gate，或將目前結果明確稱為 dev 結果。

**理由／限制**：本產品先要語意正確台語，不是證明零樣本 voice cloning。保留大部分可用語音並設計真正分組的 dev 比為小型既有 test 刪掉 85.9% 更合適。client ID 仍只是 speaker proxy；內容或來源域可能與 speaker 相依，需報分布偏移。ADI 的 native-reading dev 即使成功，也不證明直接普通華語輸入成功，語意 eval 不能被這組取代。這個設計保留主評測的隔離原則，同時縮限 official test 的結論，不能美化成所有測試均完全獨立。

## Boundaries

文字契約連接資料產線與模型；tokenizer＋embedding 連接訓練、checkpoint 和 inference；split manifest 連接資料取得與評測；ASR 只連接候選標註與代理評分，不能取代目標語音語種／自然度真值。推論 frontend 的 ASCII-only 邊界與 whitespace word heuristics 是本次看到的具體相容性風險，應加入中文 `。！？、` 與日文標點的端到端測例，再決定修改；本 explorer 未實作。

## Actionable small experiments（建議，未執行）

1. **表示與資料離線稽核**：先固定普通繁體華語→台語契約、tokenizer 指標、ADI metadata/text/source 重疊、可用 label types；報告保留量，沒有主觀「乾淨 95%」數字。三語表示僅作文字層比較。
2. **codec／judge 基線**：少量合法已有台語 reference 音訊作原音／重建對照；完成後再使用相同 judge 評生成。沒有合格 judge 不把結果稱為通用品質。
3. **同語言與直接任務分開的小樣本學習**：Taigi 保持普通中文語意配對作產品主實驗，ADI native-reading 作明確標記的診斷分支。native-reading 成功只證明部分聲學／閱讀學習，不能宣稱直接語意轉換成功。Mandarin／Japanese 的逐字配對是將來若擴展時的對照設計，不加入當前交付或預算。
4. **有限 tokenizer 對照**：先原 tokenizer＋保留 embeddings vs 新 tokenizer＋reset；相同資料／曝露量。以同內容與未見內容自由生成判別文字依賴及泛化；必要時補原 tokenizer＋reset 分離原因。
5. **決策**：若訓練句也不能依輸入換內容，先查資料配對／loader／text conditioning；若訓練句可而未見句不行，才判為泛化／coverage 問題；若語意可而聲調差，調整發音資料／前端／聲學評測。這是診斷樹，不能從目前「1,000 updates 說不相關內容」直接斷言唯一根因是資料太少或缺台羅。

## Files Read

- `/tmp/pstack-taiwanese-20261025/skills/how/references/explorer-prompt.md`
- `/home/weiren/.codex/skills/principle-prove-it-works/SKILL.md`（成果核對原則）
- 工作樹 `docs/research/2026-10-25-direct-zh-to-taigi-pocket-tts-research.md`
- 工作樹 `docs/research/2026-10-25-adi-hokkien-evidence.json`
- 工作樹 `pocket_tts/modules/text_conditioner.py`
- 工作樹 `training/scripts/train_tokenizer.py`
- 工作樹 `training/taigi/evaluate.py`
- 工作樹 `training/taigi/schema.py`
- 工作樹 `pocket_tts/models/text_chunking.py`
- 工作樹 `training/modules/conditioner.py`（只搜尋 token 組裝相關行）
- 工作樹 `pocket_tts/models/tts_model.py`（只搜尋 frontend 呼叫相關行）
- 主 checkout `/mnt/HDD2/proj_tts/pocket-tts/data/taigi-adi-20261026/raw/audit.json`（重點欄位讀取；含上述數字與已執行稽核結果）
- 主 checkout `/mnt/HDD2/proj_tts/pocket-tts/data/taigi-adi-20261026/raw/LICENSE`（完整讀取）
- 外部 primary sources 均已在各事實旁提供連結；ADI HF license 網頁取用失敗，故授權結論依本地 pinned 原始 LICENSE。TAT 新網址回 404，引用仍可讀的研究團隊原站。

## Open Questions

- 未對本任務 tokenizer 做實際三語 token-length benchmark；不能聲稱某表示已勝出。
- 未獨立聽辨或重新 decode ADI 音訊；raw audit 數字是已在磁碟的證據，非本 explorer 重跑。
- 無已驗證的獨立台語語種／聲調／自然度 judge；Breeze 的訓練來源可能與其他 verifier 相關。
- ADI native-Han 轉普通華語標籤的準確率、短詞資料轉長句能力、以及 multilingual base 對本任務轉移效益都需實驗。
- 何種資料量／更新量能使 pinned Pocket teacher 達到可用品質沒有本次證據；不可保證時數門檻或訓練成功。

`throughput checkpoint: n/a, read-only investigation`
