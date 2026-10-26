# 非英語 Pocket TTS 研究結論與台語下一步

我們目前的交付目標是「一般繁體華語語意輸入，產生意思正確、自然的台語」，沿用私人 Hugging Face、總預算 US$150、今天 22:00 的限制。華語與日語只供配方比較，不增加訓練任務，也不要求使用者補台羅或逐段聽打。

目前選用英文 24 層 teacher、重訓文字詞表並重置 embedding，是有官方依據的起點。Kyutai 的 [finetune_language.yaml](https://github.com/kyutai-labs/pocket-tts/blob/main/training/configs/finetune_language.yaml) 就採這條路。官方捷克語實驗用 976 小時資料，不能據此保證約 18 小時台語弱標註也能快速成功。新詞表尚未學會文字條件、配對品質、說話人混合與訓練量都仍是可能原因；現有證據不能判定 tokenizer 或 optimizer 的最佳方案。

step 1000 的 validation loss 從約 0.528 降到 0.490，但固定 12 句生成仍重複、偏離輸入，Breeze 字串相似度平均僅 0.0301，且比 step 200 差。訓練時模型看到正確的前段音訊，生成時只能接續自己剛產生的聲音，因此 loss 下降不代表會照文字說話。這個相似度是 SequenceMatcher，並非 CER；同一個 Breeze 同時產生弱標籤與評分，也不能獨立證明台語品質。

主流程正在本地 3090 跑至 step 3000，本報告不修改它。batch 8 的 3000 updates 約等於 1.06 次資料量，並非每筆都讀過的保證。先比較相同句子、seed、生成設定的結果，再決定是否值得繼續；不能只因 loss 改善就加長訓練或開始蒸餾。

直接華語文字到台語有研究先例：[ACL 2023](https://aclanthology.org/2023.findings-acl.307/) 用華語字幕監督台語語音單元，但資料約千小時、架構也不同。ADI 有漢字標註，但主代理實查首個 tar 的 JSON，看到「食薰」「人懸漢大」等原生台文，沒有獨立華語翻譯欄位，不能視為已備妥產品所需的中文語意配對。依使用者最新指示，需另行處理華語語意標籤。保留原文，以 Breeze 產生語意弱標註候選並記錄來源，不確定項隔離；不能宣稱已驗證翻譯。原生台文仍可作聲學及文字條件診斷，其成功不等於產品語意轉換成功。

建議先用既有參考音訊校準 Mimi 重建及評分器，再用 32–64 筆既有、可追溯配對檢查「連訓練句都能否照文字說」。如只有原生台文配對，就明確稱為輔助診斷，不阻擋產品主線，也不要求新增人工標註。若連訓練句都失敗，查文字條件、配對與最佳化；若訓練句成功而新句失敗，再處理泛化與資料覆蓋。今晚能交付哪些品質，必須依實際自由生成結果決定，研究不能保證準時練成可用模型。

| 優先順序 | 具體工作與判斷 | 狀態／邊界 |
|---|---|---|
| 1 | step 3000 完成後，重跑原 12 句並比較 200／1000／3000；分報重複、終止、內容與台語語種證據。 | 訓練由主流程執行；此處評測建議未執行。 |
| 2 | ADI 原生台文與另產生的華語語意弱標註分欄。保留大部分 strict train，另切 speaker-disjoint dev；清除跨 train/dev 重複句與音訊。官方 test 不進訓練，依語者／句子是否見過分桶報告。 | 八個 tar 已下載；adapter 由另一代理處理。不要為隔離官方 test 語者而刪掉 85.9% train。語意標註完成度另報。 |
| 3 | 用既有原音與 Mimi 重建檢查評分器；加入數字、否定及專名對照，分開語意與台語語種。 | 建議未執行；無可靠獨立判讀時標示未驗證，不把回譯中文當作台語證明。 |
| 4 | 若剩餘時間及預算容許，先以目前設定做 32–64 筆小樣本診斷，固定生成檢查點。失敗才做一組最佳化配方對照。 | 建議未執行；不搶占現跑資源。比較時固定資料與曝露量，另報算力，不能只比 updates。 |
| 5 | 交付時封存 checkpoint、tokenizer、normalizer、資料／split manifest、固定樣本及品質限制；另測公開推論入口的中文長句與標點。 | 建議未執行；teacher 通過語意與自由生成檢查後才考慮六層蒸餾。 |

比較研究提供以下界限，沒有新增實驗要求。

| 來源 | 可採用的證據 | 不能推出的結論 |
|---|---|---|
| [Kyutai 官方配方](../../training/configs/finetune_language.yaml) | 英文 24 層 warm start、全 FlowLM 更新、文字 embedding reset；參考 LR `2e-4`、batch 64、FBM 4。 | 這些參數在本專案小型跨語言弱標註資料上最佳。現跑是 LR `3e-5`、batch 8、FBM 1，尚無匹配對照。 |
| [日語社群模型](https://huggingface.co/takuma104/pocket-tts-ja)及[公開配方](https://github.com/takuma104/pocket-tts-ja) | 作者公開新 4k 詞表、reset、英文 teacher、長期訓練及後續蒸餾；另有 decoder 適應。 | 官方已支援日語，或台語今晚可複製其品質；不能未查重建就改 Mimi。 |
| [華語社群模型卡](https://huggingface.co/hustfyb/PocketTTS-ZH-6L) | 作者描述保留英文詞表並擴充到 9,174 tokens，搭配數字等正規化。 | 其完整訓練流程已可重現，或已證明繁體台灣華語品質。 |
| [官方模型卡](https://huggingface.co/kyutai/pocket-tts)與當前語言 config | 新增歐洲語言是各自的 checkpoint。 | 存在一個已驗證適合台語的多語 teacher；換較新 base 必然更好。 |

證據檢查點為 2026-10-26，工作樹 HEAD `64e7ecd24d7d17790e8221d5bdca6c661c0e6067`。本文件是 pstack `how` explainer 的整合產物，pstack pin `18b6c36e6e201b110acc62472de987416ac4e23b`，已讀 explainer prompt 與 unslop 規則。研究由 Sol 彙整[架構／訓練配方](2026-10-26-nonenglish-pocket-tts-recipe-evidence.md)，Astra 彙整[文字／資料／評測](2026-10-26-nonenglish-language-evidence.md)。本輪直接複核了官方 config、[embedding reset 與凍結 Mimi](../../training/modules/builders.py)、[SequenceMatcher 評測](../../training/taigi/evaluate.py)、[推論文字處理](../../pocket_tts/models/text_chunking.py)與[連續 latent 的 DummyQuantizer](../../pocket_tts/modules/dummy_quantizer.py)。Mimi 在此不使用離散 codebook，替換元件須驗證 latent 相容性。

訓練數值與資料規模沿用 explorer 的已存證據；3000 正在訓練與 ADI adapter 分工是主代理交接狀態，不代表本 explainer 重新監控或驗證完成。本輪只新增此報告，未改程式、設定、資料、訓練或資源，也未上傳 Hugging Face。表內後續評測、小樣本實驗、對照及交付檢查均為未執行建議。
