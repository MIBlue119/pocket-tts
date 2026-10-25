# Taiwanese ASR Evaluate

## 目標
- 提供未來中文/台語語音混合語音辨識引擎評估資料

## 評估資料集準備
a. 尋找具備中文/台語語音交雜資料

b. 音訊前處理去掉背景噪音/配樂/音效/回響等干擾
- [Deepfilternet](https://github.com/Rikorose/DeepFilterNet): 去除背景噪音
- [Resemble-Enhance](https://github.com/resemble-ai/resemble-enhance)

c. 先經由一定程度支援台語/中文之語音辨識引擎生成.srt檔

d. 尋找聽得懂台語的人員，邊聽音檔編修正逐字稿內容

## 資料集可能來源

- [台視台語大頭條](https://www.youtube.com/playlist?list=PLCcVwtUesa9LX7x5ng0-VDzjAbQwH43dA): 主播以台語/記者會用中文解說，中間會混有混雜音效/採訪噪音，可能有其他語言

- [公視台語台-台灣新視界](https://www.youtube.com/@ptstaigitai-tnv): 主播以台語解說/被採訪者可能會說中文，影片有內嵌字幕可以對照

- [台語珍輪轉](https://www.youtube.com/playlist?list=PLzlt9nxVmI3t9R7U3ZKE4nvlmBDJ88IXq): 兩個主持人對話教台語，偶爾穿插中文單字。錄音環境有echo/音效

- [台南妹仔教你講台語](https://www.youtube.com/watch?v=F4J5hCDbEYA): 一個台語教學Youtuber

## [TODO] 評估指標
- CER
