# AI-CUP-2022-Spring-NLP
## 運行環境  
### 使用 Anaconda 建立環境 :   
conda env create -f enviroment.yml  
conda activate nlp-aicup  

## 資料  
data/raw : 裡面是未經過處理的原始資料集  
data/Stage 1 : 經過Stage 1 Stage1_BM25Final.ipynb 及 Stage1_Un_BM25.ipynb處理過的資料集  
data/Stage 2 : 經過 SentenceRetrievalTrain.ipynb 運算後產出的數據集  
data/Stage2ModelTrainData : Stage 2 的訓練集  
data/Stage3ModelTrainData : Stage 3 的訓練集  
還有一項資料wiki pages因檔案太大，所以沒傳上來，解壓後麻煩放在這個資料夾
## BM25
Stage1_Un_BM25.ipnb 處理訓練資料
Stage1_BM25Final.ipynb 處理Test及Private的資料
## 訓練 
Stage 2 : SentenceRetrievalTrain.ipynb
Stage 3 : ClaimVerTrain.ipynb
## 預測  
### [權重下載網址](https://drive.google.com/drive/folders/1ejU6aEcdF7dcGH85tKRLN4wNgHPahtS0?usp=sharing)  
請先下載權重並且放在weights資料夾中    
Stage 2 : SentenceRetrievalInfer.ipynb  
Stage 3 : ClaimVerInfer.ipynb  
