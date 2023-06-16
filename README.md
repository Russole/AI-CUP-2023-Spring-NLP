# AICUP-2023-Spring-NLP 真相只有一個: 事實文字檢索與查核競賽   
## 運行環境  
### 使用 Anaconda 建立環境 :   
conda env create -f environment.yml  
conda activate AI_CUP  

## 資料  
data/raw : 裡面是未經過處理的原始資料集  
data/Stage 1 : 經過Stage 1 Stage1_BM25Final.ipynb 及 Stage1_Un_BM25.ipynb處理過的資料集  
data/Stage 2/PublicTrain/WithoutBM25 : Train Data 經過 SentenceRetrievalTrain.ipynb 運算後產出的訓練數據集  
data/Stage 2/Private : Private Data 經過 SentenceRetrievalInfer.ipynb 運算後產出的Infer數據集  
data/Stage2ModelTrainData : Stage 2 的訓練集  
data/Stage3ModelTrainData : Stage 3 的訓練集  
**還有一項資料wiki pages因檔案太大，所以沒傳上來，執行程式前，請先解壓後再放在data資料夾(路徑:data/wiki-pages)**  
**wiki pages為競賽網頁上的資料，這裡有提供下載連結**  
### [wiki pages下載網址](https://drive.google.com/drive/folders/195FIG2ZCyI-VqZJtZLG3aHcJxQ13KjdI?usp=sharing)  
## Stage 1
因為會遇到wikipedia 在search時候的一些問題，所以Stage 1是在Colab雲端上運算  
Stage1_BM25Final.ipynb : 透過BM25演算法搜尋文章  
Stage1_Un_BM25.ipynb ，處理訓練資料，以利後續Stage 2 的訓練  
Stage1_BM25Final.ipynb 處理Test及Private的資料，後續Stage 2 Infer用的資料
## 訓練 
Stage 2 : SentenceRetrievalTrain.ipynb ，產生的訓練資料會繼續在Stage 3訓練  
Stage 3 : ClaimVerTrain.ipynb
## 預測  
### [權重下載網址](https://drive.google.com/drive/folders/1ejU6aEcdF7dcGH85tKRLN4wNgHPahtS0?usp=sharing)  
請先下載權重並且放在weights資料夾中，連結有提供Stage2 及 3的模型權重  
Stage 2 : SentenceRetrievalInfer.ipynb  ，產生的Infer資料會在Stage 3運算  
Stage 3 : ClaimVerInfer.ipynb  ， 產生最後的預測檔案  
備註: 因為在實驗的過程中，觀察到wikipedia.search()會傳的資料會有浮動，再還沒執行Stage 1 及Stage 2的情況下，下載好模型權重後，可以直接執行ClaimVerInfer.ipynb，產生最後的預測檔案  
