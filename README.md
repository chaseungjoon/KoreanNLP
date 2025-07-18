# 국립국어원 한국어 어문 규범 기반 생성 (RAG) 경진대회

[대회 링크](https://kli.korean.go.kr/benchmark/taskOrdtm/taskList.do?taskOrdtmId=182&clCd=ING_TASK&subMenuId=sub01)

---

## 실행 방법

### 0) Git clone and **setup virtual environment (recommended)**

```bash
git clone https://github.com/chaseungjoon/KoreanNLP.git
cd KoreanNLP
python -m venv .venv
source .venv/bin/activate
```
### 2) Install Dependencies

```bash
chmod +x setup && ./setup
```

### 3) Run (train + predict)
```bash
chmod +x run && ./run
```

### (Optional) Train and predict separately
```bash
chmod +x train && ./train
chmod +x predict && ./predict
```


---

## Score

### Baseline (7b)
* Model : `mistralai/Mistral-7B-Instruct-v0.2`
* NO RAG

|                      평가 점수                      |  Exact Match | BLEURT | BERTScore | ROUGE-1 
|:-----------------------------------------------:|:------------:|:---:|:---:|:---:|
| <span style="color:green">**48.2280559**</span> |  45.5823293  |  43.7259803 |  73.9317909 |  34.9635763

### RAG_3 (12.8b)
* Model : `kullm-polyglot-12.8b-v2`

| 평가 점수 | Exact Match |   BLEURT   |   BERTScore   | ROUGE-1 
|:--------------:|:-----------:|:----------:|:-------------:|:---:|
| <span style="color:green">**45.8880031**</span> | 41.7670682  | 50.8583344 | 71.5458505    | 27.622629

### RAG_5 (8b)
* Model : `beomi/Llama-3-Open-Ko-8B-Instruct-preview`

| 평가 점수 | Exact Match |   BLEURT   |   BERTScore   | ROUGE-1 
|:--------------:|:-----------:|:----------:|:-------------:|:---:|
| <span style="color:green">**48.9918717**</span> | 46.1847389  | 49.2321253 | 74.1369496    | 32.0279384

### RAG_6 (7b)
* Model : `Qwen/Qwen1.5-7B-Chat`

| 평가 점수 | Exact Match |   BLEURT   |   BERTScore   | ROUGE-1 
|:--------------:|:-----------:|:----------:|:-------------:|:---:|
| <span style="color:green">**36.5028852**</span> | 26.3052208  | 43.3782998 | 71.3783302 | 25.3450191

### RAG_7 (8b)
* Model : `beomi/Llama-3-Open-Ko-8B-Instruct-preview`
* Retriever : `HybridRerankRetriever`
  * Sparse: `BM25Okapi`
  * Dense: `snunlp/KR-SBERT-V40K-klueNLI-augSTS`
  * Reranking: `bongsoo/klue-cross-encoder-v1`
* Aditional Tokenization

| 평가 점수 | Exact Match |   BLEURT   |   BERTScore   | ROUGE-1 
|:--------------:|:-----------:|:----------:|:-------------:|:---:|
| <span style="color:green">**47.9063702**</span> | 44.37751 | 49.2970475 | 73.8285655 | 31.180078

### RAG_8 (8b)
* Model : `beomi/Llama-3-Open-Ko-8B-Instruct-preview`
* Retriever : `HybridRerankRetriever`
  * Sparse: `BM25Okapi`
  * Dense: `snunlp/KR-SBERT-V40K-klueNLI-augSTS`
  * Reranking: `bongsoo/klue-cross-encoder-v1`
* Additional Tokenization + Sentence Splitting
* PEFT (Parameter-Efficient Fine-Tuning)
  
|                      평가 점수                      | Exact Match |   BLEURT   | BERTScore  | ROUGE-1 
|:-----------------------------------------------:|:-----------:|:----------:|:----------:|:---:|
| <span style="color:green">**50.5155902**</span> | 48.3935743  | 49.9495335 | 74.3985577 |  33.5647271

### RAG_9 (10.8b)
* Model : `yanolja/EEVE-Korean-Instruct-10.8B-v1.0`
* Retriever : `HybridRerankRetriever`
  * Dense: `intfloat/multilingual-e5-large`
  * Reranking: `BAAI/bge-reranker-large`
* Semantic chunking
* Early stopping

|                      평가 점수                      | Exact Match | BLEURT  | BERTScore | ROUGE-1 
|:-----------------------------------------------:|:-----------:|:-------:|:---------:|:---:|
| <span style="color:green">**22.5340542**</span> |    0.00     |    44.1237656     |      68.6064641     |  22.4740958

### RAG_10 (10.8b)
* Model : `yanolja/EEVE-Korean-Instruct-10.8B-v1.0`
* Retriever : RAG_8 Ver.
* Early stopping

|                          평가 점수                           |      Exact Match      |      BLEURT       | BERTScore | ROUGE-1 
|:--------------------------------------------------------:|:---------------------:|:-----------------:|:---------:|:---:|
|     <span style="color:green">**49.4470553**</span>      |      50.6024096       |    46.5091527     |     70.6664575      |  27.6994927

### RAG_11 (10.8b)
* Model : `yanolja/EEVE-Korean-Instruct-10.8B-v1.0`
* Retriever : AdvancedKoreanRetriever
    * Dense: `jhgan/ko-sroberta-multitask`
    * Reranking: `bongsoo/klue-cross-encoder-v1`
* Multi-Stage Reranking
* Prompt Engineering w/ chain of thought

<span style="color:red">***-> Token generation error (contaminated data)***</span>

|                      평가 점수                      | Exact Match | BLEURT  | BERTScore | ROUGE-1 
|:-----------------------------------------------:|:-----------:|:-------:|:---------:|:---:|
| <span style="color:green">**16.5065031**</span> |    0.00     |   24.4582521    |   63.3902907    |  11.1904761

### RAG_12 (10.8b)
* Model : `yanolja/EEVE-Korean-Instruct-10.8B-v1.0`
* Retriever : AdvancedKoreanRetriever
    * Dense: `snunlp/KR-SBERT-V40K-klueNLI-augSTS`
    * Reranking: `bongsoo/klue-cross-encoder-v1`
* Two pass prompting
* Ensemble voting
* pre-Augmented training data

|                      평가 점수                      | Exact Match | BLEURT | BERTScore | ROUGE-1 
|:-----------------------------------------------:|:----------:|:------:|:---------:|:---:|
| <span style="color:green">****</span> |            |        |           |  