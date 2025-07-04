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
### 2) Install Dependencies (윈도우 환경은 2, 3 직접 실행)

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

### Baseline
* Model : `mistralai/Mistral-7B-Instruct-v0.2`
* NO RAG

|                      평가 점수                      |  Exact Match | BLEURT | BERTScore | ROUGE-1 
|:-----------------------------------------------:|:------------:|:---:|:---:|:---:|
| <span style="color:green">**48.2280559**</span> |  45.5823293  |  43.7259803 |  73.9317909 |  34.9635763

### RAG_3
* Model : `kullm-polyglot-12.8b-v2`

| 평가 점수 | Exact Match |   BLEURT   |   BERTScore   | ROUGE-1 
|:--------------:|:-----------:|:----------:|:-------------:|:---:|
| <span style="color:green">**45.8880031**</span> | 41.7670682  | 50.8583344 | 71.5458505    | 27.622629

### RAG_5
* Model : `beomi/Llama-3-Open-Ko-8B-Instruct-preview`

| 평가 점수 | Exact Match |   BLEURT   |   BERTScore   | ROUGE-1 
|:--------------:|:-----------:|:----------:|:-------------:|:---:|
| <span style="color:green">**48.9918717**</span> | 46.1847389  | 49.2321253 | 74.1369496    | 32.0279384

### RAG_6
* Model : `Qwen/Qwen1.5-7B-Chat`

| 평가 점수 | Exact Match |   BLEURT   |   BERTScore   | ROUGE-1 
|:--------------:|:-----------:|:----------:|:-------------:|:---:|
| <span style="color:green">**36.5028852**</span> | 26.3052208  | 43.3782998 | 71.3783302 | 25.3450191

### RAG_7
* Model : `beomi/Llama-3-Open-Ko-8B-Instruct-preview`
* Retriever : `HybridRerankRetriever`
  * Sparse: `BM25Okapi`
  * Dense: `snunlp/KR-SBERT-V40K-klueNLI-augSTS`
  * Reranking: `bongsoo/klue-cross-encoder-v1`
* Aditional Tokenization

| 평가 점수 | Exact Match |   BLEURT   |   BERTScore   | ROUGE-1 
|:--------------:|:-----------:|:----------:|:-------------:|:---:|
| <span style="color:green">**47.9063702**</span> | 44.37751 | 49.2970475 | 73.8285655 | 31.180078

### RAG_8
* Model : `beomi/Llama-3-Open-Ko-8B-Instruct-preview`
* Retriever : `HybridRerankRetriever`
  * Sparse: `BM25Okapi`
  * Dense: `snunlp/KR-SBERT-V40K-klueNLI-augSTS`
  * Reranking: `bongsoo/klue-cross-encoder-v1`
* Additional Tokenization + Sentence Splitting
* PEFT (Parameter-Efficient Fine-Tuning)
  
| 평가 점수 | Exact Match |   BLEURT   | BERTScore    | ROUGE-1 
|:--------------:|:-----------:|:----------:|:------------:|:---:|
| <span style="color:green">****</span> |  |  |              | 