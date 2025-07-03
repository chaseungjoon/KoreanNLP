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

### NO_RAG (Mistral 7B Instruct)
| **평가 점수** | Exact Match | BLEURT | BERTScore | ROUGE-1 
|:---:|:-----------:|:---:|:---:|:---:|
|  **48.2280559** | 45.5823293  |  43.7259803 |  73.9317909 |  34.9635763

### RAG_3 (kullm-polyglot-12.8b-v2)
|   **평가 점수**    | Exact Match |   BLEURT   |   BERTScore   | ROUGE-1 
|:--------------:|:-----------:|:----------:|:-------------:|:---:|
| **45.8880031** | 41.7670682  | 50.8583344 | 71.5458505    | 27.622629

### RAG_5 (Llama-3-Open-Ko-8B-Instruct-preview + Additional Tokenization)
|   **평가 점수**    | Exact Match |   BLEURT   |   BERTScore   | ROUGE-1 
|:--------------:|:-----------:|:----------:|:-------------:|:---:|
| **48.9918717** | 46.1847389  | 49.2321253 | 74.1369496    | 32.0279384