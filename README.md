# 국립국어원 한국어 어문 규범 기반 생성 (RAG) 모델

[대회 링크](https://kli.korean.go.kr/benchmark/taskOrdtm/taskList.do?taskOrdtmId=182&clCd=ING_TASK&subMenuId=sub01)

---

## 실행 방법

### 0) Git clone and **Setup venv (recommended)**

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

### 3) Run
```bash
chmod +x run && ./run
```

---

## Score

### RAG_2 (Mistral 7B Instruct)
| **평가 점수** | Exact Match | BLEURT | BERTScore | ROUGE-1 
|:---:|:-----------:|:---:|:---:|:---:|
|  **48.2280559** | 45.5823293  |  43.7259803 |  73.9317909 |  34.9635763

### RAG_3 (kullm-polyglot-12.8b-v2 + KR-SBERT-V40K-klueNLI-augSTS)
| **평가 점수** | Exact Match | BLEURT | BERTScore | ROUGE-1 
|:---:|:-----------:|:---:|:---:|:---:|
|  |  |  |  | 
