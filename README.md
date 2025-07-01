# 국립국어원 한국어 어문 규범 기반 생성 (RAG) 모델

https://kli.korean.go.kr/benchmark/taskOrdtm/taskList.do?taskOrdtmId=182&clCd=ING_TASK&subMenuId=sub01

## 실행 방법

### 1) huggingface 토큰 (write) 환경 변수로 설정

```bash
export HF_TOKEN="hf_XXXXXXXXXXXXXXX"
```
### 2) 다운로드

```bash
git clone this-repo-url
cd this-repo-name
pip install -r requirements.txt
```

### 3) 훈련 및 inference 실행 (CUDA GPU 필요)
```bash
chmod +x run
./run
```

### 4) submission.jsonl 생성 & 제출