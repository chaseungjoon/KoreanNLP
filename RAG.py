# RAG.py  ── python3 RAG.py ───────────────────────────────────────────
import json, gc, torch, faiss, os, time, sys
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          GenerationConfig, TextStreamer)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TRAIN_PATH = "data/korean_language_rag_V1.0_train.json"
TEST_PATH  = "data/korean_language_rag_V1.0_test.json"
OUT_PATH   = "submission2.json"

LLM_NAME       = "mistralai/Mistral-7B-Instruct-v0.2"
RETRIEVER_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

DEVICE_LLM = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"● device = {DEVICE_LLM}")

def jload(p): return json.load(open(p, encoding="utf-8"))
train, test = jload(TRAIN_PATH), jload(TEST_PATH)

corpus = [ex["input"]["question"] + " " + ex["output"]["answer"]
          for ex in train]

print("▣ SBERT 임베딩…")
retriever = SentenceTransformer(RETRIEVER_NAME, device="cpu")
emb = retriever.encode(corpus, batch_size=16,
                       convert_to_numpy=True, show_progress_bar=True)
faiss.normalize_L2(emb)
index = faiss.IndexFlatIP(emb.shape[1]);  index.add(emb)

def ctx(q, k=4):
    e = retriever.encode([q], convert_to_numpy=True);  faiss.normalize_L2(e)
    _, I = index.search(e, k);  return [corpus[i] for i in I[0]]

print("▣ LLM 로드(fp16)…")
model = AutoModelForCausalLM.from_pretrained(
            LLM_NAME,
            device_map={"": DEVICE_LLM},
            torch_dtype=torch.float16)
tok = AutoTokenizer.from_pretrained(LLM_NAME, use_fast=True)

tok.pad_token = tok.eos_token
model.config.pad_token_id = tok.pad_token_id = tok.eos_token_id

gen_cfg = GenerationConfig(
            max_new_tokens=320,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id
)

def build_prompt(q, cs):
    refs = "\n\n".join(f"◎ 참고자료\n{c}" for c in cs)
    return (f"다음 질문에 어문 규범에 맞는 정답과 이유를 제시하라.\n"
            f"{refs}\n\n질문: {q}\n답:")


def heartbeat(msg):
    sys.stdout.write(msg + "\n"); sys.stdout.flush()

print("▣ Test 생성 시작…"); heartbeat(" ")

_ = model.generate(**tok("warm-up", return_tensors="pt").to("mps"),
                   max_new_tokens=1)

tot = len(test)
pbar = tqdm(total=tot, ncols=90, desc="Generating")
preds = []

for ex in test:
    q        = ex["input"]["question"]
    prompt   = build_prompt(q, ctx(q, k=4))
    inp      = tok(prompt, return_tensors="pt").to("mps")

    start = time.time(); last_ping = start
    with torch.no_grad():
        out = model.generate(**inp, generation_config=gen_cfg)

    end   = time.time()

    gen_tok   = out.shape[-1] - inp.input_ids.shape[-1]
    tok_speed = gen_tok / (end - start + 1e-9)

    heartbeat(f"[sample {len(preds)+1}/{tot}] {gen_tok} tok "
              f"│ {end-start:4.1f}s │ {tok_speed:.2f} tok/s")

    ans = tok.decode(out[0], skip_special_tokens=True).split("답:",1)[-1].strip()
    preds.append({"id": ex["id"], "input": ex["input"],
                  "output": {"answer": ans}})
    pbar.update(1)
    pbar.set_postfix(tok=gen_tok, spd=f"{tok_speed:.2f}/s")
pbar.close(); print("✓ inference done")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(preds, f, ensure_ascii=False, indent=2)
print(f"✓ 완료 → {OUT_PATH}")

gc.collect();  torch.mps.empty_cache()