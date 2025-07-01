#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU-only inference + 진행 상황 표시(스피너 · progress bar · heartbeat)
"""

import os, sys, time, json, gc, warnings, threading, itertools
import torch, faiss
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          GenerationConfig, TextStreamer)

# ── 환경 설정 ────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"]      = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BAR"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = "data"
TRAIN_PATH = f"{DATA_DIR}/korean_language_rag_V1.0_train.json"
TEST_PATH  = f"{DATA_DIR}/korean_language_rag_V1.0_test.json"
OUT_PATH   = "submission_cpu.json"

LLM_NAME       = "mistralai/Mistral-7B-Instruct-v0.2"
RETRIEVER_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

DEVICE_LLM = "cpu"

def jload(p):  return json.load(open(p, encoding="utf-8"))
def heartbeat(msg: str):
    sys.stdout.write(msg + "\n"); sys.stdout.flush()

train, test = jload(TRAIN_PATH), jload(TEST_PATH)
corpus = [ex["input"]["question"] + " " + ex["output"]["answer"]
          for ex in train]

print("Imbedding SBERT")
retriever = SentenceTransformer(RETRIEVER_NAME, device="cpu")
emb = retriever.encode(corpus,
                       batch_size=16,
                       convert_to_numpy=True,
                       show_progress_bar=True)
faiss.normalize_L2(emb)
index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)

def ctx(q, k=3):
    e = retriever.encode([q], convert_to_numpy=True); faiss.normalize_L2(e)
    _, I = index.search(e, k)
    return [corpus[i] for i in I[0]]

print("Loading LLM")
model = AutoModelForCausalLM.from_pretrained(
            LLM_NAME,
            device_map={"": "cpu"},
            torch_dtype=torch.float32)
tok = AutoTokenizer.from_pretrained(LLM_NAME, use_fast=True)
tok.pad_token = tok.eos_token
model.config.pad_token_id = tok.pad_token_id = tok.eos_token_id

gen_cfg = GenerationConfig(
    max_new_tokens = 200,
    temperature    = 0.3,
    top_p          = 0.9,
    do_sample      = True,
    pad_token_id   = tok.pad_token_id,
    eos_token_id   = tok.eos_token_id
)

def build_prompt(q, cs):
    refs = "\n\n".join(f"◎ 참고자료\n{c}" for c in cs)
    return (f"다음 질문에 어문 규범에 맞는 정답과 이유를 제시하라.\n"
            f"{refs}\n\n질문: {q}\n답:")

def spin(msg="Compiling"):
    for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
        if spin.stop: break
        sys.stdout.write(f"\r{msg}{ch}")
        sys.stdout.flush()
        time.sleep(0.15)
spin.stop = False
threading.Thread(target=spin, daemon=True).start()

warm_inp = tok("warm-up", return_tensors="pt")
_ = model.generate(**warm_inp, generation_config=gen_cfg)

spin.stop = True
sys.stdout.write("\r✓ warm-up done\n"); sys.stdout.flush()

tot  = len(test)
pbar = tqdm(total=tot, ncols=100, desc="Generating")
streamer = TextStreamer(tok, skip_prompt=True)

preds = []
for idx, ex in enumerate(test, 1):
    q      = ex["input"]["question"]
    prompt = build_prompt(q, ctx(q))
    inp    = tok(prompt, return_tensors="pt")

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, generation_config=gen_cfg,
                             streamer=streamer)
    t1 = time.time()

    gen_tok = out.shape[-1] - inp.input_ids.shape[-1]
    tok_spd = gen_tok / (t1 - t0 + 1e-9)
    heartbeat(f"[{idx:>3}/{tot}] {gen_tok:>4} tok │ {t1 - t0:5.1f}s │ "
              f"{tok_spd:4.2f} tok/s")

    ans = tok.decode(out[0], skip_special_tokens=True) \
            .split("답:", 1)[-1].strip()

    preds.append({"id": ex["id"],
                  "input": ex["input"],
                  "output": {"answer": ans}})

    pbar.update(1)
    pbar.set_postfix(tok=gen_tok, spd=f"{tok_spd:4.2f}/s")

    del inp, out
    gc.collect()

pbar.close()
print("inference done!")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(preds, f, ensure_ascii=False, indent=2)
print(f"Saved → {OUT_PATH}")