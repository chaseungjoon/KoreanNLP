"""
python RAG_Mistral.py \
  --train korean_language_rag_V1.0_train.json \
  --dev   korean_language_rag_V1.0_dev.json \
  --test  korean_language_rag_V1.0_test.json \
  --output submission.jsonl \
  --model_id mistralai/Mistral-7B-Instruct-v0.2 \
  --device cuda
"""

import argparse, json, os, tqdm, torch, faiss
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

hf_token = os.getenv("HF_TOKEN")
login(hf_token)

sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
ref_texts = [line.strip() for line in open("reference.txt", encoding="utf-8") if line.strip()]
ref_embs = sbert.encode(ref_texts, convert_to_numpy=True, show_progress_bar=True)

dim = ref_embs.shape[1]
faiss.normalize_L2(ref_embs)
index = faiss.IndexFlatIP(dim)
index.add(ref_embs)

def retrieve_docs(q, topk=3):
    emb = sbert.encode([q], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    D, I = index.search(emb, topk)
    return [ref_texts[i] for i in I[0]]

PROMPT = "You are a helpful AI assistant. 당신은 한국어 어문 규범 전문가입니다."
INST = {
    "선택형": "[지침] 보기 중 올바른 표현을 골라 “○○가 옳다. 이유: …”로 답하십시오.",
    "교정형": "[지침] 틀린 부분을 고친 뒤 “○○가 옳다. 이유: …”로 답하십시오.",
    "선다형": "[지침] 가장 적절한 보기 번호만 숫자로 답하십시오.",
    "단답형": "[지침] 두 단어 이내로 간단히 답하십시오.",
    "서술형": "[지침] 완전한 문장으로 서술하십시오."
}

class RAGDataset(Dataset):
    def __init__(self, path, tokenizer, train=True):
        data = json.load(open(path, encoding="utf-8"))
        self.samples = []
        for it in data:
            qt, q = it["input"]["question_type"], it["input"]["question"]
            docs = retrieve_docs(q)
            ctx = PROMPT + "\n" + INST.get(qt, "") + "\n\n[질문]\n" + q
            ctx += "\n\n[참고 문서]\n" + "\n".join(docs)
            if train and it.get("output", {}).get("answer"):
                ctx += "\n\n" + it["output"]["answer"]
            toks = tokenizer(
                ctx, max_length=512, truncation=True, padding="max_length",
                return_tensors="pt"
            )
            input_ids = toks.input_ids.squeeze()
            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": toks.attention_mask.squeeze(),
                "labels": input_ids.clone()
            })

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        return self.samples[i]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tk = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    tk.pad_token = tk.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", quantization_config=bnb, token=hf_token
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM", inference_mode=False,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules="all-linear"
    )
    model.add_adapter(lora_cfg, adapter_name="lora_1")
    model.set_adapter("lora_1")

    train_ds = RAGDataset(args.train, tk, train=True)
    dev_ds = RAGDataset(args.dev, tk, train=True)

    tr_args = TrainingArguments(
        output_dir="qlora_ckpt",
        per_device_train_batch_size=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=torch.cuda.is_bf16_supported(),
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False
    )
    trainer = Trainer(
        model=model, args=tr_args,
        train_dataset=train_ds, eval_dataset=dev_ds,
        tokenizer=tk
    )
    trainer.train()

    test_data = json.load(open(args.test, encoding="utf-8"))
    model.eval()
    with open(args.output, "w", encoding="utf-8") as fout:
        for it in tqdm.tqdm(test_data):
            qt, q = it["input"]["question_type"], it["input"]["question"]
            docs = retrieve_docs(q)
            prompt = PROMPT + "\n" + INST.get(qt, "") + "\n\n[질문]\n" + q
            prompt += "\n\n[참고 문서]\n" + "\n".join(docs)
            enc = tk(prompt, max_length=512, truncation=True, return_tensors="pt").to(args.device)
            gen = model.generate(
                **enc, max_new_tokens=200, pad_token_id=tk.eos_token_id
            )
            txt = tk.decode(
                gen[0, enc.input_ids.shape[-1]:], skip_special_tokens=True
            ).strip()
            it["output"] = {"answer": f'"{txt}"'}
            json.dump(it, fout, ensure_ascii=False); fout.write("\n")

    print("Saved to ", os.path.abspath(args.output))

if __name__ == "__main__":
    main()