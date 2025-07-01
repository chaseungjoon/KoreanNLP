"""
python RAG_2.py \
  --train  /content/korean_language_rag_V1.0_train.json \
  --dev    /content/korean_language_rag_V1.0_dev.json \
  --test   /content/korean_language_rag_V1.0_test.json \
  --output /content/submission.json \
  --model_id mistralai/Mistral-7B-Instruct-v0.2 \
  --device cuda
"""

import argparse, json, os, tqdm, torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling)

PROMPT_SYSTEM = "You are a helpful AI assistant. 당신은 한국어 어문 규범 전문가입니다. 질문을 읽고 올바른 답과 이유를 제시하세요."
INST = {
    "선택형": "[지침] 보기 중 올바른 표현을 골라 “○○가 옳다. 이유: …”로 답하십시오.",
    "교정형": "[지침] 틀린 부분을 고친 뒤 “○○가 옳다. 이유: …”로 답하십시오.",
    "선다형": "[지침] 가장 적절한 보기 번호만 숫자로 답하십시오.",
    "단답형": "[지침] 두 단어 이내로 간단히 답하십시오.",
    "서술형": "[지침] 완전한 문장으로 서술하십시오."
}

class RAGDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.data = json.load(open(path, encoding="utf-8"))
        self.tokenizer = tokenizer
        self.samples = []
        for item in self.data:
            qt = item["input"]["question_type"]
            q = item["input"]["question"]
            a = item["output"]["answer"] if "output" in item and "answer" in item["output"] else ""
            chat = [
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": f"{INST.get(qt, '')}\n\n[질문]\n{q}"},
                {"role": "assistant", "content": a}
            ]
            prompt = tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=False)[0]
            self.samples.append(prompt)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    hf_token = open("hf_token.txt").read().strip() if os.path.exists("hf_token.txt") else None
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model_kwargs = dict(device_map="auto", quantization_config=bnb, max_memory={0: "13GiB", "cpu": "32GiB"})
    if hf_token: model_kwargs["use_auth_token"] = hf_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_auth_token=hf_token if hf_token else None)
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = RAGDataset(args.train, tokenizer)
    dev_ds = RAGDataset(args.dev, tokenizer)

    training_args = TrainingArguments(
        output_dir="./rag_finetune_ckpt",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        save_strategy="epoch",
        save_total_limit=1,
        num_train_epochs=3,
        learning_rate=2e-5,
        bf16=True if torch.cuda.is_available() else False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()

    # Inference
    test_data = json.load(open(args.test, encoding="utf-8"))
    model.eval()
    with torch.no_grad():
        for idx, item in enumerate(tqdm.tqdm(test_data)):
            qt = item["input"]["question_type"]
            q = item["input"]["question"]
            chat = [{"role": "system", "content": PROMPT_SYSTEM},
                    {"role": "user", "content": f"{INST.get(qt, '')}\n\n[질문]\n{q}"}]
            enc = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt")[0].to(args.device)
            gen = model.generate(input_ids=enc.unsqueeze(0), max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
            txt = tokenizer.decode(gen[0][enc.shape[-1]:], skip_special_tokens=True).strip()
            for p in ("답변:", "답:", "Answer:", "answer:"):
                if txt.lower().startswith(p.lower()): txt = txt[len(p):].strip()
            if not txt.startswith('"'): txt = '"' + txt
            if not txt.endswith('"'): txt = txt + '"'
            item["output"] = {"answer": txt}

    with open(args.output, "w", encoding="utf-8") as f:
        for item in test_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\\n")

if __name__ == "__main__":
    main()
