"""
HOW TO RUN (Advised to use virtual environment)

1. Install dependencies
chmod +x setup && ./setup

2. Train the model
chmod +x train && ./train

3. Run inference
chmod +x predict && ./predict

"""

import argparse, json, os, pathlib, warnings, re
from tqdm import tqdm
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from transformers.utils import logging as hf_logging
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=UserWarning)

THIS_DIR = pathlib.Path(__file__).resolve().parent
TOKEN_FALLBACK_FILE = THIS_DIR / "hf_token.txt"

PROMPT = "You are a helpful AI assistant. 당신은 한국어 어문 규범 전문가 입니다."
INST = {
    "선택형": "[지침] 보기 중 올바른 표현을 골라 “○○가 옳다. 이유: …”로 답하십시오.",
    "교정형": "[지침] 틀린 부분을 고친 뒤 “○○가 옳다. 이유: …”로 답하십시오.",
    "선다형": "[지침] 가장 적절한 보기 번호만 숫자로 답하십시오.",
    "단답형": "[지침] 두 단어 이내로 간단히 답하십시오.",
    "서술형": "[지침] 완전한 문장으로 서술하십시오."
}

"""
HUGGINGFACE TOKEN RESOLVER
"""
def resolve_token(cli_token: Optional[str] = None) -> Optional[str]:
    if cli_token:
        return cli_token
    env_token = os.getenv("HUGGINGFACE_TOKEN")
    if env_token:
        return env_token
    if TOKEN_FALLBACK_FILE.exists():
        return TOKEN_FALLBACK_FILE.read_text(encoding="utf-8").strip()
    print("No Hugging Face token provided.\nPlease set it via HUGGINGFACE_TOKEN environment variable, or hf_token.txt file.")
    return None

"""
SBERT RETRIEVER
"""
class SBERTRetriever:
    def __init__(self, passages: List[str], model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS", batch: int = 64):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(model_name, device=device)
        self.passages = passages
        self.embeddings = self.encoder.encode(passages, batch_size=batch, show_progress_bar=True,
                                              normalize_embeddings=True)
        self.nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(self.embeddings)

    def query(self, question: str, k: int = 5) -> List[str]:
        emb = self.encoder.encode([question], normalize_embeddings=True)
        _, idx = self.nn.kneighbors(emb, n_neighbors=k)
        return [self.passages[i] for i in idx[0]]

"""
DATASET FOR RAG
"""
class KoreanRAGDataset(Dataset):
    def __init__(self, items: List[Dict], retriever: SBERTRetriever, tokenizer, max_len: int = 1024):
        self.items = items
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def _prompt(self, question: str, qtype: str) -> str:
        ctx = self.retriever.query(question, k=5)
        ctx_block = "\n".join(f"- {c}" for c in ctx)
        inst = INST.get(qtype, INST["서술형"])
        return (
            f"{PROMPT}\n\n"
            f"[참고 문헌]\n{ctx_block}\n\n"
            f"[질문 유형]\n{qtype}\n\n"
            f"[질문]\n{question}\n\n"
            f"{inst}\n\n답변:"
        )

    def __getitem__(self, idx):
        sample = self.items[idx]
        q = sample["input"]["question"]
        qtype = sample["input"].get("type", "서술형")
        a = sample["output"]["answer"]
        text = self._prompt(q, qtype) + " " + a
        tok = self.tokenizer(text, truncation=True, max_length=self.max_len, padding="max_length")
        tok["labels"] = tok["input_ids"].copy()
        return {k: torch.tensor(v) for k, v in tok.items()}

"""
LOAD JSON DATA
"""
def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

"""
LOAD BASE MODEL
"""
def load_base_model(name: str, load_in_4bit: bool = True, token: Optional[str] = None):
    bnb_cfg = None
    if load_in_4bit:
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                     bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        name, device_map="auto", trust_remote_code=True, quantization_config=bnb_cfg, token=token
    )
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, token=token)
    tok.pad_token = tok.eos_token
    return model, tok

"""
TRAINING
"""
def train(args):
    token = resolve_token(args.hf_token)

    passages = [l.strip() for l in open(args.reference_path, encoding="utf-8").read().splitlines() if l.strip()]
    retriever = SBERTRetriever(passages)

    train_data = load_json(args.train_path)
    dev_data = load_json(args.dev_path) if args.dev_path else train_data[: max(1, len(train_data) // 10)]

    model, tokenizer = load_base_model(args.model_name, token=token)
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)
    gpt_neox_targets = ["query_key_value", "dense"]

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=gpt_neox_targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = KoreanRAGDataset(train_data, retriever, tokenizer)
    eval_ds = KoreanRAGDataset(dev_data, retriever, tokenizer)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=2e-4,
        bf16=True,
        gradient_accumulation_steps=args.grad_accum,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
    )

    Trainer(model=model, tokenizer=tokenizer, args=targs,
            train_dataset=train_ds, eval_dataset=eval_ds).train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    torch.save({"passages": passages, "embeddings": retriever.embeddings}, args.output_dir + "/retriever.pt")

"""
INFERENCE
"""
def load_retriever(saved_dir: str, sbert_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS") -> SBERTRetriever:
    obj = torch.load(os.path.join(saved_dir, "retriever.pt"), map_location="cpu")
    ret = SBERTRetriever(obj["passages"], sbert_name)
    ret.embeddings = obj["embeddings"]
    ret.nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(ret.embeddings)
    return ret


def postprocess(text: str) -> str:
    text = text.strip()

    if re.match(r'.+가 옳다', text):
        return text

    m = re.search(r'[\"“”]([^\"”]+)[\"”]', text)
    if m:
        fix = f'{m.group(1)}가 옳다. '
        rest = text[m.end():].lstrip(' .')
        return fix + rest

    first = text.split()[0]
    return f'{first}가 옳다. ' + text


def predict(args):
    token = resolve_token(args.hf_token)

    model, tokenizer = load_base_model(args.model_name, load_in_4bit=True, token=token)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, token=token)
    model.eval()

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    retriever = load_retriever(args.adapter_path or pathlib.Path(args.reference_path).parent)
    data = load_json(args.test_path)
    hf_logging.set_verbosity_error()

    with open(args.output_path, "w", encoding="utf-8") as outf:
        for sample in tqdm(data, desc="Generating", unit="sample", total=len(data)):
            q = sample["input"]["question"]
            qtype = sample["input"].get("type", "서술형")
            ctx = retriever.query(q, k=5)
            ctx_block = "\n".join(f"- {c}" for c in ctx)
            inst = INST.get(qtype, INST["서술형"])
            prompt = (
                f"{PROMPT}\n\n"
                f"[참고 문헌]\n{ctx_block}\n\n"
                f"[질문 유형]\n{qtype}\n\n"
                f"[질문]\n{q}\n\n"
                f"{inst}\n\n"
                f"※ 답변 형식: ① 정답 표현만 먼저 쓰고, ② 바로 ‘가 옳다. 이유: …’를 이어서 쓰십시오.\n\n"
                f"답변:"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            inputs.pop("token_type_ids", None)
            gen = model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            ans = postprocess(tokenizer.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
            result = {
                "id": sample["id"],
                "input": {
                    "question_type": sample["input"].get("question_type")
                                     or sample["input"].get("type")
                                     or "서술형",
                    "question": sample["input"]["question"],
                },
                "output": {
                    "answer": ans
                }
            }
            outf.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Saved predictions → {args.output_path}")

"""
MAIN FUNCTION 
"""
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["train", "predict"])
    p.add_argument("--model_name", required=True)
    p.add_argument("--reference_path", required=True)
    p.add_argument("--hf_token")

    p.add_argument("--train_path")
    p.add_argument("--dev_path")
    p.add_argument("--output_dir", default="ckpt")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--grad_accum", type=int, default=8)

    p.add_argument("--test_path")
    p.add_argument("--adapter_path")
    p.add_argument("--output_path", default="submission.jsonl")

    args = p.parse_args()

    if args.mode == "train":
        train(args)
        if args.test_path:
            print("\nTraining complete — starting inference …\n")
            args.mode = "predict"
            args.adapter_path = args.output_dir
            predict(args)
        else:
            print("\nNo --test_path supplied, so inference was skipped.")
    else:
        predict(args)

if __name__ == "__main__":
    main()