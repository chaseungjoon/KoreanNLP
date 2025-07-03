"""
HOW TO RUN (Advised to use virtual environment)

1. Install dependencies
chmod +x setup && ./setup

2. Run the model (train + predict)
chmod +x run && ./run

(Optional) Run inference after training
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

PROMPT = "You are a helpful AI assistant. Please answer the user's questions kindly. \
            당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요. \
            단, 동일한 문장을 절대 반복하지 마시오."
INST = {
    "선다형": (
        "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
        "[지침]\n"
        "주어진 보기 중에서 가장 적절한 답을 숫자로만 응답하시오.\n\n"
        "[예시]\n"
        "질문: 다음 한국의 전통 놀이 중 '조선시대'에 행한 놀이는?\n"
        "1) 주사위 놀이\n"
        "2) 검무\n"
        "3) 격구\n"
        "4) 영고\n"
        "5) 무애무\n"
        "답변: 3"
    ),
    "서술형": (
        "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
        "[지침]\n"
        "질문에 대한 답변을 완성된 문장으로 서술하시오.\n\n"
        "[예시]\n"
        "질문: 대한민국의 행정구역 체계를 서술하세요.\n"
        "답변: 대한민국의 행정구역은 여러 종류의 지역 단위로 나뉘어 구성되어 있으며, 먼저 특별시와 광역시부터 살펴볼 수 있다. 특별시로는 수도인 서울특별시가 있으며, 광역시에는 인천광역시, 부산광역시, 대전광역시, 광주광역시, 대구광역시, 울산광역시 등이 포함된다. 이 외에도 대한민국은 일반 도 단위로 6개의 도를 두고 있는데, 그 이름은 경기도, 충청북도, 충청남도, 전라남도, 경상북도, 경상남도로 구성되어 있다. 특별한 자치권을 부여받은 도인 특별자치도로는 제주특별자치도, 전북특별자치도, 강원특별자치도가 있다. 마지막으로 특별자치시로는 세종특별자치시가 존재한다."
    ),
    "단답형": (
        "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
        "[지침]\n"
        "질문에 대한 답을 2단어 이내로 간단히 답하시오.\n\n"
        "[예시]\n"
        "질문: 조선 후기의 실학 사상가로 목민심서를 쓴 인물은?\n"
        "답변: 정약용"
    ),
    "교정형": (
        "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
        "[지침]\n"
        "주어진 문장이 올바른지 판단하고, 틀린 경우 올바르게 교정하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.\n\n"
        "[예시]\n"
        "질문: 다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n\"오늘은 퍼즐 마추기를 해 볼 거예요.\"\n"
        "답변: \"오늘은 퍼즐 맞추기를 해 볼 거예요.\"가 옳다. '제자리에 맞게 붙이다, 주문하다, 똑바르게 하다, 비교하다' 등의 뜻이 있는 말은 '마추다'가 아닌 '맞추다'로 적는다."
    ),
    "선택형": (
        "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
        "[지침]\n"
        "주어진 보기들 중에서 가장 적절한 것을 선택하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.\n\n"
        "[예시]\n"
        "질문: \"나는 그를 본 적이 있음을 {기억해냈다/기억해 냈다}.\" 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.\n"
        "답변: \"나는 그를 본 적이 있음을 기억해 냈다.\"가 옳다. '기억해 냈다'는 '기억하-+-아+냈다'의 구성이다. 이처럼 '본용언+-아/-어+보조 용언' 구성인 경우 본용언과 보조 용언을 붙여 쓰는 것이 허용되지만, 이러한 구성을 갖더라도 앞말이 3음절 이상의 합성어나 파생어라면 보조 용언을 붙여 쓰는 것이 허용되지 않는다. '기억하다'는 '기억'과 '-하다'가 결합한 파생어이며 '기억해'는 3음절이다. 따라서 '기억해'와 '냈다'는 띄어 써야 한다."
    )
}

"""
HUGGINGFACE TOKEN RESOLVER
"""
def get_token(path: Optional[str] = None) -> Optional[str]:
    with open(path, "r") as f:
        hf_token = f.read().strip()
    return hf_token

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
    token = get_token(args.hf_token)

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
    token = get_token(args.hf_token)

    model, tokenizer = load_base_model(args.model_name, load_in_4bit=True, token=token)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, token=token)
    model.eval()

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>") or tokenizer.convert_tokens_to_ids("<|endoftext|>")
    ]

    retriever = load_retriever(args.adapter_path or pathlib.Path(args.reference_path).parent)
    data = load_json(args.test_path)
    hf_logging.set_verbosity_error()

    with open(args.output_path, "w", encoding="utf-8") as outf:
        for sample in tqdm(data, desc="Generating", unit="sample", total=len(data)):
            q      = sample["input"]["question"]
            qtype  = sample["input"].get("type", "서술형")

            ctx  = retriever.query(q, k=5)
            ctx_block = "\n".join(f"- {c}" for c in ctx)
            inst = INST.get(qtype, INST["서술형"])
            prompt = (
                f"{PROMPT}\n\n"
                f"[참고 문헌]\n{ctx_block}\n\n"
                f"[질문 유형]\n{qtype}\n\n"
                f"[질문]\n{q}\n\n"
                f"{inst}\n\n"
                f"답변:"
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            inputs.pop("token_type_ids", None)

            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.05,
                temperature=0.7,
                top_p=0.8,
                stop_sequences=[tokenizer.encode("\n- ", add_special_tokens=False)[-1]]
            )

            text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

            if text.startswith("답변: "):
                text = text[4:]
            elif text.startswith("답변:"):
                text = text[3:]

            answer = postprocess(text)

            outf.write(json.dumps({
                "id": sample["id"],
                "input": sample["input"],
                "output": {"answer": answer}
            }, ensure_ascii=False) + "\n")

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