"""
HOW TO RUN

1. Install dependencies
chmod +x setup && ./setup

2. Run the model (train + predict)
chmod +x run && ./run

(Optional) Predict separately after training

chmod +x train && ./train
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
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

PROMPT = "You are a helpful AI assistant. Please answer the user's questions kindly based *only* on the provided reference material.             당신은 주어진 참고 문헌만을 바탕으로 사용자의 질문에 대해 친절하게 답변하는 AI 어시스턴트입니다. 참고 문헌에 없는 내용은 답변에 포함하지 마시오.             단, 동일한 문장을 절대 반복하지 마시오."
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
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        hf_token = f.read().strip()
    return hf_token

"""
TOKENIZER FOR BM25
"""
def ko_tokenize(text: str):
    return re.findall(r"[가-힣]+|[a-zA-Z0-9]+", text.lower())

"""
HYBRID RETRIEVAL + RERANKING
"""
class HybridRerankRetriever:
    def __init__(self, passages: List[str], sbert_model: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS", reranker_model: str = "bongsoo/klue-cross-encoder-v1", batch: int = 64):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.passages = passages
        
        # Bi-Encoder for retrieval
        self.bi_encoder = SentenceTransformer(sbert_model, device=device)
        self.embeddings = self.bi_encoder.encode(passages, batch_size=batch, show_progress_bar=True, normalize_embeddings=True)
        
        # BM25
        self.pass_tokens = [ko_tokenize(p) for p in self.passages]
        self.bm25 = BM25Okapi(self.pass_tokens)
        
        # Cross-Encoder for re-ranking
        self.cross_encoder = CrossEncoder(reranker_model, device=device)

    def query(self, question: str, top_k: int = 5, hybrid_k: int = 50, alpha: float = 0.6):
        # 1. Hybrid Retrieval (BM25 + SBERT)
        q_tok = ko_tokenize(question)
        bm25_scores = self.bm25.get_scores(q_tok)
        
        q_emb = self.bi_encoder.encode([question], normalize_embeddings=True)
        sbert_scores = (q_emb @ self.embeddings.T)[0]
        
        # Normalize scores (min-max scaling) to combine them
        norm_bm25 = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-6)
        norm_sbert = (sbert_scores - np.min(sbert_scores)) / (np.max(sbert_scores) - np.min(sbert_scores) + 1e-6)
        
        hybrid_scores = alpha * norm_bm25 + (1 - alpha) * norm_sbert
        
        hybrid_top_idx = np.argsort(hybrid_scores)[-hybrid_k:][::-1]
        hybrid_docs = [self.passages[i] for i in hybrid_top_idx]
        
        # 2. Re-ranking with Cross-Encoder
        cross_inputs = [[question, doc] for doc in hybrid_docs]
        cross_scores = self.cross_encoder.predict(cross_inputs, show_progress_bar=False)
        
        # Sort based on re-ranking scores
        reranked_results = sorted(zip(cross_scores, hybrid_docs), key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in reranked_results[:top_k]]

"""
DATASET FOR RAG
"""
class KoreanRAGDataset(Dataset):
    def __init__(self, items: List[Dict], retriever: HybridRerankRetriever, tokenizer, max_len: int = 2048):
        self.items = items
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def _prompt(self, question: str, qtype: str) -> str:
        q_emb = self.retriever.bi_encoder.encode([question], normalize_embeddings=True)
        scores = (q_emb @ self.retriever.embeddings.T)[0]
        top_idx = np.argsort(scores)[-5:][::-1]
        ctx = [self.retriever.passages[i] for i in top_idx]

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
    retriever = HybridRerankRetriever(passages)

    train_data = load_json(args.train_path)
    dev_data = load_json(args.dev_path) if args.dev_path else train_data[: max(1, len(train_data) // 10)]

    model, tokenizer = load_base_model(args.model_name, token=token)
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)
    
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
    # Save retriever components
    torch.save({
        "passages": retriever.passages, 
        "embeddings": retriever.embeddings, 
        "pass_tokens": retriever.pass_tokens
    }, os.path.join(args.output_dir, "retriever.pt"))

"""
INFERENCE
"""
def load_retriever_for_inference(saved_dir: str) -> HybridRerankRetriever:
    obj = torch.load(os.path.join(saved_dir, "retriever.pt"), map_location="cpu")
    ret = HybridRerankRetriever(obj["passages"])
    ret.embeddings = obj["embeddings"]
    ret.pass_tokens = obj["pass_tokens"]
    ret.bm25 = BM25Okapi(ret.pass_tokens)
    return ret


def postprocess(text: str) -> str:
    text = text.strip()
    if not text:
        return "답변을 생성하지 못했습니다."

    if re.match(r'.+가 옳다', text):
        return text
    m = re.search(r'[\"“”]([^\"”]+)[\"”]', text)
    if m:
        fix = f'{m.group(1)}가 옳다.'
        rest = text[m.end():].lstrip(' .')
        if rest:
            return f"{fix} {rest}"
        return fix

    if text.startswith("답변:"):
        text = text[3:].lstrip()

    return text


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
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    retriever = load_retriever_for_inference(args.adapter_path or pathlib.Path(args.reference_path).parent)
    data = load_json(args.test_path)
    hf_logging.set_verbosity_error()

    with open(args.output_path, "w", encoding="utf-8") as outf:
        for sample in tqdm(data, desc="Generating", unit="sample", total=len(data)):
            q      = sample["input"]["question"]
            qtype  = sample["input"].get("type", "서술형")

            ctx  = retriever.query(q, top_k=5, hybrid_k=50)
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
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.05,
                temperature=0.7,
                top_p=0.9
            )

            text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
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
    p.add_argument("--model_name", default="beomi/Llama-3-Open-Ko-8B-Instruct-preview")
    p.add_argument("--reference_path", required=True)
    p.add_argument("--hf_token", default="hf_token.txt")

    p.add_argument("--train_path")
    p.add_argument("--dev_path")
    p.add_argument("--output_dir", default="RAG7_ckpt")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--grad_accum", type=int, default=8)

    p.add_argument("--test_path")
    p.add_argument("--adapter_path")
    p.add_argument("--output_path", default="submission_RAG7.jsonl")

    args = p.parse_args()
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.adapter_path:
        os.makedirs(args.adapter_path, exist_ok=True)

    if args.mode == "train":
        if not args.train_path:
            raise ValueError("Training mode requires --train_path")
        train(args)
        if args.test_path:
            print("\nTraining complete — starting inference …\n")
            args.mode = "predict"
            args.adapter_path = args.output_dir
            predict(args)
        else:
            print("\nNo --test_path supplied, so inference was skipped.")
    else:
        if not args.test_path:
            raise ValueError("Prediction mode requires --test_path")
        if not args.adapter_path and not os.path.exists(os.path.join(pathlib.Path(args.reference_path).parent, "retriever.pt")):
             raise ValueError("Prediction mode requires a trained adapter or a pre-built retriever.pt file.")
        predict(args)

if __name__ == "__main__":
    main()
