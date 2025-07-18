import argparse, json, os, pathlib, warnings, re
import faiss
from tqdm import tqdm
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.utils import logging as hf_logging
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

PROMPT = "You are a helpful AI assistant. Answer the user's questions based *only* on the provided reference material. Do not include information not present in the references. 당신은 주어진 참고 문헌만을 바탕으로 사용자의 질문에 답변하는 AI 어시스턴트입니다. 참고 문헌에 없는 내용은 답변에 포함하지 마시오."
INST = {
    "선다형": (
        "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
        "[지침]\n"
        "주어진 보기 중에서 가장 적절한 답을 숫자로만 응답하시오."
    ),
    "서술형": (
        "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
        "[지침]\n"
        "질문에 대한 답변을 완성된 문장으로 서술하시오."
    ),
    "단답형": (
        "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
        "[지침]\n"
        "질문에 대한 답을 2단어 이내로 간단히 답하시오."
    ),
    "교정형": (
        "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
        "[지침]\n"
        "주어진 문장이 올바른지 판단하고, 틀린 경우 올바르게 교정하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오."
    ),
    "선택형": (
        "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
        "[지침]\n"
        "주어진 보기들 중에서 가장 적절한 것을 선택하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오."
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

def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

"""
HYBRID RETRIEVAL + RERANKING
"""
class HybridRerankRetriever:
    def __init__(self, sentences: List[str], sbert_model: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS", reranker_model: str = "bongsoo/klue-cross-encoder-v1", batch: int = 32, embeddings: Optional[np.ndarray] = None, sent_tokens: Optional[List[List[str]]] = None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sentences = sentences
        
        self.bi_encoder = SentenceTransformer(sbert_model, device=device)
        self.cross_encoder = CrossEncoder(reranker_model, device=device)

        if embeddings is not None:
            self.embeddings = embeddings
        else:
            print(f"Encoding {len(self.sentences)} sentences...")
            self.embeddings = self.bi_encoder.encode(self.sentences, batch_size=batch, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
            self.embeddings = self.embeddings.astype(np.float32)
        
        if self.embeddings.shape[0] > 50000:
            nlist = min(100, int(np.sqrt(self.embeddings.shape[0])))
            quantizer = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index = faiss.IndexIVFFlat(quantizer, self.embeddings.shape[1], nlist)
            self.index.train(self.embeddings)
            self.index.add(self.embeddings)
            self.index.nprobe = min(10, nlist // 4)
        else:
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)
        
        if sent_tokens is not None:
            self.sent_tokens = sent_tokens
        else:
            self.sent_tokens = [ko_tokenize(s) for s in self.sentences]
        
        self.bm25 = BM25Okapi(self.sent_tokens)

    def query(self, question: str, top_k: int = 7, hybrid_k: int = 75, alpha: float = 0.5):
        q_tok = ko_tokenize(question)
        bm25_scores = self.bm25.get_scores(q_tok)
        
        q_emb = self.bi_encoder.encode([question], normalize_embeddings=True)
        sbert_scores = (q_emb @ self.embeddings.T)[0]
        
        norm_bm25 = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-6)
        norm_sbert = (sbert_scores - np.min(sbert_scores)) / (np.max(sbert_scores) - np.min(sbert_scores) + 1e-6)
        
        hybrid_scores = alpha * norm_bm25 + (1 - alpha) * norm_sbert
        
        hybrid_top_idx = np.argsort(hybrid_scores)[-hybrid_k:][::-1]
        hybrid_docs = [self.sentences[i] for i in hybrid_top_idx]
        
        cross_inputs = [[question, doc] for doc in hybrid_docs]
        cross_scores = self.cross_encoder.predict(cross_inputs, show_progress_bar=False)
        
        reranked_results = sorted(zip(cross_scores, hybrid_docs), key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in reranked_results[:top_k]]
    
    def query_batch(self, questions: List[str], top_k: int = 5, hybrid_k: int = 20, alpha: float = 0.5) -> List[List[str]]:
        if len(questions) > 16:
            results = []
            for i in range(0, len(questions), 16):
                batch = questions[i:i+16]
                batch_results = self.query_batch(batch, top_k, hybrid_k, alpha)
                results.extend(batch_results)
            return results
            
        bm25_scores_list = [self.bm25.get_scores(ko_tokenize(q)) for q in questions]

        q_embs = self.bi_encoder.encode(questions, normalize_embeddings=True, show_progress_bar=False, batch_size=16, convert_to_numpy=True).astype(np.float32)
        
        sbert_scores_batch, sbert_indices_batch = self.index.search(q_embs, min(hybrid_k * 2, self.embeddings.shape[0]))

        results = []
        for i, question in enumerate(questions):
            top_indices = sbert_indices_batch[i][:hybrid_k]
            bm25_scores = bm25_scores_list[i]
            sbert_scores = sbert_scores_batch[i][:hybrid_k]
            
            norm_bm25 = (bm25_scores[top_indices] - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-6)
            norm_sbert = (sbert_scores - np.min(sbert_scores)) / (np.max(sbert_scores) - np.min(sbert_scores) + 1e-6)
            
            hybrid_scores = alpha * norm_bm25 + (1 - alpha) * norm_sbert
            sorted_idx = np.argsort(hybrid_scores)[::-1]
            
            top_docs = [self.sentences[top_indices[idx]] for idx in sorted_idx]
            
            cross_inputs = [[question, doc] for doc in top_docs]
            cross_scores = self.cross_encoder.predict(cross_inputs, show_progress_bar=False, batch_size=8)
            reranked = sorted(zip(cross_scores, top_docs), key=lambda x: x[0], reverse=True)
            results.append([doc for _, doc in reranked[:top_k]])

        return results

"""
DATASET FOR RAG
"""
class KoreanRAGDataset(Dataset):
    def __init__(self, cached_items: List[Dict], tokenizer, max_len: int = 1536):
        self.items = cached_items
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        q = item["question"]
        qtype = item["type"]
        a = item["answer"]
        ctx_block = item["context"]
        inst = INST.get(qtype, INST["서술형"])

        # Use simple single-pass approach like RAG_8 to avoid contamination
        text = (
            f"{PROMPT}\n\n"
            f"[참고 문헌]\n{ctx_block}\n\n"
            f"[질문]\n{q}\n\n"
            f"{inst}\n\n답변: {a}"
        )

        tok = self.tokenizer(text, truncation=True, max_length=self.max_len, padding="max_length")
        tok["labels"] = tok["input_ids"].copy()
        return {k: torch.tensor(v) for k, v in tok.items()}

def preprocess_and_cache_dataset(data: List[Dict], retriever: HybridRerankRetriever, cache_path: str) -> List[Dict]:
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        return torch.load(cache_path)

    print(f"Preprocessing and caching data to {cache_path}...")
    questions = [item['input']['question'] for item in data]
    
    retrieved_contexts = retriever.query_batch(questions, top_k=5)

    cached_data = []
    for item, contexts in tqdm(zip(data, retrieved_contexts), total=len(data), desc="Caching"):
        ctx_block = "\n".join(f"- {c[:300]}" for c in contexts)
        cached_data.append({
            "question": item["input"]["question"],
            "type": item["input"].get("type", "서술형"),
            "answer": item["output"]["answer"],
            "context": ctx_block
        })

    torch.save(cached_data, cache_path)
    print("Caching complete.")
    return cached_data

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
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    model = AutoModelForCausalLM.from_pretrained(
        name, 
        device_map="auto", 
        trust_remote_code=True, 
        quantization_config=bnb_cfg, 
        token=token,
        torch_dtype=torch.bfloat16
    )
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, token=token)
    tok.pad_token = tok.eos_token
    return model, tok

"""
TRAINING
"""
def train(args):
    token = get_token(args.hf_token)

    reference_text = open(args.reference_path, encoding="utf-8").read()
    sentences = split_into_sentences(reference_text)
    retriever = HybridRerankRetriever(sentences)

    train_data = load_json(args.train_path)
    dev_data = load_json(args.dev_path) if args.dev_path else train_data[: max(1, len(train_data) // 10)]

    train_cache_path = os.path.join(args.output_dir, "train_cache.pt")
    dev_cache_path = os.path.join(args.output_dir, "dev_cache.pt")
    cached_train_data = preprocess_and_cache_dataset(train_data, retriever, train_cache_path)
    cached_dev_data = preprocess_and_cache_dataset(dev_data, retriever, dev_cache_path)

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

    train_ds = KoreanRAGDataset(cached_train_data, tokenizer)
    eval_ds = KoreanRAGDataset(cached_dev_data, tokenizer)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=2e-4,
        bf16=True,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=5,
        logging_steps=10,
        report_to="none",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=targs,
        train_dataset=train_ds, 
        eval_dataset=eval_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    torch.save({
        "sentences": retriever.sentences, 
        "embeddings": retriever.embeddings, 
        "sent_tokens": retriever.sent_tokens
    }, os.path.join(args.output_dir, "retriever.pt"))

"""
INFERENCE
"""
def load_retriever_for_inference(saved_dir: str) -> HybridRerankRetriever:
    obj = torch.load(os.path.join(saved_dir, "retriever.pt"), map_location="cpu")
    sentences = obj.get("sentences", obj.get("passages"))
    embeddings = obj["embeddings"]
    sent_tokens = obj.get("sent_tokens", obj.get("pass_tokens"))
    
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    ret = HybridRerankRetriever(sentences, embeddings=embeddings, sent_tokens=sent_tokens)
    return ret


def postprocess(text: str) -> str:
    text = text.strip()
    
    if "답변:" in text:
        text = text.split("답변:")[-1].strip()

    m = re.search(r'([""`])(.*?)\1\s*가\s*옳다', text)
    if m:
        quote_content = m.group(2).strip()
        corrected_sentence = f'\"{quote_content}\"가 옳다.'
        explanation = text[m.end():].strip(' .')
        if explanation:
            return f"{corrected_sentence} {explanation}"
        return corrected_sentence

    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        return "답변을 생성하지 못했습니다."
        
    return text

def predict(args):
    token = get_token(args.hf_token)

    model, tokenizer = load_base_model(args.model_name, load_in_4bit=True, token=token)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, token=token)
    model.eval()

    tokenizer.padding_side = "left"
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
        for i in tqdm(range(0, len(data), args.batch), desc="Generating", unit="batch"):
            batch_data = data[i:i+args.batch]
            batch_questions = [sample["input"]["question"] for sample in batch_data]
            
            batch_contexts = retriever.query_batch(batch_questions, top_k=5, hybrid_k=35, alpha=0.5)
            
            # Use simple single-pass approach matching training
            prompts = []
            for sample, ctx in zip(batch_data, batch_contexts):
                q = sample["input"]["question"]
                qtype = sample["input"].get("type", "서술형")
                ctx_block = "\n".join(f"- {c[:300]}" for c in ctx)
                inst = INST.get(qtype, INST["서술형"])
                prompts.append(
                    f"{PROMPT}\n\n"
                    f"[참고 문헌]\n{ctx_block}\n\n"
                    f"[질문]\n{q}\n\n"
                    f"{inst}\n\n답변:"
                )
            
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1536).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    eos_token_id=terminators,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.05,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    use_cache=True
                )
            
            texts = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

            for j, sample in enumerate(batch_data):
                final_answer = postprocess(texts[j])
                outf.write(json.dumps({
                    "id": sample["id"],
                    "input": sample["input"],
                    "output": {"answer": final_answer}
                }, ensure_ascii=False) + "\n")

    print(f"Saved predictions → {args.output_path}")

"""
MAIN FUNCTION 
"""
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["train", "predict"])
    p.add_argument("--model_name", default="yanolja/EEVE-Korean-Instruct-10.8B-v1.0")
    p.add_argument("--reference_path", default="reference.txt")
    p.add_argument("--hf_token", default="hf_token.txt")

    p.add_argument("--train_path")
    p.add_argument("--dev_path")
    p.add_argument("--output_dir", default="RAG12_ckpt")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_len", type=int, default=1536)

    p.add_argument("--test_path")
    p.add_argument("--adapter_path", default="RAG12_ckpt")
    p.add_argument("--output_path", default="submission_RAG12.jsonl")
    p.add_argument("--hybrid_k", type=int, default=35)

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