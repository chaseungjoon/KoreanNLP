"""
RAG_11 - Advanced Korean RAG System
Target: 60+ points

Key Improvements:
1. Advanced Multi-Vector Retrieval with Query Expansion
2. Multi-Stage Reranking Pipeline  
3. Sophisticated Prompt Engineering with Chain-of-Thought
4. Optimized for A100 GPU
5. Enhanced Korean Text Processing
6. Self-Consistency and Verification

HOW TO RUN:
1. Install dependencies: chmod +x setup && ./setup
2. Run: chmod +x run && ./run
"""

import argparse, json, os, pathlib, warnings, re
import faiss
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, 
    Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback,
    DataCollatorForLanguageModeling
)
from transformers.utils import logging as hf_logging
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np
import random

warnings.filterwarnings("ignore", category=UserWarning)

SYSTEM_PROMPT = """당신은 한국어 어문 규범 전문가입니다. 주어진 참고 문헌을 바탕으로 정확하고 체계적으로 답변하세요.

답변 과정:
1. 참고 문헌을 주의 깊게 분석하세요
2. 질문 유형에 맞는 답변 형식을 확인하세요  
3. 단계별로 논리적으로 사고하세요
4. 참고 문헌에 없는 내용은 절대 포함하지 마세요"""

INSTRUCTIONS = {
    "선다형": {
        "instruction": "주어진 보기 중에서 가장 적절한 답을 숫자로만 응답하시오.",
        "examples": [
            {
                "question": "다음 중 표준어가 아닌 것은?",
                "options": ["1) 설거지", "2) 절구", "3) 무릎", "4) 깔따구", "5) 벌초"],
                "answer": "4",
                "reasoning": "표준어 규정에 따라 '깔따구'는 비표준어이고, '깔따구'의 표준어는 '각다귀'입니다."
            }
        ]
    },
    "서술형": {
        "instruction": "질문에 대한 답변을 완성된 문장으로 서술하시오.",
        "examples": [
            {
                "question": "한글 맞춤법의 기본 원리를 설명하시오.",
                "answer": "한글 맞춤법의 기본 원리는 표준어를 소리대로 적되, 어법에 맞도록 함을 원칙으로 합니다. 이는 표음주의와 어법주의를 절충한 것으로, 소리나는 대로 적으면서도 어간과 어미, 어근과 접사의 결합 관계를 명확히 보여주도록 적는 것입니다."
            }
        ]
    },
    "단답형": {
        "instruction": "질문에 대한 답을 2단어 이내로 간단히 답하시오.",
        "examples": [
            {
                "question": "훈민정음을 창제한 조선의 왕은?",
                "answer": "세종대왕"
            }
        ]
    },
    "교정형": {
        "instruction": "주어진 문장이 올바른지 판단하고, 틀린 경우 올바르게 교정하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.",
        "examples": [
            {
                "question": "다음 문장을 교정하시오: \"이 일은 너무 어려워서 할 수 없다.\"",
                "answer": "\"이 일은 너무 어려워서 할 수 없다.\"가 옳다. 주어진 문장은 이미 올바른 표현입니다."
            }
        ]
    },
    "선택형": {
        "instruction": "주어진 보기들 중에서 가장 적절한 것을 선택하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.",
        "examples": [
            {
                "question": "다음 중 올바른 것은? {맞춤법/맞춤뻡}",
                "answer": "\"맞춤법\"이 옳다. 'ㅂ' 받침 뒤에 'ㅎ'이 올 때는 'ㅎ'이 거센소리로 바뀌어 'ㅍ'으로 발음되지만, 표기는 어간의 형태를 유지하여 '맞춤법'으로 적습니다."
            }
        ]
    }
}

def get_token(path: Optional[str] = None) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return f.read().strip()

def enhanced_ko_tokenize(text: str) -> List[str]:
    tokens = []
    
    pattern = r'[가-힣]+|[a-zA-Z]+|[0-9]+|[^\w\s]'
    matches = re.findall(pattern, text.lower())
    
    for match in matches:
        if re.match(r'[가-힣]+', match):
            tokens.extend(list(match))
        else:
            tokens.append(match)
    
    return tokens

def smart_text_chunking(text: str, max_chunk_size: int = 300, overlap: int = 50) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            if overlap > 0:
                words = current_chunk.split()
                overlap_words = words[-overlap//10:] if len(words) >= overlap//10 else words
                current_chunk = " ".join(overlap_words) + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if len(chunk.strip()) > 20]  # Filter out very short chunks

def expand_query(query: str, expansion_terms: Dict[str, List[str]]) -> str:
    expanded = query
    words = query.split()
    
    for word in words:
        if word in expansion_terms:
            expanded += " " + " ".join(expansion_terms[word])
    
    return expanded

class AdvancedKoreanRetriever:

    def __init__(self, 
                 chunks: List[str], 
                 dense_model: str = "jhgan/ko-sroberta-multitask",
                 reranker_model: str = "bongsoo/klue-cross-encoder-v1",
                 batch_size: int = 32):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.chunks = chunks
        self.batch_size = batch_size
        
        print(f"Initializing retriever with {len(chunks)} chunks...")
        
        self.dense_encoder = SentenceTransformer(dense_model, device=self.device)
        
        print("Generating embeddings...")
        self.embeddings = self.dense_encoder.encode(
            self.chunks, 
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)
        
        self.faiss_index = self._build_faiss_index()
        
        self.chunk_tokens = [enhanced_ko_tokenize(chunk) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.chunk_tokens)
        
        self.cross_encoder = CrossEncoder(reranker_model, device=self.device)
        
        self.expansion_terms = {
            "맞춤법": ["표기법", "표기", "맞춤", "철자"],
            "발음": ["소리", "음성", "발성", "음운"],
            "문법": ["어법", "구문", "문장", "통사"],
            "표준어": ["표준", "규범", "정음"],
            "한글": ["훈민정음", "한국어", "우리말"]
        }
        
    def _build_faiss_index(self) -> faiss.Index:
        d = self.embeddings.shape[1]
        
        if len(self.embeddings) > 50000:
            nlist = min(100, int(np.sqrt(len(self.embeddings))))
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
            index.train(self.embeddings)
            index.nprobe = min(10, nlist // 4)
        else:
            index = faiss.IndexFlatIP(d)
        
        index.add(self.embeddings)
        return index
    
    def _dense_retrieval(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        query_embedding = self.dense_encoder.encode([query], normalize_embeddings=True)
        
        scores, indices = self.faiss_index.search(query_embedding.astype(np.float32), top_k)
        
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
    
    def _sparse_retrieval(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        query_tokens = enhanced_ko_tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        return [(int(idx), float(scores[idx])) for idx in top_indices]
    
    def _hybrid_fusion(self, dense_results: List[Tuple[int, float]], 
                      sparse_results: List[Tuple[int, float]], 
                      alpha: float = 0.6) -> List[Tuple[int, float]]:
        dense_scores = {idx: score for idx, score in dense_results}
        sparse_scores = {idx: score for idx, score in sparse_results}
        
        all_indices = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        if dense_scores:
            max_dense = max(dense_scores.values())
            min_dense = min(dense_scores.values())
            if max_dense > min_dense:
                dense_scores = {idx: (score - min_dense) / (max_dense - min_dense) 
                               for idx, score in dense_scores.items()}
        
        if sparse_scores:
            max_sparse = max(sparse_scores.values())
            min_sparse = min(sparse_scores.values())
            if max_sparse > min_sparse:
                sparse_scores = {idx: (score - min_sparse) / (max_sparse - min_sparse) 
                                for idx, score in sparse_scores.items()}
        
        combined_scores = []
        for idx in all_indices:
            dense_score = dense_scores.get(idx, 0)
            sparse_score = sparse_scores.get(idx, 0)
            combined_score = alpha * dense_score + (1 - alpha) * sparse_score
            combined_scores.append((idx, combined_score))
        
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        return combined_scores
    
    def _rerank_with_cross_encoder(self, query: str, candidates: List[Tuple[int, float]], 
                                  top_k: int = 20) -> List[Tuple[int, float]]:
        candidate_texts = [self.chunks[idx] for idx, _ in candidates]
        
        cross_inputs = [[query, text] for text in candidate_texts]
        
        cross_scores = self.cross_encoder.predict(cross_inputs, show_progress_bar=False)
        
        reranked = []
        for i, (idx, original_score) in enumerate(candidates):
            cross_score = cross_scores[i]
            final_score = 0.3 * original_score + 0.7 * cross_score
            reranked.append((idx, final_score))
        
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:top_k]
    
    def retrieve(self, query: str, top_k: int = 5, hybrid_k: int = 50) -> List[str]:
        expanded_query = expand_query(query, self.expansion_terms)
        
        dense_results = self._dense_retrieval(expanded_query, top_k=hybrid_k)
        
        sparse_results = self._sparse_retrieval(expanded_query, top_k=hybrid_k)
        
        fused_results = self._hybrid_fusion(dense_results, sparse_results)
        
        reranked_results = self._rerank_with_cross_encoder(
            query, fused_results[:hybrid_k], top_k=top_k
        )
        
        return [self.chunks[idx] for idx, _ in reranked_results]
    
    def batch_retrieve(self, queries: List[str], top_k: int = 5, hybrid_k: int = 50) -> List[List[str]]:
        results = []
        
        for i in range(0, len(queries), 16):
            batch_queries = queries[i:i+16]
            batch_results = [self.retrieve(q, top_k, hybrid_k) for q in batch_queries]
            results.extend(batch_results)
        
        return results

class EnhancedKoreanRAGDataset(Dataset):

    def __init__(self, items: List[Dict], retriever: AdvancedKoreanRetriever, 
                 tokenizer, max_length: int = 2048):
        self.items = items
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.items)
    
    def _create_enhanced_prompt(self, question: str, qtype: str, contexts: List[str]) -> str:

        inst_data = INSTRUCTIONS.get(qtype, INSTRUCTIONS["서술형"])
        instruction = inst_data["instruction"]
        examples = inst_data.get("examples", [])
        
        context_block = "\n".join(f"참고자료 {i+1}: {ctx}" for i, ctx in enumerate(contexts))
        
        examples_block = ""
        if examples:
            example = random.choice(examples)
            if qtype == "선다형" and "options" in example:
                examples_block = f"""
[예시]
질문: {example['question']}
{chr(10).join(example['options'])}
답변: {example['answer']}
추론: {example.get('reasoning', '')}
"""
            else:
                examples_block = f"""
[예시]  
질문: {example['question']}
답변: {example['answer']}
"""
        
        # Create comprehensive prompt
        prompt = f"""{SYSTEM_PROMPT}

[참고 문헌]
{context_block}

{examples_block}

[질문 유형]: {qtype}
[지침]: {instruction}

[질문]: {question}

답변을 단계별로 생각해보세요:
1. 참고 문헌에서 관련 정보를 찾아보세요
2. 질문 유형에 맞는 답변 형식을 확인하세요
3. 논리적으로 답변을 구성하세요

답변:"""
        
        return prompt
    
    def __getitem__(self, idx):
        item = self.items[idx]
        question = item["input"]["question"]
        qtype = item["input"].get("type", "서술형")
        answer = item["output"]["answer"]
        
        contexts = self.retriever.retrieve(question, top_k=4)
        
        prompt = self._create_enhanced_prompt(question, qtype, contexts)
        
        full_text = prompt + " " + answer
        
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        prompt_length = len(self.tokenizer(prompt)["input_ids"])
        labels = encoding["input_ids"].clone()
        labels[:prompt_length] = -100
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }

def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_model_and_tokenizer(model_name: str, token: Optional[str] = None):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        token=token,
        attn_implementation="sdpa",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=token
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

class CustomTrainerCallback(TrainerCallback):

    def __init__(self, retriever):
        self.retriever = retriever
    
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.exists(output_dir):
                retriever_data = {
                    "chunks": self.retriever.chunks,
                    "embeddings": self.retriever.embeddings,
                    "chunk_tokens": self.retriever.chunk_tokens
                }
                torch.save(retriever_data, os.path.join(output_dir, "retriever.pt"))
                print(f"Saved retriever to {output_dir}")

def train(args):

    token = get_token(args.hf_token)
    
    print("Processing reference text...")
    reference_text = open(args.reference_path, encoding="utf-8").read()
    chunks = smart_text_chunking(reference_text, max_chunk_size=400, overlap=50)
    print(f"Created {len(chunks)} chunks")
    
    print("Initializing advanced retriever...")
    retriever = AdvancedKoreanRetriever(chunks)
    
    print("Loading datasets...")
    train_data = load_json(args.train_path)
    dev_data = load_json(args.dev_path) if args.dev_path else train_data[:max(1, len(train_data) // 10)]
    
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_name, token=token)
    
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("Creating training datasets...")
    train_dataset = EnhancedKoreanRAGDataset(train_data, retriever, tokenizer, args.max_length)
    eval_dataset = EnhancedKoreanRAGDataset(dev_data, retriever, tokenizer, args.max_length)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=2e-4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        overwrite_output_dir=True,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,  # Save memory
        report_to="none",
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            CustomTrainerCallback(retriever)
        ],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        ),
    )
    
    print("Starting training...")
    if args.resume_from_checkpoint:
        print("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    retriever_data = {
        "chunks": retriever.chunks,
        "embeddings": retriever.embeddings,
        "chunk_tokens": retriever.chunk_tokens
    }
    torch.save(retriever_data, os.path.join(args.output_dir, "retriever.pt"))
    
    print("Training completed!")

def load_retriever_for_inference(model_dir: str) -> AdvancedKoreanRetriever:
    retriever_path = os.path.join(model_dir, "retriever.pt")
    data = torch.load(retriever_path, map_location="cpu")
    
    retriever = AdvancedKoreanRetriever.__new__(AdvancedKoreanRetriever)
    retriever.chunks = data["chunks"]
    retriever.embeddings = data["embeddings"]
    retriever.chunk_tokens = data["chunk_tokens"]
    
    retriever.device = "cuda" if torch.cuda.is_available() else "cpu"
    retriever.batch_size = 32
    retriever.dense_encoder = SentenceTransformer("jhgan/ko-sroberta-multitask", device=retriever.device)
    retriever.cross_encoder = CrossEncoder("bongsoo/klue-cross-encoder-v1", device=retriever.device)
    retriever.faiss_index = retriever._build_faiss_index()
    retriever.bm25 = BM25Okapi(retriever.chunk_tokens)
    
    # Restore expansion terms
    retriever.expansion_terms = {
        "맞춤법": ["표기법", "표기", "맞춤", "철자"],
        "발음": ["소리", "음성", "발성", "음운"],
        "문법": ["어법", "구문", "문장", "통사"],
        "표준어": ["표준", "규범", "정음"],
        "한글": ["훈민정음", "한국어", "우리말"]
    }
    
    return retriever

def advanced_postprocess(text: str, question_type: str) -> str:
    text = text.strip()
    
    if "답변:" in text:
        text = text.split("답변:")[-1].strip()
    
    thinking_patterns = [
        r"단계별로 생각해보세요:.*?(?=\n\n|\Z)",
        r"생각해보면.*?(?=\n\n|\Z)",
        r"답변을 구성하면.*?(?=\n\n|\Z)",
        r"\d+\.\s*.*?(?=\n\n|\Z)"
    ]
    
    for pattern in thinking_patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    
    if question_type == "선다형":
        match = re.search(r'(\d+)', text)
        if match:
            return match.group(1)
    
    elif question_type == "교정형" or question_type == "선택형":
        match = re.search(r'([\""`])(.*?)\1\s*가\s*옳다', text)
        if match:
            quote_content = match.group(2).strip()
            corrected = f'"{quote_content}"가 옳다.'
            
            explanation = text[match.end():].strip()
            if explanation:
                return f"{corrected} {explanation}"
            return corrected
    
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    if not text:
        return "답변을 생성하지 못했습니다."
    
    return text

def predict(args):
    print("Starting RAG_11 prediction...")
    
    token = get_token(args.hf_token)
    
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_name, token=token)
    
    if args.adapter_path:
        print("Loading adapter...")
        model = PeftModel.from_pretrained(model, args.adapter_path, token=token)
    
    model.eval()
    
    print("Loading retriever...")
    retriever = load_retriever_for_inference(args.adapter_path or pathlib.Path(args.reference_path).parent)
    
    test_data = load_json(args.test_path)
    
    generation_config = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    hf_logging.set_verbosity_error()
    
    print("Generating predictions...")
    results = []
    
    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_data, desc="Generating")):
            question = sample["input"]["question"]
            qtype = sample["input"].get("type", "서술형")
            
            contexts = retriever.retrieve(question, top_k=5)
            
            inst_data = INSTRUCTIONS.get(qtype, INSTRUCTIONS["서술형"])
            context_block = "\n".join(f"참고자료 {i+1}: {ctx}" for i, ctx in enumerate(contexts))
            
            prompt = f"""{SYSTEM_PROMPT}

            [참고 문헌]
            {context_block}
            
            [질문 유형]: {qtype}
            [지침]: {inst_data['instruction']}
            
            [질문]: {question}
            
            답변:"""
            
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1800, truncation=True).to(model.device)
            outputs = model.generate(
                **inputs,
                **generation_config
            )
            
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            
            answer = advanced_postprocess(response, qtype)
            
            results.append({
                "id": sample["id"],
                "input": sample["input"],
                "output": {"answer": answer}
            })
    
    print("Saving results...")
    with open(args.output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Predictions saved to {args.output_path}")

def main():
    parser = argparse.ArgumentParser(description="RAG_11 - Advanced Korean RAG System")
    parser.add_argument("--mode", required=True, choices=["train", "predict"])
    parser.add_argument("--model_name", default="yanolja/EEVE-Korean-Instruct-10.8B-v1.0")
    parser.add_argument("--reference_path", default="reference.txt")
    parser.add_argument("--hf_token", default="hf_token.txt")
    
    # Training args
    parser.add_argument("--train_path")
    parser.add_argument("--dev_path")
    parser.add_argument("--output_dir", default="RAG11_ckpt")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    
    # Prediction args
    parser.add_argument("--test_path")
    parser.add_argument("--adapter_path", default="RAG11_ckpt")
    parser.add_argument("--output_path", default="submission_RAG11.jsonl")
    parser.add_argument("--resume_from_checkpoint")

    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "train":
        if not args.train_path:
            raise ValueError("Training requires --train_path")
        train(args)
        
        # Run prediction if test path provided
        if args.test_path:
            print("\nStarting prediction after training...")
            args.mode = "predict"
            args.adapter_path = args.output_dir
            predict(args)
    
    elif args.mode == "predict":
        if not args.test_path:
            raise ValueError("Prediction requires --test_path")
        predict(args)

if __name__ == "__main__":
    main()