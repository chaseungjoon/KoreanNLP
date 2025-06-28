"""
evaluate.py
-----------
Exact Match · ROUGE-L · BERTScore · BLEURT 계산 유틸
"""

from typing import List, Dict
import re
import numpy as np
import evaluate as hf_eval          # pip install evaluate


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def exact_match(preds: List[str], refs: List[str]) -> float:
    return np.mean([_normalize(p) == _normalize(r) for p, r in zip(preds, refs)])


def rouge_l(preds: List[str], refs: List[str]) -> float:
    rouge = hf_eval.load("rouge")
    return rouge.compute(predictions=preds, references=refs)["rougeL"]


def bert_score(preds: List[str], refs: List[str]) -> float:
    scorer = hf_eval.load("bertscore")
    return np.mean(
        scorer.compute(predictions=preds, references=refs, lang="ko")["f1"]
    )


def bleurt_score(preds: List[str], refs: List[str]) -> float:
    bleurt = hf_eval.load("bleurt", module_type="metric")
    return np.mean(bleurt.compute(predictions=preds, references=refs)["scores"])


def aggregate_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    return {
        "EM": exact_match(preds, refs),
        "ROUGE-L": rouge_l(preds, refs),
        "BERTScore": bert_score(preds, refs),
        "BLEURT": bleurt_score(preds, refs),
    }