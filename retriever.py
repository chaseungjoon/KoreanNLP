"""
retriever.py
------------
Sentence-BERT + FAISS 기반 간단한 Retriever
"""

from typing import List, Tuple
import faiss                           # pip install faiss-cpu
from sentence_transformers import SentenceTransformer
import numpy as np


class Retriever:
    """Sentence-BERT 임베딩 + FAISS Inner-Product Index"""

    def __init__(
        self,
        model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        normalize: bool = True,
    ):
        self.embed_model = SentenceTransformer(model_name)
        self.normalize = normalize
        self.corpus: List[str] = []
        self.index: faiss.IndexFlatIP | None = None

    # -------------------------------------------------- #
    #                     Public API                     #
    # -------------------------------------------------- #
    def add_corpus(self, passages: List[str]) -> None:
        """말뭉치(문단 리스트) 저장"""
        self.corpus = passages

    def build(self) -> None:
        """FAISS index 구축 (내적 기반)"""
        if not self.corpus:
            raise ValueError("빈 말뭉치입니다. add_corpus() 먼저 호출하세요.")

        embeddings = self.embed_model.encode(
            self.corpus, convert_to_numpy=True, show_progress_bar=True
        )

        if self.normalize:
            faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """query → 상위 k개의 (문단, 점수) 반환"""
        if self.index is None:
            raise ValueError("Index가 없습니다. build()를 먼저 호출하세요.")

        q_emb = self.embed_model.encode([query], convert_to_numpy=True)
        if self.normalize:
            faiss.normalize_L2(q_emb)

        scores, ids = self.index.search(q_emb, k)
        return [(self.corpus[i], float(scores[0][rank])) for rank, i in enumerate(ids[0])]