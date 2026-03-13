"""
BM25 index: build and query using rank-bm25.
"""
import json
import os
import pickle
from typing import List, Tuple

from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self, index_dir: str = "data/index/bm25"):
        self.index_dir = index_dir
        self.bm25 = None
        self.doc_ids: List[str] = []
        self.docs: List[dict] = []

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def build(self, docs: List[dict]):
        self.docs = docs
        self.doc_ids = [d["doc_id"] for d in docs]
        corpus = [self._tokenize(f"{d['title']} {d['text']}") for d in docs]
        self.bm25 = BM25Okapi(corpus)
        os.makedirs(self.index_dir, exist_ok=True)
        with open(os.path.join(self.index_dir, "bm25.pkl"), "wb") as f:
            pickle.dump({"bm25": self.bm25, "doc_ids": self.doc_ids, "docs": self.docs}, f)
        print(f"BM25 index built with {len(docs)} docs -> {self.index_dir}")

    def load(self):
        path = os.path.join(self.index_dir, "bm25.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"BM25 index not found at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.doc_ids = data["doc_ids"]
        self.docs = data["docs"]

    def query(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        tokens = self._tokenize(text)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.doc_ids[i], float(s)) for i, s in ranked]

    def get_doc(self, doc_id: str) -> dict | None:
        for d in self.docs:
            if d["doc_id"] == doc_id:
                return d
        return None
