"""
Vector index: sentence-transformers embeddings + FAISS (CPU).
"""
import json
import os
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"
METADATA_FILE = "metadata.json"


class VectorIndex:
    def __init__(self, index_dir: str = "data/index/vector", model_name: str = DEFAULT_MODEL):
        self.index_dir = index_dir
        self.model_name = model_name
        self.model = None
        self.index = None
        self.doc_ids: List[str] = []
        self.docs: List[dict] = []

    def _load_model(self):
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def build(self, docs: List[dict]):
        self._load_model()
        self.docs = docs
        self.doc_ids = [d["doc_id"] for d in docs]
        texts = [f"{d['title']} {d['text']}" for d in docs]
        print(f"Encoding {len(texts)} documents...")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        embeddings = embeddings.astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.index_dir, "faiss.index"))
        with open(os.path.join(self.index_dir, "doc_ids.json"), "w") as f:
            json.dump(self.doc_ids, f)
        with open(os.path.join(self.index_dir, "docs.json"), "w") as f:
            json.dump(self.docs, f)
        with open(os.path.join(self.index_dir, METADATA_FILE), "w") as f:
            json.dump({"model_name": self.model_name, "dimension": dim, "num_docs": len(docs)}, f)
        print(f"Vector index built: dim={dim}, docs={len(docs)} -> {self.index_dir}")

    def load(self):
        meta_path = os.path.join(self.index_dir, METADATA_FILE)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Vector index metadata not found at {meta_path}")
        with open(meta_path) as f:
            meta = json.load(f)
        # Validate model name matches
        if meta["model_name"] != self.model_name:
            raise ValueError(
                f"Model mismatch: index built with '{meta['model_name']}', "
                f"but current model is '{self.model_name}'. Rebuild index."
            )
        self.index = faiss.read_index(os.path.join(self.index_dir, "faiss.index"))
        # Validate dimension
        if self.index.d != meta["dimension"]:
            raise ValueError(
                f"Dimension mismatch: index has dim={self.index.d}, "
                f"metadata says dim={meta['dimension']}. Rebuild index."
            )
        with open(os.path.join(self.index_dir, "doc_ids.json")) as f:
            self.doc_ids = json.load(f)
        with open(os.path.join(self.index_dir, "docs.json")) as f:
            self.docs = json.load(f)
        self._load_model()
        print(f"Vector index loaded: model={self.model_name}, dim={meta['dimension']}, docs={len(self.doc_ids)}")

    def query(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        self._load_model()
        vec = self.model.encode([text]).astype("float32")
        faiss.normalize_L2(vec)
        scores, indices = self.index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.doc_ids[idx], float(score)))
        return results

    def get_doc(self, doc_id: str) -> dict | None:
        for d in self.docs:
            if d["doc_id"] == doc_id:
                return d
        return None
