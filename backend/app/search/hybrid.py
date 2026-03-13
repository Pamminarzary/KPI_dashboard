"""
Hybrid search: combines BM25 + vector scores with configurable alpha.
Normalization strategies: min-max and z-score.
"""
import re
from typing import Dict, List, Optional

from app.search.bm25 import BM25Index
from app.search.vector import VectorIndex


def minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    rng = mx - mn
    if rng == 0:
        return {k: 0.0 for k in scores}
    return {k: (v - mn) / rng for k, v in scores.items()}


def zscore_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    import numpy as np
    vals = np.array(list(scores.values()), dtype=float)
    mean, std = vals.mean(), vals.std()
    if std == 0:
        return {k: 0.0 for k in scores}
    normalized = (vals - mean) / std
    return {k: float(v) for k, v in zip(scores.keys(), normalized)}


def get_snippet(text: str, query: str, window: int = 150) -> str:
    lower = text.lower()
    words = query.lower().split()
    best_pos = 0
    for w in words:
        pos = lower.find(w)
        if pos != -1:
            best_pos = pos
            break
    start = max(0, best_pos - window // 2)
    end = min(len(text), start + window)
    snippet = text[start:end].strip()
    for w in words:
        snippet = re.sub(f"(?i)({re.escape(w)})", r"**\1**", snippet)
    return f"...{snippet}..."


class HybridSearchEngine:
    def __init__(self, bm25_index: BM25Index, vector_index: VectorIndex, normalization: str = "minmax"):
        self.bm25 = bm25_index
        self.vector = vector_index
        self.normalization = normalization

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        fetch_k = top_k * 3

        bm25_raw = dict(self.bm25.query(query, top_k=fetch_k))
        vector_raw = dict(self.vector.query(query, top_k=fetch_k))

        normalize = minmax_normalize if self.normalization == "minmax" else zscore_normalize
        bm25_norm = normalize(bm25_raw)
        vector_norm = normalize(vector_raw)

        all_ids = set(bm25_norm) | set(vector_norm)
        results = []
        for doc_id in all_ids:
            b = bm25_norm.get(doc_id, 0.0)
            v = vector_norm.get(doc_id, 0.0)
            hybrid = alpha * b + (1 - alpha) * v
            doc = self.bm25.get_doc(doc_id) or self.vector.get_doc(doc_id)
            if not doc:
                continue
            results.append({
                "doc_id": doc_id,
                "title": doc.get("title", ""),
                "bm25_score": round(bm25_raw.get(doc_id, 0.0), 4),
                "vector_score": round(vector_raw.get(doc_id, 0.0), 4),
                "bm25_norm": round(b, 4),
                "vector_norm": round(v, 4),
                "hybrid_score": round(hybrid, 4),
                "snippet": get_snippet(doc.get("text", ""), query),
                "source": doc.get("source", ""),
            })

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:top_k]
