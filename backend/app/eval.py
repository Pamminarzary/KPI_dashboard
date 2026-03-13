"""
Evaluation harness: computes nDCG@10, Recall@10, MRR@10.
Usage: python -m app.eval --queries data/eval/queries.jsonl --qrels data/eval/qrels.json
"""
import argparse
import csv
import json
import math
import os
import subprocess
from datetime import datetime

from app.search.bm25 import BM25Index
from app.search.vector import VectorIndex
from app.search.hybrid import HybridSearchEngine


def dcg(relevances: list, k: int) -> float:
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += rel / math.log2(i + 2)
    return score


def ndcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    rels = [1 if doc_id in relevant else 0 for doc_id in retrieved[:k]]
    ideal = sorted(rels, reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg(rels, k) / idcg


def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / len(relevant)


def mrr_at_k(retrieved: list, relevant: set, k: int) -> float:
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def get_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def run_eval(queries_path: str, qrels_path: str, alpha: float = 0.5, normalization: str = "minmax", model: str = "all-MiniLM-L6-v2"):
    bm25 = BM25Index()
    bm25.load()
    vector = VectorIndex(model_name=model)
    vector.load()
    engine = HybridSearchEngine(bm25, vector, normalization=normalization)

    with open(queries_path) as f:
        queries = [json.loads(l) for l in f if l.strip()]
    with open(qrels_path) as f:
        qrels = json.load(f)

    ndcg_scores, recall_scores, mrr_scores = [], [], []
    K = 10
    for q in queries:
        qid = q["query_id"]
        text = q["text"]
        relevant = set(qrels.get(qid, []))
        results = engine.search(text, top_k=K, alpha=alpha)
        retrieved = [r["doc_id"] for r in results]
        ndcg_scores.append(ndcg_at_k(retrieved, relevant, K))
        recall_scores.append(recall_at_k(retrieved, relevant, K))
        mrr_scores.append(mrr_at_k(retrieved, relevant, K))

    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_mrr = sum(mrr_scores) / len(mrr_scores)

    print(f"\n=== Evaluation Results ===")
    print(f"Queries: {len(queries)} | alpha={alpha} | norm={normalization} | model={model}")
    print(f"nDCG@{K}:   {avg_ndcg:.4f}")
    print(f"Recall@{K}: {avg_recall:.4f}")
    print(f"MRR@{K}:    {avg_mrr:.4f}")

    # Append to experiments.csv
    os.makedirs("data/metrics", exist_ok=True)
    csv_path = "data/metrics/experiments.csv"
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "commit", "alpha", "normalization", "model", "ndcg_at_10", "recall_at_10", "mrr_at_10"])
        writer.writerow([
            datetime.utcnow().isoformat(), get_commit(), alpha, normalization, model,
            round(avg_ndcg, 4), round(avg_recall, 4), round(avg_mrr, 4)
        ])
    print(f"Results appended to {csv_path}")
    return {"ndcg_at_10": avg_ndcg, "recall_at_10": avg_recall, "mrr_at_10": avg_mrr}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="data/eval/queries.jsonl")
    parser.add_argument("--qrels", default="data/eval/qrels.json")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--normalization", default="minmax")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()
    run_eval(args.queries, args.qrels, args.alpha, args.normalization, args.model)
