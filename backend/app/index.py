"""
Build BM25 and vector indexes from processed docs.
Usage: python -m app.index --input data/processed/docs.jsonl
"""
import argparse
import json
import os

from app.search.bm25 import BM25Index
from app.search.vector import VectorIndex


def build_indexes(input_path: str, model_name: str = "all-MiniLM-L6-v2"):
    docs = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    print(f"Loaded {len(docs)} documents from {input_path}")

    print("\n--- Building BM25 Index ---")
    bm25 = BM25Index()
    bm25.build(docs)

    print("\n--- Building Vector Index ---")
    vector = VectorIndex(model_name=model_name)
    vector.build(docs)

    print("\nAll indexes built successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/docs.jsonl")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()
    build_indexes(args.input, args.model)
