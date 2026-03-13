"""
Data ingestion pipeline: reads raw JSONL and normalizes into processed docs.
Usage: python -m app.ingest --input data/raw --out data/processed
"""
import argparse
import json
import os
import re
from datetime import datetime


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text


def process_doc(doc: dict) -> dict:
    return {
        "doc_id": doc.get("doc_id", ""),
        "title": clean_text(doc.get("title", "")),
        "text": clean_text(doc.get("text", "")),
        "source": doc.get("source", ""),
        "created_at": doc.get("created_at", datetime.utcnow().isoformat())
    }


def ingest(input_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "docs.jsonl")
    docs = []

    for fname in os.listdir(input_dir):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(input_dir, fname)
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    doc = process_doc(raw)
                    if doc["text"] and doc["doc_id"]:
                        docs.append(doc)
                except Exception as e:
                    print(f"  [warn] skipping bad line: {e}")

    with open(out_path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    print(f"Ingested {len(docs)} documents -> {out_path}")
    return docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()
    ingest(args.input, args.out)
