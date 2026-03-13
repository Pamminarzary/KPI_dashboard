# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────┐
│                    Frontend (React)                  │
│  Search │ KPI Dashboard │ Evaluation │ Debug Logs   │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP (localhost:8000)
┌──────────────────────▼──────────────────────────────┐
│                FastAPI Backend                       │
│  /health  /search  /metrics  /logs  /eval           │
└──────┬──────────────────────────────┬───────────────┘
       │                              │
┌──────▼──────┐              ┌────────▼────────┐
│ BM25 Index  │              │  Vector Index   │
│ rank-bm25   │              │  FAISS + SBERT  │
│ data/index/ │              │  data/index/    │
│ bm25/       │              │  vector/        │
└─────────────┘              └─────────────────┘
       │                              │
┌──────▼──────────────────────────────▼──────────────┐
│              HybridSearchEngine                      │
│  hybrid = alpha*norm(bm25) + (1-alpha)*norm(vector)  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                SQLite (data/search_logs.db)          │
│  query_logs: request_id, query, latency, results    │
└─────────────────────────────────────────────────────┘
```

## Data Flow

1. **Ingest**: `data/raw/*.jsonl` → cleaned JSONL → `data/processed/docs.jsonl`
2. **Index**: docs.jsonl → BM25 pickle + FAISS index + metadata JSON
3. **Query**: user query → parallel BM25 + vector search → min-max normalize → hybrid score → ranked results
4. **Log**: every request logged to SQLite with latency, alpha, result count
5. **Evaluate**: queries.jsonl + qrels.json → nDCG/Recall/MRR → experiments.csv
6. **Dashboard**: React polls /metrics/json, /logs, /eval/experiments every load

## Key Design Decisions

See `docs/decision_log.md` for full rationale on:
- Normalization strategy (min-max default)
- Embedding model selection (all-MiniLM-L6-v2)
- FAISS index type (IndexFlatIP)
- Alpha default (0.5)
- Storage (SQLite)
- Frontend (single HTML file)
