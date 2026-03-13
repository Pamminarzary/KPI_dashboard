# 🔎 Knowledge Search + KPI Dashboard

A hybrid search engine (BM25 + semantic vectors) with a real-time KPI dashboard.  
Built with FastAPI, sentence-transformers, FAISS, and React.

---

## ⚡ Quickstart (< 5 minutes)

```bash
git clone https://github.com/Pamminarzary/KPI_dashboard.git
cd knowledge-search-dashboard
./up.sh
```

Then open **http://localhost:8000** in your browser.

---

## Architecture

```
User Query
    │
    ▼
FastAPI /search
    ├── BM25Index (rank-bm25)        → lexical score
    └── VectorIndex (FAISS + SBERT)  → semantic score
         │
         ▼
    HybridSearchEngine
    hybrid = alpha * norm(bm25) + (1-alpha) * norm(vector)
         │
         ▼
    SQLite logs → /metrics/json → React Dashboard
```

## Components

| Component | Tech | Purpose |
|-----------|------|---------|
| Ingestion | Python | Download + clean Wikipedia corpus |
| BM25 Index | rank-bm25 + pickle | Lexical keyword search |
| Vector Index | sentence-transformers + FAISS | Semantic similarity search |
| Hybrid Engine | Custom Python | Score fusion with alpha blending |
| API | FastAPI + Uvicorn | REST endpoints with rate limiting |
| Logging | SQLite | Structured query logs + metrics |
| Dashboard | React (single HTML) | Search UI + KPIs + Eval + Debug |
| Evaluation | Custom Python | nDCG@10, Recall@10, MRR@10 |

---

## How to Run

### Full system
```bash
./up.sh
```

### Individual steps
```bash
source .venv/bin/activate
cd backend

# Download data
python -m app.download_data

# Ingest
python -m app.ingest --input ../data/raw --out ../data/processed

# Build indexes
python -m app.index --input ../data/processed/docs.jsonl

# Start API
uvicorn app.main:app --port 8000 --reload
```

### Run tests
```bash
source .venv/bin/activate
cd backend
pytest tests/ -v
```

### Run evaluation
```bash
source .venv/bin/activate
cd backend
python -m app.eval --queries ../data/eval/queries.jsonl --qrels ../data/eval/qrels.json --alpha 0.5
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check + version |
| POST | /search | Hybrid search with score breakdown |
| POST | /feedback | Log relevance feedback |
| GET | /metrics | Prometheus-style metrics |
| GET | /metrics/json | JSON metrics for dashboard |
| GET | /logs | Recent query logs |
| GET | /eval/experiments | Experiment results from CSV |

### Search request example
```json
POST /search
{
  "query": "machine learning neural networks",
  "top_k": 10,
  "alpha": 0.5,
  "normalization": "minmax"
}
```

---

## SQLite Schema

```sql
CREATE TABLE query_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT,
    query TEXT,
    latency_ms REAL,
    top_k INTEGER,
    alpha REAL DEFAULT 0.5,
    normalization TEXT DEFAULT 'minmax',
    result_count INTEGER,
    error TEXT,
    created_at TEXT
);
```

---

## Hybrid Scoring

```
hybrid_score = alpha * norm_bm25 + (1 - alpha) * norm_vector
```

Two normalization strategies available:
- **min-max** (default): scales scores to [0, 1]
- **z-score**: normalizes by mean and standard deviation

See `docs/decision_log.md` for rationale.

---

## Prerequisites

- Python 3.11+
- Git
- Internet connection (first run only, for data download + model download)

---

## Project Structure

```
knowledge-search-dashboard/
├── backend/
│   ├── app/
│   │   ├── search/         # BM25, vector, hybrid engines
│   │   ├── db/             # SQLite storage
│   │   ├── main.py         # FastAPI app
│   │   ├── ingest.py       # Data ingestion
│   │   ├── index.py        # Index builder
│   │   └── eval.py         # Evaluation harness
│   └── tests/              # pytest tests
├── frontend/
│   └── index.html          # React dashboard (single file)
├── data/
│   ├── raw/                # Downloaded corpus
│   ├── processed/          # Cleaned JSONL
│   ├── index/              # BM25 + FAISS indexes
│   ├── eval/               # Queries + qrels
│   └── metrics/            # experiments.csv
├── docs/                   # Architecture, decision, codex, break/fix logs
├── up.sh                   # One-command startup
└── requirements.txt
```

### 🔹 Project Snapshot

![snapshot of dashboard](https://github.com/Pamminarzary/Heartrate-Monitoring-using-rPPG/blob/main/homepage.png)
