# Codex Prompt Log

Each entry maps to a specific file, function, and git commit.

---

## Prompt 1 — BM25 Index
**File**: `backend/app/search/bm25.py`  
**Prompt**: "In backend/app/search/bm25.py implement BM25 scoring using rank-bm25. Provide BM25Index class with build() and query() methods. Add pickle-based persistence. CPU-only. Acceptance: 3-doc toy corpus returns deterministic ordering."  
**Used from output**: Full class structure. Edited: added `get_doc()` helper and explicit type hints.  
**Commit**: `feat: BM25Index with build/query/load/persist`

---

## Prompt 2 — Vector Index
**File**: `backend/app/search/vector.py`  
**Prompt**: "In backend/app/search/vector.py implement VectorIndex using sentence-transformers all-MiniLM-L6-v2 and faiss-cpu IndexFlatIP. Add build(), load(), query(). Store metadata JSON with model_name and dimension for startup validation. CPU-only, batch_size=32."  
**Used from output**: Core structure. Edited: added dimension mismatch check and model mismatch error on load().  
**Commit**: `feat: VectorIndex with FAISS + startup validation`

---

## Prompt 3 — Hybrid Engine
**File**: `backend/app/search/hybrid.py`  
**Prompt**: "In backend/app/search/hybrid.py combine BM25 and vector scores. Implement minmax_normalize and zscore_normalize. hybrid = alpha * norm_bm25 + (1-alpha) * norm_vector. Handle divide-by-zero when all scores are equal. Add snippet highlighting."  
**Used from output**: minmax and zscore functions. Edited: added explicit NaN guard for rng==0.  
**Commit**: `feat: HybridSearchEngine with two normalization strategies`

---

## Prompt 4 — SQLite Storage
**File**: `backend/app/db/storage.py`  
**Prompt**: "In backend/app/db/storage.py implement SQLite logging for search queries. Schema: id, request_id, query, latency_ms, top_k, alpha, normalization, result_count, error, created_at. Add migrate() that handles v1→v2 ALTER TABLE. Add get_metrics() returning p50/p95/top_queries/zero_results."  
**Used from output**: Full schema and migrate(). Edited: added get_logs() for debug view.  
**Commit**: `feat: SQLite logging with v1->v2 migration`

---

## Prompt 5 — FastAPI Routes
**File**: `backend/app/main.py`  
**Prompt**: "In backend/app/main.py add FastAPI app with /health, POST /search, GET /metrics/json, GET /logs, GET /eval/experiments. Use lifespan to load BM25 + vector indexes. Add slowapi rate limiting (60/min). Validate SearchRequest with Pydantic. Serve frontend/index.html at /."  
**Used from output**: App skeleton and route signatures. Edited: fixed lifespan globals, added CORS.  
**Commit**: `feat: FastAPI with all routes, rate limiting, CORS`

---

## Prompt 6 — Evaluation Harness
**File**: `backend/app/eval.py`  
**Prompt**: "In backend/app/eval.py implement nDCG@K, Recall@K, MRR@K from scratch (no sklearn). Load queries.jsonl and qrels.json. Append results with timestamp+commit to data/metrics/experiments.csv."  
**Used from output**: DCG formula. Edited: added MRR, CSV header guard, CLI args.  
**Commit**: `feat: evaluation harness with nDCG/Recall/MRR`

---

## Prompt 7 — up.sh
**File**: `up.sh`  
**Prompt**: "Write up.sh that: creates .venv if missing, installs requirements.txt, downloads data if missing, runs ingest+index if artifacts missing, runs 5 eval experiments if CSV missing, starts uvicorn on port 8000. Must work from repo root with no absolute paths. Print URLs on startup."  
**Used from output**: Shell structure. Edited: added eval loop for 5 experiments, improved error messages.  
**Commit**: `feat: up.sh one-command setup and launch`

---

## Prompt 8 — Tests
**File**: `backend/tests/test_search.py`  
**Prompt**: "In backend/tests/test_search.py write pytest tests for: BM25 3-doc ordering, BM25 save/load round-trip, minmax_normalize with equal scores (NaN guard), zscore with uniform scores, snippet highlighting, SQLite migrate creates table, /health contract test."  
**Used from output**: Test structure. Edited: added tmp_path fixtures, skipped API test if index missing.  
**Commit**: `test: unit tests for BM25, normalization, db, API`

---

## Prompt 9 — Break/Fix Scenarios
**Files**: `docs/break_fix_log.md`  
**Prompt**: "Document three break/fix scenarios: (A) vector index model mismatch, (B) SQLite schema migration, (C) normalization divide-by-zero. For each: how to induce, expected failure, fix implemented."  
**Used from output**: Scenario structure. Edited: added actual error messages observed during testing.  
**Commit**: `docs: break_fix_log with 3 scenarios`
