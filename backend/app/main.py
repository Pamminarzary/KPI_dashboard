"""
FastAPI application entrypoint.
"""
import csv
import os
import uuid
import time
import subprocess
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.search.bm25 import BM25Index
from app.search.vector import VectorIndex
from app.search.hybrid import HybridSearchEngine
from app.db.storage import migrate, log_query, get_metrics, get_logs

limiter = Limiter(key_func=get_remote_address)
bm25_index = None
vector_index = None
engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bm25_index, vector_index, engine
    migrate()
    bm25_index = BM25Index()
    bm25_index.load()
    model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    vector_index = VectorIndex(model_name=model_name)
    vector_index.load()
    engine = HybridSearchEngine(bm25_index, vector_index)
    print("Search engine ready.")
    yield


app = FastAPI(title="Knowledge Search API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../frontend")


def get_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(10, ge=1, le=50)
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    normalization: str = Field("minmax", pattern="^(minmax|zscore)$")
    filters: Optional[Dict] = None


class FeedbackRequest(BaseModel):
    query: str
    doc_id: str
    relevant: bool


@app.get("/")
def root():
    path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "Knowledge Search API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0", "commit": get_commit()}


@app.post("/search")
@limiter.limit("60/minute")
async def search(request: Request, body: SearchRequest):
    request_id = str(uuid.uuid4())
    start = time.time()
    error = None
    results = []
    try:
        engine.normalization = body.normalization
        results = engine.search(query=body.query, top_k=body.top_k, alpha=body.alpha, filters=body.filters)
    except Exception as e:
        error = str(e)
        raise HTTPException(status_code=500, detail=error)
    finally:
        latency = (time.time() - start) * 1000
        log_query(request_id=request_id, query=body.query, latency_ms=latency, top_k=body.top_k,
                  alpha=body.alpha, normalization=body.normalization, result_count=len(results), error=error)
    return {"request_id": request_id, "query": body.query, "alpha": body.alpha,
            "normalization": body.normalization, "result_count": len(results),
            "latency_ms": round((time.time() - start) * 1000, 2), "results": results}


@app.post("/feedback")
async def feedback(body: FeedbackRequest):
    return {"status": "recorded", "query": body.query, "doc_id": body.doc_id, "relevant": body.relevant}


@app.get("/metrics")
def metrics():
    data = get_metrics()
    return "\n".join([
        "# HELP search_total_requests Total search requests",
        f"search_total_requests {data['total_requests']}",
        f"search_p50_latency_ms {data['p50_ms']}",
        f"search_p95_latency_ms {data['p95_ms']}",
    ])


@app.get("/metrics/json")
def metrics_json():
    return get_metrics()


@app.get("/logs")
def logs(limit: int = 100):
    return get_logs(limit)


@app.get("/eval/experiments")
def eval_experiments():
    csv_path = "data/metrics/experiments.csv"
    if not os.path.exists(csv_path):
        return []
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows
