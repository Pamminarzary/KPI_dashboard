"""
SQLite database for query logging and metrics.
Schema v2: includes alpha and normalization columns.
"""
import json
import os
import sqlite3
import time
from datetime import datetime
from typing import List, Optional


DB_PATH = os.environ.get("DB_PATH", "data/search_logs.db")


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def migrate():
    conn = get_connection()
    c = conn.cursor()
    # v1 schema
    c.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
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
        )
    """)
    # v2 migration: add normalization if missing
    cols = [r[1] for r in c.execute("PRAGMA table_info(query_logs)")]
    if "normalization" not in cols:
        c.execute("ALTER TABLE query_logs ADD COLUMN normalization TEXT DEFAULT 'minmax'")
    conn.commit()
    conn.close()


def log_query(
    request_id: str,
    query: str,
    latency_ms: float,
    top_k: int,
    alpha: float,
    normalization: str,
    result_count: int,
    error: Optional[str] = None
):
    conn = get_connection()
    conn.execute(
        """INSERT INTO query_logs
           (request_id, query, latency_ms, top_k, alpha, normalization, result_count, error, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (request_id, query, latency_ms, top_k, alpha, normalization, result_count, error,
         datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def get_metrics(limit: int = 1000) -> dict:
    conn = get_connection()
    rows = conn.execute(
        "SELECT latency_ms, result_count, query, created_at FROM query_logs ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()

    if not rows:
        return {"total_requests": 0, "p50_ms": 0, "p95_ms": 0, "zero_result_queries": [], "top_queries": [], "volume_over_time": []}

    import statistics
    latencies = [r["latency_ms"] for r in rows]
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    p50 = latencies_sorted[int(n * 0.5)]
    p95 = latencies_sorted[int(n * 0.95)]

    query_counts = {}
    zero_result = []
    for r in rows:
        q = r["query"]
        query_counts[q] = query_counts.get(q, 0) + 1
        if r["result_count"] == 0:
            zero_result.append(q)

    top_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # volume by hour
    from collections import defaultdict
    vol = defaultdict(int)
    for r in rows:
        hour = r["created_at"][:13] if r["created_at"] else "unknown"
        vol[hour] += 1

    return {
        "total_requests": len(rows),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "zero_result_queries": list(set(zero_result))[:10],
        "top_queries": [{"query": q, "count": c} for q, c in top_queries],
        "volume_over_time": [{"hour": h, "count": c} for h, c in sorted(vol.items())]
    }


def get_logs(limit: int = 100) -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM query_logs ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
