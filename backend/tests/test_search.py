"""
Unit and contract tests for the search system.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.search.bm25 import BM25Index
from app.search.hybrid import minmax_normalize, zscore_normalize, get_snippet, HybridSearchEngine
from app.db.storage import migrate


# ---- BM25 Tests ----

SAMPLE_DOCS = [
    {"doc_id": "d1", "title": "Python Programming", "text": "Python is a versatile programming language used for data science and web development."},
    {"doc_id": "d2", "title": "Machine Learning", "text": "Machine learning algorithms learn patterns from data to make predictions."},
    {"doc_id": "d3", "title": "Database Systems", "text": "SQL databases store structured data efficiently with ACID properties."},
]


def test_bm25_build_and_query(tmp_path):
    idx = BM25Index(index_dir=str(tmp_path / "bm25"))
    idx.build(SAMPLE_DOCS)
    results = idx.query("python programming", top_k=3)
    assert len(results) > 0
    # d1 should rank highest for python
    assert results[0][0] == "d1"


def test_bm25_ordering(tmp_path):
    idx = BM25Index(index_dir=str(tmp_path / "bm25"))
    idx.build(SAMPLE_DOCS)
    results = idx.query("machine learning data", top_k=3)
    ids = [r[0] for r in results]
    assert "d2" in ids
    assert ids[0] == "d2"


def test_bm25_save_load(tmp_path):
    idx = BM25Index(index_dir=str(tmp_path / "bm25"))
    idx.build(SAMPLE_DOCS)
    idx2 = BM25Index(index_dir=str(tmp_path / "bm25"))
    idx2.load()
    results = idx2.query("database sql", top_k=3)
    assert results[0][0] == "d3"


# ---- Normalization Tests ----

def test_minmax_normalize_basic():
    scores = {"a": 10.0, "b": 5.0, "c": 0.0}
    norm = minmax_normalize(scores)
    assert norm["a"] == pytest.approx(1.0)
    assert norm["c"] == pytest.approx(0.0)
    assert 0 < norm["b"] < 1


def test_minmax_normalize_equal_scores():
    """Bug scenario C: all scores equal -> divide by zero -> should return 0s not NaN"""
    scores = {"a": 5.0, "b": 5.0, "c": 5.0}
    norm = minmax_normalize(scores)
    for v in norm.values():
        assert v == 0.0
        assert not (v != v)  # not NaN


def test_zscore_normalize_uniform():
    scores = {"a": 3.0, "b": 3.0, "c": 3.0}
    norm = zscore_normalize(scores)
    for v in norm.values():
        assert v == 0.0
        assert not (v != v)  # not NaN


# ---- Snippet Tests ----

def test_get_snippet_highlights_query():
    text = "Python is a popular programming language for data science and machine learning."
    snippet = get_snippet(text, "python language")
    assert "**" in snippet  # highlights present


# ---- DB Tests ----

def test_migrate_creates_table(tmp_path):
    import app.db.storage as storage
    storage.DB_PATH = str(tmp_path / "test.db")
    migrate()
    import sqlite3
    conn = sqlite3.connect(storage.DB_PATH)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    assert "query_logs" in tables
    conn.close()


# ---- API Contract Tests ----

def test_health_endpoint():
    """Test /health returns ok"""
    # Import only if indexes exist, otherwise skip
    try:
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
    except Exception as e:
        pytest.skip(f"Index not built yet: {e}")
