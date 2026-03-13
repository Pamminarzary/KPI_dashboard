# Break/Fix Log

---

## Scenario A: Semantic Index Model Mismatch

### How to Induce
1. Build index with `all-MiniLM-L6-v2` (dim=384)
2. Change `EMBEDDING_MODEL=all-mpnet-base-v2` in environment (dim=768)
3. Restart API without rebuilding vector index

### Expected Failure
```
ValueError: Model mismatch: index built with 'all-MiniLM-L6-v2', 
but current model is 'all-mpnet-base-v2'. Rebuild index.
```
API fails at startup. All `/search` requests return 503.

### Fix Implemented
In `backend/app/search/vector.py` → `load()`:
```python
if meta["model_name"] != self.model_name:
    raise ValueError(
        f"Model mismatch: index built with '{meta['model_name']}', "
        f"but current model is '{self.model_name}'. Rebuild index."
    )
if self.index.d != meta["dimension"]:
    raise ValueError(f"Dimension mismatch: index has dim={self.index.d}...")
```
Clear error message tells operator exactly what to do: rebuild with `python -m app.index`.

---

## Scenario B: SQLite Schema Migration Break

### How to Induce
1. Start API (creates `query_logs` table with v1 schema, no `normalization` column)
2. Add `NOT NULL` constraint to a new column manually:
   ```sql
   ALTER TABLE query_logs ADD COLUMN experiment_id TEXT NOT NULL;
   ```
3. Restart API → writes fail with `NOT NULL constraint failed`

### Expected Failure
```
sqlite3.IntegrityError: NOT NULL constraint failed: query_logs.experiment_id
```
Every search request fails to log. Dashboard shows 0 metrics.

### Fix Implemented
In `backend/app/db/storage.py` → `migrate()`:
```python
cols = [r[1] for r in c.execute("PRAGMA table_info(query_logs)")]
if "normalization" not in cols:
    c.execute("ALTER TABLE query_logs ADD COLUMN normalization TEXT DEFAULT 'minmax'")
```
Migration runs at startup. New columns always use `DEFAULT` values, never `NOT NULL` without defaults.

---

## Scenario C: Hybrid Scoring Divide-by-Zero

### How to Induce
Query a single very rare token that matches exactly one document with the same BM25 score as all others (e.g., all docs score 0.0 from BM25 for the query "zzzzzzunknownword").

All BM25 scores = 0.0 → `max - min = 0` → division by zero → `NaN` hybrid scores.

### Expected Failure
- All hybrid scores become `NaN`
- Results sort incorrectly (NaN comparisons are undefined)
- Evaluation MRR/nDCG return 0.0 for all queries
- Test: `assert not (v != v)` catches NaN

### Fix Implemented
In `backend/app/search/hybrid.py` → `minmax_normalize()`:
```python
rng = mx - mn
if rng == 0:
    return {k: 0.0 for k in scores}  # all equal → all zero, not NaN
```
Same fix in `zscore_normalize()`:
```python
if std == 0:
    return {k: 0.0 for k in scores}
```

### Test Added
```python
def test_minmax_normalize_equal_scores():
    scores = {"a": 5.0, "b": 5.0, "c": 5.0}
    norm = minmax_normalize(scores)
    for v in norm.values():
        assert v == 0.0
        assert not (v != v)  # not NaN
```
Test passes after fix. Eval metrics recover to correct values.
