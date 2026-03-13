# Decision Log

## 1. Normalization Strategy: Min-Max (default)

**Decision**: Use min-max normalization as default, with z-score as an alternative.

**Rationale**:  
Min-max normalization maps all scores to [0, 1], making alpha directly interpretable as a weight between 0% BM25 and 100% vector. Z-score normalization is more robust to outliers but produces unbounded scores that are harder to reason about. Min-max was chosen as default for explainability.

**Risk**: When all BM25 scores are equal (e.g., single-token queries), min-max produces 0.0 for all docs. This is handled explicitly with a `rng == 0` check that returns 0.0 instead of NaN.

---

## 2. Embedding Model: all-MiniLM-L6-v2

**Decision**: Use `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, ~22M parameters).

**Rationale**: This model runs efficiently on CPU, produces high-quality sentence embeddings, and is under 100MB. Larger models like `all-mpnet-base-v2` offer marginal quality gains but are 4x slower on CPU.

---

## 3. Vector Search: FAISS IndexFlatIP

**Decision**: Use FAISS `IndexFlatIP` (inner product after L2 normalization = cosine similarity).

**Rationale**: For corpora under 100K docs, flat exact search is fast enough (< 10ms on CPU). ANN indexes (HNSW, IVF) would be needed at 1M+ docs. Exact search ensures correct recall for evaluation.

---

## 4. Alpha Default: 0.5

**Decision**: Default alpha=0.5 (equal BM25 and vector weights).

**Rationale**: Balanced hybrid search performs well across diverse query types. Keyword-heavy queries benefit from higher alpha; semantic queries benefit from lower alpha. Users can tune via the dashboard.

---

## 5. Storage: SQLite

**Decision**: SQLite for query logs and metrics.

**Rationale**: Zero-dependency, file-based, and sufficient for thousands of queries/day. PostgreSQL would be overkill for a single-machine deployment. Schema migrations are handled manually with ALTER TABLE.

---

## 6. Frontend: Single HTML file (React via CDN)

**Decision**: Single `index.html` with React loaded via CDN instead of Vite build.

**Rationale**: Eliminates Node.js build step complexity, making setup faster. The tradeoff is slightly slower initial load and no TypeScript. For a demo/eval system this is acceptable.
