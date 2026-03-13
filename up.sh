#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Knowledge Search + KPI Dashboard       ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Create virtual environment ────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "📦 Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate
echo "✅ Virtual environment active"

# ── 2. Install dependencies ───────────────────────────────────────
echo "📥 Installing dependencies (this may take a few minutes first time)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "✅ Dependencies installed"

# ── 3. Download data ──────────────────────────────────────────────
if [ ! -f "data/raw/wikipedia_corpus.jsonl" ]; then
  echo "🌐 Downloading Wikipedia corpus (~120 articles)..."
  cd backend && python -m app.download_data && cd ..
else
  echo "✅ Raw data already exists"
fi

# ── 4. Ingest ─────────────────────────────────────────────────────
if [ ! -f "data/processed/docs.jsonl" ]; then
  echo "🔄 Running ingestion pipeline..."
  cd backend && python -m app.ingest --input ../data/raw --out ../data/processed && cd ..
else
  echo "✅ Processed data already exists"
fi

# ── 5. Build indexes ──────────────────────────────────────────────
if [ ! -f "data/index/bm25/bm25.pkl" ] || [ ! -f "data/index/vector/faiss.index" ]; then
  echo "🔨 Building BM25 + vector indexes (this takes ~2-5 minutes first time)..."
  cd backend && python -m app.index --input ../data/processed/docs.jsonl && cd ..
else
  echo "✅ Indexes already built"
fi

# ── 6. Run eval if no experiments exist ──────────────────────────
if [ ! -f "data/metrics/experiments.csv" ]; then
  echo "🧪 Running initial evaluation..."
  cd backend
  for alpha in 0.3 0.5 0.7; do
    python -m app.eval --queries ../data/eval/queries.jsonl --qrels ../data/eval/qrels.json --alpha $alpha --normalization minmax
  done
  for norm in zscore minmax; do
    python -m app.eval --queries ../data/eval/queries.jsonl --qrels ../data/eval/qrels.json --alpha 0.5 --normalization $norm
  done
  cd ..
  echo "✅ Evaluation complete (5 experiments run)"
fi

# ── 7. Start API ──────────────────────────────────────────────────
echo ""
echo "🚀 Starting Knowledge Search API..."
echo ""
echo "┌──────────────────────────────────────────────┐"
echo "│  🔍 Search UI:   http://localhost:8000       │"
echo "│  📖 API Docs:    http://localhost:8000/docs  │"
echo "│  💓 Health:      http://localhost:8000/health│"
echo "└──────────────────────────────────────────────┘"
echo ""
echo "Press Ctrl+C to stop."
echo ""

cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
