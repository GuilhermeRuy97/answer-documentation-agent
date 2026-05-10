# Anthropic Prompt Engineering RAG Agent

An **Agentic RAG system** that answers questions over Anthropic's Prompt Engineering
documentation using LangGraph, Supabase + PGVector, Voyage AI embeddings, and Claude.

---

## High Level Workflow

### The Problem

Anthropic's prompt engineering documentation is detailed and spread across many pages. Finding a specific answer requires knowing where to look and reading through a lot of content. This project replaces that friction with a conversational interface: ask a plain-English question, get a direct answer with links to the source.

### What It Does

A user types a question. The system returns a clear, sourced answer — in seconds — pulling only from the official documentation. Every claim in the answer is traceable to a specific page, so users can verify it themselves.

### How It Works (Without the Jargon)

1. **Understand the question** — The system rephrases the question into two alternative angles to make sure it doesn't miss relevant content due to wording differences.

2. **Search the knowledge base** — It searches a pre-built index of the Anthropic documentation across all three question angles at once, then combines and deduplicates the results.

3. **Rank by relevance** — The combined results are scored and ranked against the original question. Only the most relevant passages move forward.

4. **Self-correct if needed** — If the results aren't relevant enough, the system automatically retries with a wider search — up to two times — before proceeding. This happens invisibly to the user.

5. **Generate a grounded answer** — Claude reads the top results and writes a direct answer. It is only allowed to use what it found; it cannot invent information.

6. **Cite the sources** — Every claim is tagged with a numbered citation linking back to the exact documentation page it came from.

### Key Decisions (Business Perspective)

| Decision | Why It Matters |
|---|---|
| Answers only use facts from the docs | Eliminates hallucination risk — the system cannot make things up |
| Every answer includes source citations | Users can verify claims and build trust in the output |
| Automatic retry when results are weak | The system recovers gracefully from hard questions without user intervention |
| Conversation history is preserved | Users can ask follow-up questions naturally, like a chat |
| Quality is measured on three dimensions | Relevance, faithfulness, and citation accuracy are tracked so the system can be improved over time |

### How Quality Is Measured

After building the system, 15 real questions were run through it and graded by an independent AI judge on three criteria:

| Metric | What It Measures | Score |
|---|---|---|
| Relevance | Did it find the right documentation? | 3.7 / 5 |
| Faithfulness | Did the answer stick to the facts? | 4.8 / 5 |
| Citation Quality | Were the sources accurate and useful? | 4.2 / 5 |

Faithfulness — the most critical metric for trust — scored highest. Relevance has room to improve with a larger or more diverse documentation index.

---

## Architecture

```
User → FastAPI POST /ask → LangGraph Agent
                                │
                    ┌───────────▼───────────┐
                    │   rewrite_query        │  Claude generates 2 HyDE variants
                    └───────────┬───────────┘  (hypothetical answer paragraphs)
                                │
                    ┌───────────▼───────────┐
                    │   search_docs (tool)   │  Voyage AI embed → PGVector search
                    └───────────┬───────────┘  (original + 2 variants, union deduped)
                                │
                    ┌───────────▼───────────┐
                    │   rerank              │  Voyage rerank-2 against original query
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   grade_relevance      │  Mean rerank score < 0.45? → retry
                    └──────┬────────────────┘  (threshold relaxes -0.05 per retry,
                           │        ▲           k doubles; max 2 retries)
                           │        └───────────┘ (loop back to search_docs)
                           │
                    ┌───────▼───────────────┐
                    │   generate_answer      │  Claude answers with top-k chunks
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   format_citations     │  Dedup by URL, number inline [N] markers
                    └───────────┬───────────┘
                                │
                         JSON Response
```

## Tech Stack

- **LLM**: `claude-sonnet-4-6` for generation, `claude-opus-4-7` for evaluation (Anthropic)
- **Embeddings**: Voyage AI `voyage-4` (1024 dims) + `rerank-2` reranker
- **Vector Store**: Supabase + PGVector (HNSW index)
- **Agent**: LangGraph StateGraph
- **API**: FastAPI
- **Evaluation**: LangSmith LLM-as-judge
- **Crawler**: Firecrawl

## Setup

### 1. Install dependencies
```bash
uv sync
```

### 2. Configure environment
```bash
cp .env.example .env
# Fill in all API keys in .env
```

### 3. Setup Supabase
- Open Supabase Dashboard → SQL Editor
- Run the contents of `scripts/setup_supabase.sql`

### 4. Ingest documentation
```bash
uv run python scripts/ingest.py
```

### 5. Start the API
```bash
uv run uvicorn api.main:app --reload --port 8000
```

### 6. Query the agent
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are XML tags used for in Claude prompts?"}'
```

## Evaluation

Fill in the `answer` fields in `evaluation/dataset.json` after ingestion, then:

```bash
uv run python evaluation/run_eval.py
```

Results appear in the terminal and in your LangSmith dashboard.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ask` | Ask a question, get an answer with citations |
| GET | `/health` | Check API and vector store status |
| GET | `/history/{session_id}` | Get conversation history |

## Key Decisions

### Architectural Decisions

**HyDE for query rewriting**
Instead of rephrasing the user's question, the `rewrite_query` node generates two *hypothetical answer paragraphs* written in the style of Anthropic's documentation. Document chunks are semantically closer to answers than to questions, so embedding a hypothetical answer and searching with it yields better cosine similarity than embedding the question directly.

**Multi-query retrieval with union deduplication**
The agent searches the vector store in parallel with the original question plus two hypothetical variants, then deduplicates results by `(source_url, chunk_index, id)`. This union approach improves recall at the cost of a slightly larger candidate set before reranking.

**Two-stage recall → precision pipeline**
A low recall threshold (0.30) casts a wide net at the Supabase RPC level. Voyage's `rerank-2` model then reranks all candidates against the *original* question for precision. The `grade_relevance` node prefers `rerank_score` over raw cosine similarity when deciding whether to retry.

**Adaptive retry with threshold relaxation**
On each retry, the recall threshold drops by 0.05 and the per-variant retrieval count doubles (6 → 12 → 24). This means the system widens its net gradually rather than failing hard when initial retrieval misses, up to `MAX_RETRY_COUNT` (default: 2).

**HNSW over IVFFlat for the vector index**
HNSW requires no training step, has lower query latency for collections under 1 M rows, and allows incremental inserts without rebuilding the index. IVFFlat would be preferable only at much larger scale.

**Raw SQL RPC over `SupabaseVectorStore`**
The LangChain `SupabaseVectorStore` wrapper abstracts away threshold control, metadata shape, and upsert logic. A custom Supabase RPC (`match_docs`) keeps full control over the query and allows the adaptive threshold to be passed at call time.

**Idempotent ingestion via upsert**
Chunks are upserted on a `(source_url, chunk_index)` unique constraint so re-running `scripts/ingest.py` after a partial failure or a docs update never creates duplicates.

**Separate threshold tiers**
Three thresholds serve distinct roles: `RECALL_THRESHOLD` (0.30) filters at the DB level to keep the candidate pool manageable; `RELEVANCE_THRESHOLD` (0.45) gates the retry decision in `grade_relevance`; and the `match_threshold` in the Supabase RPC is overridden at runtime by the adaptive recall logic.

**Citation deduplication by source URL**
Multiple chunks from the same documentation page collapse into a single numbered citation. Inline `[N]` markers in the answer are renumbered to reflect the deduplicated list. If Claude produced no citations, the top 3 chunks are surfaced automatically as a fallback.

**Two temperature regimes**
Query rewriting uses temperature 0.7 (creative, varied hypothetical answers). Answer generation uses temperature 0.2 (deterministic, faithful to the retrieved context). Mixing temperatures at different stages avoids both retrieval homogeneity and answer hallucination.

---

### Business Decisions

**Single-domain focus**
The corpus is intentionally limited to Anthropic's prompt engineering documentation. A narrow domain allows tighter relevance thresholds and makes faithfulness evaluation meaningful — claims can be directly traced back to a small, well-defined source.

**In-memory session storage**
Session history is kept in a plain dict keyed by session ID. This is sufficient for a single-instance prototype and avoids infrastructure overhead. The migration path to Redis (`RedisChatMessageHistory`) is well-defined and not required until horizontal scaling is needed.

**LangSmith for tracing and evaluation**
LangSmith tracing is enabled via environment variables with zero changes to application code. Using it for both tracing and LLM-as-judge evaluation avoids building a separate annotation UI and keeps all observability in one place.

**Model split between generation and evaluation**
`claude-sonnet-4-6` handles query rewriting and answer generation to keep per-query cost low at inference time. `claude-opus-4-7` is used only in the offline evaluation loop where answer quality matters more than latency or cost.

**15-question evaluation dataset**
The dataset covers the major topic areas in the prompt engineering docs without over-investing in curation for a prototype. It is small enough to run locally in a reasonable time while being large enough to surface meaningful score differences across metrics.

---

## Tradeoffs & What I'd Build Next

- **Session storage**: Currently in-memory. Production: Redis via `RedisChatMessageHistory`
- **Streaming**: FastAPI SSE for real-time token streaming
- **MCP server**: Expose `search_docs` as an MCP tool for Claude Code / Claude Desktop
- **Hybrid search**: Combine PGVector similarity with BM25 full-text search for better recall on keyword-heavy queries

