# Anthropic Prompt Engineering RAG Agent

An **Agentic RAG system** that answers questions over Anthropic's Prompt Engineering
documentation using LangGraph, Supabase + PGVector (hybrid search), Voyage AI embeddings,
and Claude. Ships with a chat web UI, persistent conversation memory, API-key auth,
rate limiting, and a LangSmith evaluation pipeline.

---

## High Level Workflow

### The Problem

Anthropic's prompt engineering documentation is detailed and spread across many pages. Finding a specific answer requires knowing where to look and reading through a lot of content. This project replaces that friction with a conversational interface: ask a plain-English question, get a direct answer with links to the source.

### What It Does

A user types a question. The system returns a clear, sourced answer — in seconds — pulling only from the official documentation. Every claim in the answer is traceable to a specific page, so users can verify it themselves.

### How It Works (Without the Jargon)

1. **Remember the conversation** — Previous turns (and a rolling summary of older ones) are loaded so follow-up questions work naturally.

2. **Understand the question** — The system rewrites the question into hypothetical answer paragraphs plus a keyword query, so it doesn't miss relevant content due to wording differences.

3. **Search the knowledge base two ways** — Every query variant runs both a semantic (vector) search and a keyword (full-text) search; the two rankings are fused (Reciprocal Rank Fusion), then results across variants are fused again.

4. **Rank by relevance** — A dedicated reranking model scores the combined candidates against the original question. Only the most relevant passages move forward.

5. **Self-correct if needed** — If the results aren't relevant enough, the system automatically retries with a wider search — up to two times — before proceeding.

6. **Generate a grounded answer** — Claude reads the top results and writes a direct answer. It is only allowed to use what it found; it cannot invent information.

7. **Cite the sources** — Every claim is tagged with a numbered citation linking back to the exact documentation page it came from.

8. **Persist memory** — The turn is stored in the database; long conversations are summarized so context never overflows.

---

## Architecture

```
User → Chat UI / POST /ask → LangGraph Agent
                                │
                    ┌───────────▼───────────┐
                    │   load_memory          │  History + rolling summary from session store
                    └───────────┬───────────┘
                    ┌───────────▼───────────┐
                    │   rewrite_query        │  Claude: 2 HyDE paragraphs + keyword query
                    └───────────┬───────────┘  (JSON via structured outputs)
                    ┌───────────▼───────────┐
                    │   search_docs          │  Hybrid search (vector + full-text, RRF in SQL)
                    └───────────┬───────────┘  per variant → RRF fusion across variants
                    ┌───────────▼───────────┐
                    │   rerank               │  Voyage rerank-2 against original question
                    └───────────┬───────────┘
                    ┌───────────▼───────────┐
                    │   grade_relevance      │  Mean rerank score < 0.45? → retry
                    └──────┬────────────────┘  (threshold relaxes, k widens; max 2 retries)
                           │        ▲
                           │        └──────────── loop back to search_docs
                    ┌──────▼────────────────┐
                    │   generate_answer      │  Claude, XML-tagged context, inline [N] markers
                    └───────────┬───────────┘
                    ┌───────────▼───────────┐
                    │   format_citations     │  Dedup by URL, renumber markers
                    └───────────┬───────────┘
                    ┌───────────▼───────────┐
                    │   save_memory          │  Persist turn + summarize long histories
                    └───────────┬───────────┘
                         JSON Response (+ trace_id for feedback)
```

## Tech Stack

- **LLM**: `claude-sonnet-4-6` for generation, `claude-opus-4-7` for evaluation (Anthropic)
- **Embeddings**: Voyage AI `voyage-4` (1024 dims) + `rerank-2` reranker
- **Vector Store**: Supabase + PGVector (HNSW) + tsvector/GIN full-text, fused with RRF
- **Agent**: LangGraph StateGraph
- **API**: FastAPI + slowapi (rate limit) + API-key auth + CORS
- **UI**: Zero-build static chat app served by FastAPI at `/`
- **Memory**: In-memory TTL cache + Supabase `chat_messages`/`chat_sessions` + Claude summarization
- **Observability**: LangSmith tracing, experiments, and user feedback
- **Crawler**: Firecrawl with explicit seed pages, retries, and content hashing
- **Config**: pydantic-settings (`core/config.py`) — single source of truth

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
- Run `scripts/setup_supabase.sql`
- Run `scripts/migrate_hybrid_search.sql` (hybrid search + chat persistence)

### 4. Ingest documentation
```bash
uv run python scripts/ingest.py            # uses local crawl cache if present
uv run python scripts/ingest.py --force-crawl   # re-crawl from Firecrawl
```

Re-running is cheap: unchanged chunks are detected by content hash and skipped.

### 5. Start the API + UI
```bash
uv run uvicorn api.main:app --reload --port 8000
```

Open http://localhost:8000 for the chat UI.

### 6. Or query the API directly
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are XML tags used for in Claude prompts?"}'
```

### Docker
```bash
docker compose up --build
```

## Tests

```bash
uv run pytest          # unit tests, fully mocked, no API keys needed
```

## Evaluation

```bash
uv run python evaluation/run_eval.py
```

Five metrics: `relevance`, `faithfulness`, `citation_quality`, `answer_relevance`
(LLM-as-judge, 1-5) and `retrieval_hit_rate` (deterministic, did the expected page
get retrieved). Results print as a summary table with latency percentiles and appear
in the LangSmith dashboard when `LANGCHAIN_API_KEY` is set.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | No | Chat web UI |
| POST | `/ask` | Yes* | Ask a question, get an answer with citations + trace_id |
| GET | `/health` | No | API and vector store status |
| GET | `/history/{session_id}` | Yes* | Get conversation history |
| DELETE | `/history/{session_id}` | Yes* | Clear a conversation (cache + DB) |
| POST | `/feedback` | Yes* | Thumbs up/down on an answer, recorded in LangSmith |

*Auth is enforced via the `X-API-Key` header only when `API_KEYS` is set; it is
disabled by default for local development. `POST /ask` is rate limited (default 20/minute).

## Key Decisions

The full rationale for every technical choice lives in
[decisions.md](decisions.md). The short version:

- **Agentic RAG, not a linear pipeline** — a LangGraph state machine grades its own
  retrieval and retries (with relaxed thresholds and wider k) before answering.
- **HyDE + keyword query rewriting** — hypothetical answer paragraphs feed the vector
  leg; a keyword variant feeds the full-text leg. Each retrieval leg gets a query
  shaped for it.
- **Hybrid search with double RRF fusion** — vector + full-text fused in Postgres,
  then fused again across query variants in Python. Rank-based fusion needs no score
  normalization.
- **Supabase + PGVector via raw SQL RPCs** — HNSW index, generated tsvector + GIN,
  RRF inside the database. No LangChain vectorstore abstraction.
- **Two-stage recall → precision** — wide hybrid retrieval, then Voyage `rerank-2`
  against the *original* question; rerank scores (not cosine) gate the retry loop.
- **Contextual chunks + hash-diffed ingestion** — every chunk carries its page title
  and section heading; sha256 content hashes make re-ingestion idempotent and cheap.
- **Grounded generation** — XML-tagged documents, prompt-injection hardening,
  validated and URL-deduplicated `[N]` citations.
- **Two-tier memory** — in-process TTL cache over Supabase tables, with Claude-built
  rolling summaries keeping context bounded.
- **LangSmith end-to-end** — tracing, 5-metric evaluation (4 LLM-judge + 1
  deterministic hit rate), and user feedback recorded on the exact trace.

---

## Further Reading

- [decisions.md](decisions.md) — all technical decisions: RAG paradigm, retrieval techniques, vector DB internals, trade-offs
- [project_knowledge.md](project_knowledge.md) — lessons learned, bugs found and fixed, production guidance
- [possible_steps.md](possible_steps.md) — prioritized future improvements with why and how
