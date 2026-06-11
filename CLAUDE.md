# CLAUDE.md — Anthropic Prompt Engineering RAG Agent

> This file is the single source of truth for Claude Code working in this project.
> Read it fully before touching any file.

---

## 1. What This Project Does

An **Agentic RAG system** that answers questions over Anthropic's Prompt Engineering documentation.

A LangGraph agent loads conversation memory, rewrites the user question into HyDE + keyword
variants, runs hybrid (vector + full-text) search on Supabase+PGVector fused with RRF,
reranks with Voyage, grades relevance (with retries), generates a cited answer with Claude,
formats citations, and persists memory (with rolling summarization). Exposed via FastAPI
with a static chat UI, API-key auth, and rate limiting. Evaluated with LangSmith.

---

## 2. Architecture

```
User Question (UI or POST /ask)
 │
 ▼
LangGraph Agent
 ├─► load_memory       → history + rolling summary from session store
 ├─► rewrite_query     → 2 HyDE paragraphs + 1 keyword query (JSON prefill)
 ├─► search_docs       → hybrid_match_docs RPC per variant → RRF fusion → Voyage rerank
 ├─► grade_relevance   → mean rerank score; retry_count counts search→grade cycles
 │     ├─ score < RELEVANCE_THRESHOLD AND retries_used < MAX_RETRY_COUNT → search_docs
 │     └─ otherwise → generate_answer
 ├─► generate_answer   → Claude, XML-tagged documents, inline [N] markers
 ├─► format_citations  → dedup by URL, renumber markers
 └─► save_memory       → persist turn to Supabase + summarize long histories
 │
 ▼
JSON Response (answer, citations, session_id, rewritten_queries, trace_id)
```

---

## 3. Tech Stack — Do Not Deviate

| Layer | Technology | Notes |
|---|---|---|
| LLM | `claude-sonnet-4-6` via `anthropic` SDK | All generation; judge is `claude-opus-4-7` |
| Embeddings | `voyage-4` via `voyageai` | 1024 dims, `input_type` param required |
| Reranker | Voyage `rerank-2` | Precision stage after hybrid retrieval |
| Vector Store | Supabase + PGVector + tsvector | Raw SQL RPCs: `match_docs`, `hybrid_match_docs` |
| Agent | LangGraph `StateGraph` | See `agent/graph.py` |
| API | FastAPI + slowapi | Async, Pydantic v2 schemas, API-key auth, CORS |
| Config | pydantic-settings | `core/config.py` — never read env vars directly |
| Tracing | LangSmith | Enabled via env vars; run metadata/tags on /ask |
| Evaluation | LangSmith SDK | 5 metrics incl. deterministic retrieval_hit_rate |
| Crawler | Firecrawl Python SDK | One-time ingestion with seed URLs + content hashing |
| UI | Static HTML/JS | `static/index.html`, no build step |

---

## 4. Environment Variables

All secrets live in `.env` (see `.env.example` for the full annotated list).
Never hardcode them. All configuration is read through `core.config.get_settings()`.

Key tunables and their defaults: `CHUNK_SIZE=1200`, `CHUNK_OVERLAP=200`,
`RETRIEVAL_TOP_K=6`, `RERANK_TOP_K=6`, `RECALL_THRESHOLD=0.30`,
`RELEVANCE_THRESHOLD=0.45`, `MAX_RETRY_COUNT=2`, `USE_HYBRID_SEARCH=true`,
`RRF_K=60`, `SESSION_TTL_SECONDS=3600`, `MAX_HISTORY_MESSAGES=12`,
`API_KEYS=` (empty = auth disabled), `RATE_LIMIT_ASK=20/minute`.

---

## 5. File Responsibilities

### `core/`
| File | Responsibility |
|---|---|
| `config.py` | pydantic-settings `Settings` + cached `get_settings()`. Single source for all env vars |
| `clients.py` | Shared lazy `get_anthropic_client()` used by agent nodes and evaluators |
| `logging.py` | `setup_logging()` used by all entry points |

### `ingestion/`
| File | Responsibility |
|---|---|
| `crawler.py` | Firecrawl wrapper. Seed URL list + map discovery, include/exclude patterns, per-page retry/backoff, JSON cache with content hashes |
| `chunker.py` | RecursiveCharacterTextSplitter + markdown cleaning + contextual headers (page title + section) + `content_hash()` |
| `embedder.py` | Lazy Voyage client. `embed_documents(texts)` / `embed_query(text)` with correct `input_type` |
| `pipeline.py` | crawl → chunk → hash-diff → embed only changed → upsert → prune stale chunks |

### `retrieval/`
| File | Responsibility |
|---|---|
| `vector_store.py` | Lazy Supabase client. `upsert_chunks`, `similarity_search`, `hybrid_search` (falls back to vector-only pre-migration), `fetch_existing_hashes`, `delete_stale_chunks`, `health_check` |
| `fusion.py` | `reciprocal_rank_fusion()` — merges ranked lists across query variants |
| `reranker.py` | Voyage rerank wrapper with graceful fallback |
| `retriever.py` | High-level `retrieve(query, k, threshold)` used by agent + tool |

### `agent/`
| File | Responsibility |
|---|---|
| `state.py` | `AgentState` TypedDict (includes `summary`, `error`) |
| `prompts.py` | ALL runtime prompts: XML-tagged, few-shot, prefill constants |
| `nodes.py` | `load_memory`, `rewrite_query`, `grade_relevance`, `generate_answer`, `save_memory`; Claude calls via `core.clients`; per-call latency/token logging |
| `citations.py` | `build_citations()` shared by graph node and tool |
| `tools.py` | Thin LangChain tool wrappers over retriever + citations |
| `graph.py` | `search_docs_node`, `should_retry`, `format_citations_node`, `build_graph()` |
| `session.py` | Two-tier memory: TTL in-memory cache + Supabase `chat_messages`/`chat_sessions`. `get_history`, `get_summary`, `append_messages`, `save_summary`, `trim_history`, `clear_history`. DB reload is bounded to the last `MAX_HISTORY_MESSAGES` (older turns live in the summary) |

### `api/`
| File | Responsibility |
|---|---|
| `schemas.py` | Pydantic v2 models incl. `FeedbackRequest`, `DeleteHistoryResponse`; `trace_id` on `AskResponse` |
| `security.py` | `require_api_key` dependency (constant-time compare) + slowapi `limiter` |
| `routes.py` | `POST /ask`, `GET /health` (public), `GET/DELETE /history/{id}`, `POST /feedback`. Sanitized 500s, graph run in threadpool, LangSmith run_id/tags/metadata |
| `main.py` | FastAPI app: CORS, rate-limit handler, static mount, `/` serves the chat UI |

### `static/`
| File | Responsibility |
|---|---|
| `index.html` | Zero-build chat UI: markdown rendering, citation chips, feedback buttons, localStorage session, dark theme |

### `evaluation/`
| File | Responsibility |
|---|---|
| `dataset.json` | 15 Q&A pairs over the prompt engineering docs |
| `evaluators.py` | Judges: `relevance`, `faithfulness`, `citation_quality`, `answer_relevance`; deterministic `retrieval_hit_rate` |
| `run_eval.py` | LangSmith experiment (or local loop) + summary table with latency p50/p95 |

### `scripts/`
| File | Responsibility |
|---|---|
| `setup_supabase.sql` | Base table, HNSW index, `match_docs` RPC |
| `migrate_hybrid_search.sql` | tsvector + GIN, `hybrid_match_docs` RRF RPC, `chunk_index` fix, chat tables, `content_hash` |
| `ingest.py` | CLI: `uv run python scripts/ingest.py [--force-crawl]` |

### `tests/`
Unit tests, fully mocked, no API keys required. Run with `uv run pytest`.
Live tests must be marked `@pytest.mark.integration` (deselected by default).

---

## 6. LangGraph State Shape

```python
class AgentState(TypedDict):
    session_id: str
    messages: Annotated[List[BaseMessage], operator.add]  # reducer: append
    query: str
    summary: str               # rolling summary of older turns
    rewritten_queries: List[str]
    retrieved_chunks: List[dict]
    relevance_score: float     # mean rerank/cosine score of candidates
    answer: str
    citations: List[dict]
    final_response: str
    retry_count: int           # completed search→grade cycles (1 = initial search)
    error: str                 # non-empty when a node degraded gracefully
```

---

## 7. Graph Flow — Conditional Logic

```python
# retry_count counts search→grade cycles; retries_used = retry_count - 1.
def should_retry(state: AgentState) -> str:
    settings = get_settings()
    retries_used = state["retry_count"] - 1
    if state["relevance_score"] < settings.relevance_threshold and retries_used < settings.max_retry_count:
        return "search_docs"
    return "generate_answer"
```

With `MAX_RETRY_COUNT=2`: at most 3 searches (1 initial + 2 retries).

---

## 8. Retrieval Pipeline

1. Each query variant → `hybrid_match_docs` RPC (vector + full-text, RRF in SQL).
2. Variant result lists fused in Python via `reciprocal_rank_fusion()`.
3. Voyage `rerank-2` reranks fused candidates against the ORIGINAL question.
4. On retry: per-variant k doubles and the recall floor drops 0.05/attempt
   (the floor applies to pure-vector search and the hybrid fallback; the
   hybrid RRF ranking itself has no score floor and widens through k).

---

## 9. FastAPI Endpoints

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/` | no | Chat UI |
| POST | `/ask` | X-API-Key* | Rate limited; returns `trace_id` |
| GET | `/health` | no | Public for healthchecks |
| GET | `/history/{session_id}` | X-API-Key* | 404 when empty |
| DELETE | `/history/{session_id}` | X-API-Key* | Clears cache + DB |
| POST | `/feedback` | X-API-Key* | Records user rating on the LangSmith trace |

*Only when `API_KEYS` is set; disabled by default for local dev.

---

## 10. Evaluation Approach

Five metrics, judges scored 1-5, hit rate 0/1:

| Evaluator | Question it answers |
|---|---|
| `relevance` | Are the retrieved chunks relevant to the question? |
| `faithfulness` | Does the answer only use facts from the retrieved chunks? |
| `citation_quality` | Are citations accurate and properly linked? |
| `answer_relevance` | Does the answer actually address the question? |
| `retrieval_hit_rate` | Was the expected source page retrieved? (deterministic) |

Run with: `uv run python evaluation/run_eval.py`

---

## 11. Coding Conventions

- Python 3.12, type hints everywhere
- Async FastAPI route handlers; blocking work via `run_in_threadpool`
- All LangGraph nodes are pure functions: `(state: AgentState) -> dict` returning only changed keys
- All configuration via `core.config.get_settings()` — never `os.getenv()` in business logic
- All external clients (Anthropic, Voyage, Supabase) are LAZY — modules must import without env vars
- All logging via `logging` stdlib, not `print()`
- Pydantic v2 syntax
- All prompts live in `agent/prompts.py`
- No emojis in code

---

## 12. What NOT to Do

- ❌ Do not use `langchain_community.vectorstores.SupabaseVectorStore` — raw SQL RPC only
- ❌ Do not store API keys in code — settings only
- ❌ Do not use `gpt-*` models — this is an Anthropic project
- ❌ Do not make the ingestion pipeline re-run on API startup
- ❌ Do not skip `input_type` in Voyage AI calls
- ❌ Do not return the full state from LangGraph nodes — only changed keys
- ❌ Do not create module-level API clients — keep them lazy
- ❌ Do not leak exception details in HTTP responses

---

## 13. How to Run (Full Setup)

```bash
# 1. Install dependencies
uv sync

# 2. Supabase: run scripts/setup_supabase.sql, then scripts/migrate_hybrid_search.sql

# 3. Run ingestion
uv run python scripts/ingest.py

# 4. Start API + UI
uv run uvicorn api.main:app --reload --port 8000

# 5. Run tests
uv run pytest

# 6. Run evaluation
uv run python evaluation/run_eval.py
```

# Important

1. Make real tests to check if the implementation is correct and the code is working as expected.
- Define success criteria. Loop until verified.
2. Don't assume:
- If multiple interpretations exist, present them - don't pick silently.
- If something is unclear, stop. Name what is confusing. Ask.
3. Simplicity first:
- Ask yourself: "Would a senior engineer say this is overcomplicated?" If the answer is yes, simplify.
4. Touch only what you must. Clean up only your own mess.
5. If you noticed unrelated dead code, mention it - don't delete it. When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.
