# CLAUDE.md — Anthropic Prompt Engineering RAG Agent

> This file is the single source of truth for Claude Code working in this project.
> Read it fully before touching any file.

---

## 1. What This Project Does

An **Agentic RAG system** that answers questions over Anthropic's Prompt Engineering documentation
(`docs.anthropic.com/en/docs/build-with-claude/prompt-engineering`).

A LangGraph agent receives a user question, rewrites it into multiple query variants,
searches a Supabase+PGVector vector store, grades the retrieved chunks for relevance,
generates an answer using Claude, formats citations, and returns a structured JSON response.
The whole thing is exposed via a FastAPI endpoint and evaluated with LangSmith.

---

## 2. Architecture

```
User Question
     │
     ▼
FastAPI  POST /ask
     │
     ▼
LangGraph Agent
     │
     ├─► Node: rewrite_query     → generates 2-3 query variants via Claude
     │
     ├─► Tool: search_docs       → embeds query with Voyage AI, queries PGVector
     │
     ├─► Node: grade_relevance   → checks avg similarity score
     │         │
     │         ├─ score < 0.70 AND retry < 2 → back to search_docs
     │         └─ score >= 0.70              → generate_answer
     │
     ├─► Node: generate_answer   → Claude answers using retrieved chunks
     │
     └─► Tool: format_citations  → builds [{title, url, snippet}] list
          │
          ▼
     Final JSON Response
```

---

## 3. Tech Stack — Do Not Deviate

| Layer | Technology | Notes |
|---|---|---|
| LLM | `claude-sonnet-4-6` via `anthropic` SDK | Use for all generation |
| Embeddings | `voyage-4` via `voyageai` | 1024 dims, `input_type` param required |
| Vector Store | Supabase + PGVector | Use `supabase-py` client |
| Agent | LangGraph `StateGraph` | See `agent/graph.py` |
| Retriever | LangChain | `SupabaseVectorStore` or raw SQL via RPC |
| API | FastAPI | Async, Pydantic v2 schemas |
| Tracing | LangSmith | Enabled via env vars, zero extra code |
| Evaluation | LangSmith SDK | `langsmith.evaluation` module |
| Crawler | Firecrawl Python SDK | One-time ingestion script |

---

## 4. Environment Variables

All secrets live in `.env`. Never hardcode them. Load with `python-dotenv`.

```
# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Voyage AI (MongoDB-managed key)
VOYAGE_API_KEY=pa-...

# Supabase
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...       # Use service key (not anon) for server-side writes

# Firecrawl
FIRECRAWL_API_KEY=fc-...

# LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=anthropic-rag-agent

# App settings
RETRIEVAL_TOP_K=5
RELEVANCE_THRESHOLD=0.70
MAX_RETRY_COUNT=2
CHUNK_SIZE=800
CHUNK_OVERLAP=100
```

---

## 5. File Responsibilities

### `ingestion/`
| File | Responsibility |
|---|---|
| `crawler.py` | Firecrawl SDK wrapper. Crawls docs URL, returns list of `{url, markdown, title}` dicts |
| `chunker.py` | RecursiveCharacterTextSplitter. Input: raw page. Output: list of chunk dicts with metadata preserved |
| `embedder.py` | Voyage AI client wrapper. `embed_documents(texts)` and `embed_query(text)` with correct `input_type` |
| `pipeline.py` | Orchestrates: crawl → chunk → embed → upsert to Supabase. The main ingestion entry point |

### `retrieval/`
| File | Responsibility |
|---|---|
| `vector_store.py` | Supabase client setup. `upsert_chunks(chunks)` and `similarity_search(embedding, k, threshold)` via RPC |
| `retriever.py` | High-level retriever used by the agent tool. Takes a query string, returns ranked chunk dicts |

### `agent/`
| File | Responsibility |
|---|---|
| `state.py` | `AgentState` TypedDict. All fields the graph reads/writes |
| `tools.py` | Two LangChain tools: `search_docs` and `format_citations` |
| `nodes.py` | Three node functions: `rewrite_query`, `grade_relevance`, `generate_answer` |
| `graph.py` | Assembles the `StateGraph`, adds nodes, edges, conditional edges, compiles it |
| `session.py` | In-memory dict `{session_id: List[messages]}`. `get_history`, `save_history`, `clear_history` |

### `api/`
| File | Responsibility |
|---|---|
| `schemas.py` | Pydantic v2 models: `AskRequest`, `AskResponse`, `Citation`, `HealthResponse` |
| `routes.py` | Route handlers: `POST /ask`, `GET /health`, `GET /history/{session_id}` |
| `main.py` | FastAPI app, lifespan context manager (compiles graph on startup), mounts router |

### `evaluation/`
| File | Responsibility |
|---|---|
| `dataset.json` | 15 Q&A pairs over the prompt engineering docs |
| `evaluators.py` | LangSmith LLM-as-judge evaluators: `relevance`, `faithfulness`, `citation_quality` |
| `run_eval.py` | Loads dataset → runs agent on each → scores → prints summary table |

### `scripts/`
| File | Responsibility |
|---|---|
| `setup_supabase.sql` | Creates `docs_chunks` table, enables pgvector, creates HNSW index, creates RPC function |
| `ingest.py` | CLI entry: `python scripts/ingest.py`. Calls `ingestion/pipeline.py` |

---

## 6. LangGraph State Shape

```python
class AgentState(TypedDict):
    session_id: str
    messages: Annotated[List[BaseMessage], operator.add]  # reducer: append
    query: str                        # current user question
    rewritten_queries: List[str]      # Claude-generated query variants
    retrieved_chunks: List[dict]      # [{content, source_url, page_title, similarity}]
    relevance_score: float            # mean similarity of top-k chunks
    answer: str                       # raw generated answer
    citations: List[dict]             # [{title, url, snippet}]
    final_response: str               # answer + formatted citations
    retry_count: int                  # retrieval retry counter
```

---

## 7. Graph Flow — Conditional Logic

```python
# In graph.py:
graph.add_conditional_edges(
    "grade_relevance",
    should_retry,  # returns "search_docs" or "generate_answer"
)

def should_retry(state: AgentState) -> str:
    if state["relevance_score"] < RELEVANCE_THRESHOLD and state["retry_count"] < MAX_RETRY_COUNT:
        return "search_docs"
    return "generate_answer"
```

---

## 8. The Two Custom Tools

### Tool 1: `search_docs`
```
Input:  query (str), k (int, default=5)
Action: embed query with Voyage AI input_type="query"
        call Supabase RPC match_docs(embedding, k, threshold)
Output: List[dict] with keys: content, source_url, page_title, similarity
```

### Tool 2: `format_citations`
```
Input:  answer (str), chunks (List[dict])
Action: deduplicate chunks by source_url
        extract a short snippet (first 120 chars) per unique source
        number them [1], [2], etc.
Output: {"formatted": "answer text [1][2]", "citations": [{title, url, snippet}]}
```

---

## 9. FastAPI Endpoints

### `POST /ask`
```json
// Request
{
  "question": "What are XML tags used for in prompts?",
  "session_id": "optional-uuid-for-multi-turn"
}

// Response
{
  "answer": "XML tags help Claude...",
  "citations": [
    {"title": "Use XML tags", "url": "https://docs.anthropic.com/...", "snippet": "XML tags..."}
  ],
  "session_id": "uuid",
  "query_rewritten": ["What is the role of XML tags...", "XML tags Claude prompts..."]
}
```

### `GET /health`
```json
{"status": "ok", "vector_store": "connected", "model": "claude-sonnet-4-5"}
```

### `GET /history/{session_id}`
```json
{"session_id": "uuid", "messages": [...]}
```

---

## 10. Evaluation Approach

LangSmith LLM-as-judge with three criteria, each scored 1-5:

| Evaluator | Question it answers |
|---|---|
| `relevance` | Are the retrieved chunks relevant to the question? |
| `faithfulness` | Does the answer only use facts from the retrieved chunks? |
| `citation_quality` | Are citations accurate and properly linked to the answer? |

Run with: `python evaluation/run_eval.py`
Results appear in LangSmith dashboard under project `anthropic-rag-agent`.

---

## 11. Coding Conventions

- Python 3.11+, type hints everywhere
- Async FastAPI route handlers (`async def`)
- All LangGraph nodes are pure functions: `(state: AgentState) -> dict`
  - Return only the keys you're updating, not the full state
- All Anthropic API calls use `claude-sonnet-4-5`
- All logging via `logging` stdlib, not `print()`
- Pydantic v2 syntax (`model_config`, not `class Config`)
- No global mutable state outside of `session.py`

---

## 12. What NOT to Do

- ❌ Do not use `langchain_community.vectorstores.SupabaseVectorStore` directly — use raw SQL RPC for more control
- ❌ Do not store API keys in code — always `os.getenv()`
- ❌ Do not use `gpt-*` models — this is an Anthropic project
- ❌ Do not make the ingestion pipeline re-run on API startup — it's a one-time script
- ❌ Do not skip `input_type` in Voyage AI calls — it meaningfully affects retrieval quality
- ❌ Do not return the full state from LangGraph nodes — return only changed keys

---

## 13. How to Run (Full Setup)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill env
cp .env.example .env

# 3. Run Supabase SQL setup (paste into Supabase SQL editor)
# scripts/setup_supabase.sql

# 4. Ingest the docs (run once)
python scripts/ingest.py

# 5. Start the API
uvicorn api.main:app --reload --port 8000

# 6. Test it
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are XML tags used for in Claude prompts?"}'

# 7. Run evaluation
python evaluation/run_eval.py
```
