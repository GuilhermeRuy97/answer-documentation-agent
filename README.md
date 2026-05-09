# Anthropic Prompt Engineering RAG Agent

An **Agentic RAG system** that answers questions over Anthropic's Prompt Engineering
documentation using LangGraph, Supabase + PGVector, Voyage AI embeddings, and Claude.

---

## Architecture

```
User → FastAPI POST /ask → LangGraph Agent
                                │
                    ┌───────────▼───────────┐
                    │   rewrite_query        │  Claude generates query variants
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   search_docs (tool)   │  Voyage AI embed → PGVector search
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   grade_relevance      │  Score < 0.7? Retry retrieval
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   generate_answer      │  Claude answers with context
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   format_citations     │  Deduplicate + number sources
                    └───────────┬───────────┘
                                │
                         JSON Response
```

## Tech Stack

- **LLM**: Claude claude-sonnet-4-5 (Anthropic)
- **Embeddings**: Voyage AI `voyage-3-lite` (512 dims)
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

## Tradeoffs & What I'd Build Next

- **Session storage**: Currently in-memory. Production: Redis via `RedisChatMessageHistory`
- **Reranking**: Would add Cohere or Voyage reranker between retrieval and generation
- **Streaming**: FastAPI SSE for real-time token streaming
- **Option C**: MCP server exposing `search_docs` as an MCP tool for Claude Code/Desktop
- **Hybrid search**: Combine PGVector similarity with BM25 full-text search for better recall
