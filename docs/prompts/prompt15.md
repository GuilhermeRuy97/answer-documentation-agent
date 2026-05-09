# ───────────────────────────────────────────────
# FILE 15: api/main.py
# ───────────────────────────────────────────────
"""
Create `api/main.py`.

FastAPI app with lifespan and router mounting.

Requirements:
- Import: FastAPI, from contextlib asynccontextmanager
- Import: dotenv load_dotenv (call at top)
- Import: router from api.routes
- Import: logging

- Configure logging at module level

- @asynccontextmanager
  async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting Anthropic RAG Agent API")
    logging.info("Graph compiled and ready")
    yield
    # Shutdown
    logging.info("Shutting down")

- app = FastAPI(
    title="Anthropic Prompt Engineering RAG Agent",
    description="Agentic RAG over Anthropic's prompt engineering documentation",
    version="1.0.0",
    lifespan=lifespan
  )

- app.include_router(router)

- Add a root GET / that returns {"message": "Anthropic RAG Agent is running. POST /ask to query."}
"""