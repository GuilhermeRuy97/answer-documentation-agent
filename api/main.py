"""FastAPI application: middleware, rate limiting, static UI, and routes."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.routes import public_router, router
from api.security import limiter
from core.config import get_settings
from core.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting Anthropic RAG Agent API")
    logger.info(f"Auth: {'enabled' if settings.api_key_list() else 'disabled (set API_KEYS to enable)'}")
    logger.info("Graph compiled and ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Anthropic Prompt Engineering RAG Agent",
    description="Agentic RAG over Anthropic's prompt engineering documentation",
    version="2.0.0",
    lifespan=lifespan,
)

# Rate limiting (slowapi)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key"],
)

app.include_router(router)
app.include_router(public_router)

if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the chat UI, or a JSON hint when the UI is missing."""
    index = _STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(str(index))
    return JSONResponse({"message": "Anthropic RAG Agent is running. POST /ask to query."})
