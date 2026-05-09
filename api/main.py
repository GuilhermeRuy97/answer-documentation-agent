import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from api.routes import router

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting Anthropic RAG Agent API")
    logging.info("Graph compiled and ready")
    yield
    logging.info("Shutting down")


app = FastAPI(
    title="Anthropic Prompt Engineering RAG Agent",
    description="Agentic RAG over Anthropic's prompt engineering documentation",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/")
async def root():
    return {"message": "Anthropic RAG Agent is running. POST /ask to query."}
