import os
import logging
from typing import List

import voyageai

logger = logging.getLogger(__name__)

_api_key = os.getenv("VOYAGE_API_KEY")
if not _api_key:
    raise ValueError("VOYAGE_API_KEY environment variable is not set")

client = voyageai.Client(api_key=_api_key)

_BATCH_SIZE = 128


def embed_documents(texts: List[str]) -> List[List[float]]:
    n_batches = (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE
    logger.info(f"Embedding {len(texts)} documents in {n_batches} batches")

    embeddings: List[List[float]] = []
    for i in range(n_batches):
        batch = texts[i * _BATCH_SIZE : (i + 1) * _BATCH_SIZE]
        result = client.embed(batch, model="voyage-4", input_type="document")
        embeddings.extend(result.embeddings)

    return embeddings


def embed_query(text: str) -> List[float]:
    logger.info(f"Embedding query: {text[:60]}...")
    result = client.embed([text], model="voyage-4", input_type="query")
    return result.embeddings[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    docs = ["Prompt engineering is the art of crafting effective prompts.", "XML tags help structure Claude inputs."]
    doc_embeddings = embed_documents(docs)
    print(f"Document embedding dims: {len(doc_embeddings[0])}")
    print(f"First 5 dims: {doc_embeddings[0][:5]}")

    query_embedding = embed_query("What is prompt engineering?")
    print(f"Query embedding dims: {len(query_embedding)}")
    print(f"First 5 dims: {query_embedding[:5]}")
