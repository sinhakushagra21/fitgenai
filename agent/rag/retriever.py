"""
agent/rag/retriever.py
───────────────────────
Vector retriever for the FITGEN.AI knowledge base.

Embeds all knowledge documents using OpenAI's text-embedding model,
stores them in a FAISS index when available (or NumPy fallback), and
provides a `retrieve()` function that returns top-k relevant documents.

Usage:
    from agent.rag.retriever import retrieve
    docs = retrieve("How much protein should I eat?", k=3)
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from agent.rag.knowledge_base import get_all_documents

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

load_dotenv()

# ── Config ────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Global state (lazy-loaded) ────────────────────────────────────
_index: Any | None = None
_embeddings: np.ndarray | None = None
_docs: list[dict] | None = None
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts using OpenAI's embedding model."""
    client = _get_client()
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
    )
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings, dtype=np.float32)


def _cache_key(docs: list[dict]) -> str:
    """Create a hash of document contents for cache invalidation."""
    content = "".join(d["id"] + d["content"] for d in docs)
    return hashlib.md5(content.encode()).hexdigest()


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms


def _build_index() -> tuple[Any, list[dict], np.ndarray | None]:
    """Build (or load cached) vector store from knowledge base."""
    global _index, _docs, _embeddings

    docs = get_all_documents()
    cache_hash = _cache_key(docs)
    index_path = CACHE_DIR / f"index_{cache_hash}.bin"
    docs_path = CACHE_DIR / f"docs_{cache_hash}.pkl"
    emb_path = CACHE_DIR / f"embeddings_{cache_hash}.npy"

    if docs_path.exists() and index_path.exists() and faiss is not None:
        _index = faiss.read_index(str(index_path))
        _embeddings = None
        with open(docs_path, "rb") as file:
            _docs = pickle.load(file)
        return _index, _docs, _embeddings

    if docs_path.exists() and emb_path.exists() and faiss is None:
        _index = None
        _embeddings = np.load(emb_path)
        with open(docs_path, "rb") as file:
            _docs = pickle.load(file)
        return _index, _docs, _embeddings

    texts = [f"{doc['title']}: {doc['content']}" for doc in docs]
    embeddings = _l2_normalize(_embed_texts(texts))

    with open(docs_path, "wb") as file:
        pickle.dump(docs, file)

    if faiss is not None:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(embeddings)
        faiss.write_index(index, str(index_path))
        _index = index
        _embeddings = None
    else:
        np.save(emb_path, embeddings)
        _index = None
        _embeddings = embeddings

    _docs = docs
    return _index, _docs, _embeddings


def retrieve(query: str, k: int = 3) -> list[dict]:
    """Retrieve the top-k most relevant documents for a query.

    Parameters
    ----------
    query : str
        The user's question.
    k : int
        Number of documents to return (default 3).

    Returns
    -------
    list[dict]
        Each dict contains: id, topic, title, content, source, score.
    """
    index, docs, embeddings = _build_index()

    top_k = min(k, len(docs))
    if top_k <= 0:
        return []

    query_vec = _l2_normalize(_embed_texts([query]))

    if faiss is not None and index is not None:
        scores, indices = index.search(query_vec, top_k)
    else:
        if embeddings is None:
            return []
        similarities = embeddings @ query_vec[0]
        order = np.argsort(-similarities)[:top_k]
        scores = np.array([similarities[order]], dtype=np.float32)
        indices = np.array([order], dtype=np.int64)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        doc = docs[idx].copy()
        doc["score"] = round(float(score), 4)
        results.append(doc)

    return results


def format_context(docs: list[dict]) -> str:
    """Format retrieved documents as context string for the LLM."""
    if not docs:
        return ""

    parts = ["## Retrieved Evidence\n"]
    for i, doc in enumerate(docs, 1):
        parts.append(
            f"### [{i}] {doc['title']} (relevance: {doc['score']:.2f})\n"
            f"{doc['content']}\n"
            f"*Source: {doc['source']}*\n"
        )
    return "\n".join(parts)
