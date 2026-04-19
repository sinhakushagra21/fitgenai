"""
agent/rag/personal/embedder.py
──────────────────────────────
OpenAI embedding wrapper used by the Personal RAG indexer + retriever.

  * Model: configurable via FITGEN_EMBEDDING_MODEL env
           (default ``text-embedding-3-small``, 1536 dims)
  * Batches up to 2048 inputs per API call
  * Exponential-backoff retry on transient errors
  * Returns a deterministic list ordered to match the input
"""

from __future__ import annotations

import logging
import os
import time
from typing import Iterable

from openai import OpenAI

logger = logging.getLogger("fitgen.rag.personal.embedder")

_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_DIMS = 1536
_MAX_BATCH = 256             # safe batch size (API ceiling is 2048)
_MAX_RETRIES = 5

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _embedding_model() -> str:
    return os.getenv("FITGEN_EMBEDDING_MODEL", _DEFAULT_MODEL)


def embedding_dimensions() -> int:
    """Dimensionality of the configured embedding model."""
    try:
        return int(os.getenv("FITGEN_EMBEDDING_DIMS", str(_DEFAULT_DIMS)))
    except ValueError:
        return _DEFAULT_DIMS


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    """Embed a list of strings; returns same-length list of vectors.

    Empty strings are embedded as zero-vectors to preserve index alignment.
    """
    items = list(texts)
    if not items:
        return []

    vectors: list[list[float]] = [[] for _ in items]
    model = _embedding_model()
    client = _get_client()

    # Gather indices of non-empty inputs so we don't waste API quota.
    to_send_indices: list[int] = [
        i for i, t in enumerate(items) if t and t.strip()
    ]

    for batch_start in range(0, len(to_send_indices), _MAX_BATCH):
        batch_idx = to_send_indices[batch_start:batch_start + _MAX_BATCH]
        batch_inputs = [items[i] for i in batch_idx]

        attempt = 0
        while True:
            try:
                resp = client.embeddings.create(
                    model=model, input=batch_inputs,
                )
                for local_i, datum in enumerate(resp.data):
                    vectors[batch_idx[local_i]] = list(datum.embedding)
                logger.info(
                    "[embedder] model=%s batch=%d tokens_usage=%s",
                    model, len(batch_inputs),
                    getattr(resp, "usage", None),
                )
                break
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                if attempt >= _MAX_RETRIES:
                    logger.error(
                        "[embedder] giving up after %d attempts: %s",
                        attempt, exc,
                    )
                    raise
                delay = min(30.0, 1.5 * (2 ** (attempt - 1)))
                logger.warning(
                    "[embedder] attempt %d failed (%s) — retry in %.1fs",
                    attempt, exc, delay,
                )
                time.sleep(delay)

    # Fill zero-vectors for empty inputs so caller sees aligned length.
    dims = embedding_dimensions()
    for i, v in enumerate(vectors):
        if not v:
            vectors[i] = [0.0] * dims
    return vectors


def embed_query(text: str) -> list[float]:
    """Convenience wrapper for single-query retrieval-time embeds."""
    if not text or not text.strip():
        return [0.0] * embedding_dimensions()
    return embed_texts([text])[0]
