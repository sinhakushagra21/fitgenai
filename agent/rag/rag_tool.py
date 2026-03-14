"""
agent/rag/rag_tool.py
──────────────────────
RAG-augmented wrapper tool for FITGEN.AI.

Provides a `rag_query_tool` that:
  1. Retrieves relevant knowledge from the FAISS index
  2. Augments the LLM prompt with retrieved context
  3. Generates a response grounded in evidence

This tool can be used standalone or integrated into the existing
agent graph as an additional tool alongside workout_tool and diet_tool.

Usage:
    from agent.rag.rag_tool import rag_query_tool
"""

from __future__ import annotations

import json
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from agent.rag.retriever import retrieve, format_context

logger = logging.getLogger("fitgen.rag_tool")

MODEL = "gpt-5-mini"

_RAG_SYSTEM_PROMPT = """You are FITGEN.AI, an expert fitness and nutrition coach powered by
Retrieval-Augmented Generation (RAG).

You have been provided with relevant evidence from a curated knowledge base
below. Use this evidence to ground your response with accurate, cited facts.

INSTRUCTIONS:
1. Base your answer primarily on the retrieved evidence.
2. Cite sources using [1], [2], etc. corresponding to the evidence numbers.
3. If the evidence doesn't fully cover the query, supplement with your
   general knowledge but clearly indicate which parts are from evidence
   vs. general knowledge.
4. Provide practical, actionable advice.
5. If the query is not fitness/nutrition-related, politely redirect.

{context}
"""


@tool
def rag_query_tool(query: str) -> str:
    """Answer a fitness or nutrition question using RAG (Retrieval-Augmented Generation).

    This tool retrieves relevant evidence from a curated knowledge base
    of peer-reviewed fitness and nutrition research, then generates
    an evidence-grounded response.

    Use this for queries that benefit from factual, cited responses:
    - Scientific questions about training principles
    - Evidence-based nutrition recommendations
    - Safety thresholds and medical guidelines
    - Supplement efficacy and dosage

    Args:
        query: The user's fitness or nutrition question.

    Returns:
        A JSON object with the RAG response and retrieved sources.
    """
    logger.info("[RAGTool] Query: %s", query[:80])
    t0 = time.perf_counter()

    # 1. Retrieve relevant documents
    docs = retrieve(query, k=3)
    context = format_context(docs)
    logger.info("[RAGTool] Retrieved %d documents", len(docs))

    # 2. Build augmented prompt
    system = _RAG_SYSTEM_PROMPT.format(context=context)

    # 3. Generate response
    llm = ChatOpenAI(model=MODEL, temperature=0.3)
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=query),
    ])

    elapsed = time.perf_counter() - t0
    logger.info("[RAGTool] Response generated in %.2fs", elapsed)

    # 4. Package result
    result = {
        "response": response.content,
        "sources": [
            {"title": d["title"], "source": d["source"], "score": d["score"]}
            for d in docs
        ],
        "retrieval_time_s": round(elapsed, 3),
    }

    return json.dumps(result, ensure_ascii=False)
