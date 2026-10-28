"""Vector retrieval baseline for KG-vs-vector comparison.

Runs pure vector-retrieval RAG (ChromaDB search + LLM) over the evaluation
query set, using the same source data and LLM backbone as PNKG-QA, but with
no knowledge graph tools and no GraphRAG seeding.

Outputs responses.json compatible with --score-from for EvaluatorLLM scoring.

Usage:
    SPECIES=broiler .venv/Scripts/python -m src.evaluation.kg_vs_vector_baseline k=3
    SPECIES=layer   .venv/Scripts/python -m src.evaluation.kg_vs_vector_baseline k=5 output=logs/rag_layer.json
"""

import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

from openai import OpenAI

from src.config import Config
from src.rag.poultry_db import PoultryDB
from src.retry import llm_retry

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are a poultry nutrition assistant. Answer the user's question using ONLY the provided context retrieved from a database of poultry nutrition documents.

Rules:
1. Answer directly without meta-commentary (no "Based on the context...").
2. Cite specific values exactly as they appear in the context — do not round or modify.
3. Use tables for comparisons, bullet points for lists.
4. If the context does not contain enough information to answer, state "The provided context does not contain sufficient data to answer this question."
5. Do not invent values or use your own knowledge.
6. Keep answers concise and well-structured."""


def _load_queries(species: str) -> List[dict]:
    """Load evaluation queries from the curated query file for the given species."""
    from src.utils.query_loader import load_queries as ql

    # Temporarily override SPECIES to load the correct file
    orig_species = Config.SPECIES
    try:
        Config.SPECIES = species
        # Recompute the query file path for this species
        query_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "queries", species, "evaluation_queries_curated.json",
        )
        if not os.path.exists(query_file):
            logger.warning("Query file not found: %s", query_file)
            return []
        return ql(queries_file=query_file, query_set="all")
    finally:
        Config.SPECIES = orig_species


def _build_rag_prompt(query: str, context_docs: List[dict]) -> str:
    """Build a RAG prompt from retrieved documents."""
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        source = doc.get("source", "Unknown")
        content = doc.get("content", "")
        similarity = doc.get("similarity_score", 0.0)
        context_parts.append(
            f"[Document {i}] (source: {source}, relevance: {similarity:.3f})\n{content}"
        )

    context_text = "\n\n---\n\n".join(context_parts)
    return f"""Context from database:
{context_text}

Question: {query}

Answer:"""


def create_rag_client(k: int = 3) -> dict:
    """Create a RAG client that returns a callable for invoke(messages) -> dict.

    Uses PoultryDB for vector retrieval and Config.LLM_MODEL for generation.
    """
    db = PoultryDB()

    provider = "siliconflow"
    model = Config.LLM_MODEL
    api_key = Config.LLM_API_KEY
    base_url = Config.LLM_BASE_URL
    temperature = Config.TEMPERATURE
    max_tokens = Config.MAX_TOKENS
    extra_body = Config.LLM_EXTRA_BODY

    if not api_key:
        raise RuntimeError("LLM API key not configured — set SILICONFLOW_API_KEY in .env")

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=Config.API_TIMEOUT)

    @llm_retry
    def _llm_call(messages: List[dict]) -> dict:
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        return client.chat.completions.create(**kwargs)

    def invoke(messages: List[dict]) -> dict:
        """Extract user query -> retrieve -> generate -> return {response, tokens, ...}."""
        user_query = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        if not user_query:
            return {
                "response": "[Error] No user query found",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "model": model,
                "_pricing_key": f"{provider}::{model}",
            }

        # Retrieve top-k documents from ChromaDB
        try:
            retrieved = db.search(user_query, k=k)
        except Exception as e:
            logger.error("RAG search failed for query: %s", user_query[:80])
            logger.error("Error: %s", e)
            retrieved = []

        # Build RAG prompt
        rag_content = _build_rag_prompt(user_query, retrieved)
        rag_messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": rag_content},
        ]

        try:
            resp = _llm_call(rag_messages)
            usage = resp.usage
            return {
                "response": resp.choices[0].message.content or "",
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "model": model,
                "_pricing_key": f"{provider}::{model}",
            }
        except Exception as e:
            return {
                "response": f"[RAGError] {e}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "model": model,
                "_pricing_key": f"{provider}::{model}",
            }

    return invoke


def run_rag_baseline(
    queries: List[dict],
    k: int = 3,
    output_path: Optional[str] = None,
) -> Dict:
    """Run vector retrieval baseline on all queries.

    Args:
        queries: List of query dicts with keys (id, query, category, type).
        k: Number of top documents to retrieve from ChromaDB.
        output_path: If set, saves responses JSON to this path.

    Returns:
        Dict with keys: summary, per_query.
    """
    rag_invoke = create_rag_client(k=k)
    per_query_results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_time = 0.0

    for i, q in enumerate(queries):
        t_start = time.perf_counter()
        result = rag_invoke([
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": q["query"]},
        ])
        elapsed = time.perf_counter() - t_start

        prompt_tokens = result.get("prompt_tokens", 0)
        completion_tokens = result.get("completion_tokens", 0)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_time += elapsed

        entry = {
            "query_id": q.get("id", f"q{i:03d}"),
            "query": q["query"],
            "category": q.get("category", ""),
            "query_type": q.get("type", "single-value"),
            "species": q.get("species", Config.SPECIES),
            "response": result["response"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_s": round(elapsed, 3),
        }
        per_query_results.append(entry)

        if (i + 1) % 10 == 0:
            logger.info("Progress: %d/%d", i + 1, len(queries))

    n = len(per_query_results)
    summary = {
        "method": f"Vector-RAG (k={k})",
        "queries": n,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "avg_tokens": round((total_prompt_tokens + total_completion_tokens) / max(n, 1), 1),
        "avg_latency_s": round(total_time / max(n, 1), 3),
        "total_time_s": round(total_time, 3),
    }

    output = {
        "summary": summary,
        "per_query": per_query_results,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info("Saved RAG baseline results -> %s", output_path)

    return output


def _parse_cli_args(argv: List[str]) -> dict:
    """Parse key=value CLI arguments."""
    args = {}
    for arg in argv:
        if "=" in arg:
            key, val = arg.split("=", 1)
            args[key.strip()] = val.strip()
        elif arg.isdigit():
            args["k"] = arg
        else:
            args["output"] = arg
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cli = _parse_cli_args(sys.argv[1:])
    species = os.environ.get("SPECIES", "broiler")
    k = int(cli.get("k", 3))
    output_path = cli.get(
        "output",
        f"logs/rag_baseline_{species}_k{k}.json",
    )

    all_queries = _load_queries(species)
    if not all_queries:
        logger.error("No queries found for species=%s", species)
        sys.exit(1)

    logger.info("Running vector retrieval baseline: species=%s, k=%d, n=%d", species, k, len(all_queries))

    result = run_rag_baseline(all_queries, k=k, output_path=output_path)

    s = result["summary"]
    print(f"\n=== Vector-RAG Baseline (k={k}, species={species}) ===")
    print(f"  Queries: {s['queries']}")
    print(f"  Avg tokens/query: {s['avg_tokens']}")
    print(f"  Avg latency: {s['avg_latency_s']} s")
    print(f"  Total time: {s['total_time_s']} s")
    print(f"  Output: {output_path}")
