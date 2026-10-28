"""Single-tool ReAct baseline for PNKG-QA comparison.

A minimalist ReAct agent that uses ONE retrieval tool (vector search over
PoultryDB) without any graph-specific tooling or GraphRAG seeding. This
isolates the contribution of the four KG tools and the structured graph
context from the LLM's parametric knowledge + flat retrieval.

Architecture (for each query):
  Round 1: LLM decides whether to call search_poultry_db(query) or finish
  Round 2: LLM observes results, may call again with refined query
  Round 3: LLM produces final answer
  Max iterations: 3

This is the same "agent + flat retrieval" baseline that LangChain ReAct
tutorials ship by default. It serves as a third control beyond Vector-RAG
(no graph) and Tools-off PNKG (no graph tools but with GraphRAG context).

Usage:
    SPECIES=broiler .venv/Scripts/python -m src.evaluation.react_baseline
    SPECIES=layer   .venv/Scripts/python -m src.evaluation.react_baseline output=logs/react_layer.json
"""

import json
import logging
import os
import re
import sys
import time
from typing import Dict, List, Optional

from openai import OpenAI

from src.config import Config
from src.rag.poultry_db import PoultryDB
from src.retry import llm_retry

logger = logging.getLogger(__name__)

REACT_SYSTEM_PROMPT = """You are a poultry nutrition assistant that uses a ReAct (Reasoning + Acting) loop with ONE tool: search_poultry_db(query).

Available tool:
  search_poultry_db(query: str) -> str
    Search the poultry nutrition database for documents relevant to the query.
    Returns the top-3 matching document excerpts.

Process:
  1. Think about what information you need to answer the question.
  2. If the database might contain it, call search_poultry_db with a focused query.
  3. Read the returned excerpts and decide: do you have enough to answer?
  4. If yes, write "Final Answer: <your answer>".
  5. If no, you may call search_poultry_db again with a different query (max 3 total tool calls).

Output format (strict):
  Thought: <one sentence reasoning>
  Action: search_poultry_db("<your query>")
  -- after receiving Observation --
  Thought: <reasoning about whether to call again or finish>
  Action: search_poultry_db("<refined query>")
  -- repeat as needed --
  Thought: I have enough information to answer.
  Final Answer: <direct answer without meta-commentary>

Rules:
- Cite specific values exactly as they appear in observations.
- Use tables for comparisons, bullet points for lists.
- If observations don't contain enough data, say "The provided context does not contain sufficient data to answer this question."
- Do not invent values or use your own knowledge beyond what observations provide.
- Keep final answers concise and well-structured."""


def _load_queries(species: str) -> List[dict]:
    """Load evaluation queries from the curated query file for the given species."""
    from src.utils.query_loader import load_queries as ql

    orig_species = Config.SPECIES
    try:
        Config.SPECIES = species
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


def _retrieve_docs(db: PoultryDB, query: str, k: int = 3) -> str:
    """Retrieve top-k docs and format as observation text."""
    try:
        results = db.search(query, k=k)
    except Exception as e:
        logger.error("PoultryDB search failed: %s", e)
        return f"[SearchError] {e}"

    parts = []
    for i, doc in enumerate(results, 1):
        source = doc.get("source", "Unknown")
        content = doc.get("content", "")
        similarity = doc.get("similarity_score", 0.0)
        parts.append(
            f"[Document {i}] (source: {source}, relevance: {similarity:.3f})\n{content}"
        )
    return "\n\n---\n\n".join(parts) if parts else "[No documents found]"


def _extract_action_query(text: str) -> Optional[str]:
    """Parse 'Action: search_poultry_db("query")' from LLM response. Returns query or None."""
    m = re.search(r'Action:\s*search_poultry_db\s*\(\s*"([^"]+)"\s*\)', text)
    if m:
        return m.group(1)
    m = re.search(r"Action:\s*search_poultry_db\s*\(\s*'([^']+)'\s*\)", text)
    if m:
        return m.group(1)
    m = re.search(r"Action:\s*search_poultry_db\s*\(\s*([^)]+)\s*\)", text)
    if m:
        return m.group(1).strip().strip('"').strip("'")
    return None


def _extract_final_answer(text: str) -> Optional[str]:
    """Parse 'Final Answer: ...' from LLM response. Returns the answer text or None."""
    m = re.search(r"Final Answer:\s*(.+?)(?:\n\n|\Z)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _has_final_answer(text: str) -> bool:
    return bool(re.search(r"Final Answer\s*:", text))


def create_react_client(db: Optional[PoultryDB] = None, k: int = 3, max_iterations: int = 3) -> dict:
    """Create a ReAct baseline client. Returns invoke(messages) -> dict.

    The single tool is search_poultry_db(query), backed by vector retrieval.
    """
    if db is None:
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
        """Run ReAct loop. messages[0] is system, messages[-1] is user."""
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

        scratchpad: List[dict] = [{"role": "system", "content": REACT_SYSTEM_PROMPT}]
        scratchpad.append({"role": "user", "content": user_query})

        total_prompt_tokens = 0
        total_completion_tokens = 0
        final_text = ""

        for iteration in range(max_iterations):
            try:
                resp = _llm_call(scratchpad)
            except Exception as e:
                return {
                    "response": f"[LLMError] iteration={iteration} error={e}",
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "model": model,
                    "_pricing_key": f"{provider}::{model}",
                }

            usage = resp.usage
            if usage:
                total_prompt_tokens += usage.prompt_tokens or 0
                total_completion_tokens += usage.completion_tokens or 0

            assistant_text = resp.choices[0].message.content or ""
            final_text = assistant_text
            scratchpad.append({"role": "assistant", "content": assistant_text})

            if _has_final_answer(assistant_text):
                break

            query = _extract_action_query(assistant_text)
            if query is None:
                scratchpad.append({
                    "role": "user",
                    "content": "No valid Action found. Either call search_poultry_db with a query in parentheses, or produce a Final Answer.",
                })
                continue

            observation = _retrieve_docs(db, query, k=k)
            scratchpad.append({
                "role": "user",
                "content": f"Observation: {observation}",
            })
        else:
            scratchpad.append({
                "role": "user",
                "content": "You have used all available tool calls. Based on the observations above, produce your Final Answer now (or state that the data is insufficient).",
            })
            try:
                resp = _llm_call(scratchpad)
                usage = resp.usage
                if usage:
                    total_prompt_tokens += usage.prompt_tokens or 0
                    total_completion_tokens += usage.completion_tokens or 0
                final_text = resp.choices[0].message.content or final_text
            except Exception as e:
                logger.warning("Final-answer LLM call failed: %s", e)

        final_answer = _extract_final_answer(final_text)
        if final_answer is None:
            final_answer = final_text if final_text else "[No response]"

        return {
            "response": final_answer,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "model": model,
            "_pricing_key": f"{provider}::{model}",
        }

    return invoke


def run_react_baseline(
    queries: List[dict],
    k: int = 3,
    max_iterations: int = 3,
    output_path: Optional[str] = None,
) -> Dict:
    """Run single-tool ReAct baseline on all queries."""
    db = PoultryDB()
    react_invoke = create_react_client(db=db, k=k, max_iterations=max_iterations)
    per_query_results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_time = 0.0

    for i, q in enumerate(queries):
        t_start = time.perf_counter()
        result = react_invoke([
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
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
        "method": f"Single-Tool ReAct (k={k}, max_iter={max_iterations})",
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
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info("Saved ReAct baseline results -> %s", output_path)

    return output


def _parse_cli_args(argv: List[str]) -> dict:
    """Parse key=value CLI arguments."""
    args = {}
    for arg in argv:
        if "=" in arg:
            key, val = arg.split("=", 1)
            args[key.strip()] = val.strip()
        else:
            args["output"] = arg
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cli = _parse_cli_args(sys.argv[1:])
    species = os.environ.get("SPECIES", "broiler")
    k = int(cli.get("k", 3))
    max_iter = int(cli.get("max_iter", 3))
    n_limit = int(cli["n"]) if "n" in cli else None
    output_path = cli.get(
        "output",
        f"logs/react_baseline_{species}.json",
    )

    all_queries = _load_queries(species)
    if not all_queries:
        logger.error("No queries found for species=%s", species)
        sys.exit(1)

    if n_limit is not None and n_limit < len(all_queries):
        all_queries = all_queries[:n_limit]
        logger.info("Limited to first %d queries", n_limit)

    logger.info(
        "Running ReAct baseline: species=%s, k=%d, max_iter=%d, n=%d",
        species, k, max_iter, len(all_queries),
    )

    result = run_react_baseline(
        all_queries,
        k=k,
        max_iterations=max_iter,
        output_path=output_path,
    )

    s = result["summary"]
    print(f"\n=== Single-Tool ReAct Baseline (k={k}, species={species}) ===")
    print(f"  Queries: {s['queries']}")
    print(f"  Avg tokens/query: {s['avg_tokens']}")
    print(f"  Avg latency: {s['avg_latency_s']} s")
    print(f"  Total time: {s['total_time_s']} s")
    print(f"  Output: {output_path}")
