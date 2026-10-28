"""NutriCHAT ablation experiment framework — focused on the Multi-Agent architecture

Core contribution: a Multi-Agent system (Router → Specialists → Synthesis)
The ablation dimensions are designed around the multi-agent architecture.

Dimension 1: Agent architecture ablation
  Single ReAct vs Multi-Agent (core comparison)
  Validates the value of the multi-agent architecture

Dimension 2: Specialist ablation
  Remove one specialist at a time to measure marginal contribution
  Understand the irreplaceability of each specialist

Dimension 3: Router strategy ablation
  LLM Router (current) vs Keyword Fallback vs Broadcast (invoke all)
  Validates the effect of routing precision on performance

Dimension 4: RAG ablation (Multi-Agent context)
  Multi-Agent with RAG vs without RAG
  Tests RAG's contribution within the multi-agent architecture

Usage:
    python main.py --ablation                 # all dimensions
    python main.py --ablation --ablation-dim architecture  # single dimension
    python main.py --ablation --ablation-dim specialist
    python main.py --ablation --ablation-dim synthesis
    python main.py --ablation --ablation-dim llm_backbone
    python main.py --ablation --ablation-dim language
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeout
from typing import Dict, List, Optional

from tqdm import tqdm

from src.config import Config
from src.logger import get_logger
from src.tools import get_all_tools, TOOL_DOMAINS


# ── Paths ───────────────────────────────────────────────────

_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)


# ── Query loading ───────────────────────────────────────────


# ── Variant runners ─────────────────────────────────────────

# ── Global response cache: reuse the same query×variant (shared across dimensions + persisted to disk) ──
# Nested structure: {variant_name: {query_text: response, ...}, ...}, also used as responses.json
import threading as _threading
_ABLATION_RESPONSE_CACHE: Dict[str, Dict[str, str]] = {}
_ABLATION_CACHE_FILE: str = ""  # set by run_ablation
_ABLATION_CACHE_LOCK = _threading.Lock()

from src.config import Config as _Cfg
# Concurrent query count: halved to avoid 429 rate limits from nested thread pools
# (outer variants in parallel + inner queries in parallel; total concurrency = outer × inner)
_ABLATION_MAX_WORKERS = max(1, _Cfg.API_MAX_CONCURRENT // 2)
_ABLATION_QUERY_TIMEOUT = 120  # per-query timeout (seconds)


def _load_disk_cache():
    """Load persisted cache from disk (if any)."""
    global _ABLATION_RESPONSE_CACHE
    if not _ABLATION_CACHE_FILE or not os.path.exists(_ABLATION_CACHE_FILE):
        return
    try:
        import json
        with _ABLATION_CACHE_LOCK:
            with open(_ABLATION_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Normalize from flat {variant|||query: resp} to nested {variant: {query: resp}}
            if data and any("|||" in k for k in data):
                nested = {}
                for k, v in data.items():
                    parts = k.split("|||", 1)
                    if len(parts) == 2:
                        nested.setdefault(parts[0], {})[parts[1]] = v
                data = nested
            _ABLATION_RESPONSE_CACHE.update(data)
        total = sum(len(v) for v in _ABLATION_RESPONSE_CACHE.values())
        print(f"  [Cache] Loaded {total} cached responses ({len(_ABLATION_RESPONSE_CACHE)} variants)")
    except Exception as e:
        print(f"  [Cache] Failed to load cache: {e}")


def _save_disk_cache():
    """Save cache to disk (thread-safe) as nested {variant: {query: resp}}."""
    if not _ABLATION_CACHE_FILE:
        return
    try:
        import json
        with _ABLATION_CACHE_LOCK:
            with open(_ABLATION_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(_ABLATION_RESPONSE_CACHE, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  [Cache] Failed to save cache: {e}")


def _run_concurrent(queries: List[dict], worker_fn, desc: str,
                    variant_name: str = "",
                    return_steps: bool = False):
    """Generic concurrent query executor with caching + timeout.

    Args:
        queries: query list
        worker_fn: (query_dict) -> str or (query_dict) -> (str, list)
        desc: tqdm description text
        variant_name: variant name, used as the cache key
        return_steps: if True, worker_fn returns (response, steps_list)

    Returns:
        return_steps=False: List[str]
        return_steps=True: (List[str], List[List[dict]])
    """
    responses: List[str] = [""] * len(queries)
    step_traces: List[list] = [[] for _ in queries]
    with ThreadPoolExecutor(max_workers=_ABLATION_MAX_WORKERS) as pool:
        futures = {}
        for i, q in enumerate(queries):
            qt = q["query"]
            with _ABLATION_CACHE_LOCK:
                cached = _ABLATION_RESPONSE_CACHE.get(variant_name, {}).get(qt)
            if cached is not None:
                if isinstance(cached, dict) and "resp" in cached:
                    responses[i] = cached["resp"]
                    if return_steps:
                        step_traces[i] = cached.get("trace", [])
                else:
                    responses[i] = cached
            else:
                futures[pool.submit(worker_fn, q)] = i

        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="q"):
            i = futures[future]
            try:
                result = future.result(timeout=_ABLATION_QUERY_TIMEOUT)
                if return_steps:
                    resp, steps = result
                    responses[i] = resp
                    step_traces[i] = steps
                    cache_val = {"resp": resp, "trace": steps}
                else:
                    responses[i] = result
                    cache_val = result
                qt = queries[i]["query"]
                with _ABLATION_CACHE_LOCK:
                    _ABLATION_RESPONSE_CACHE.setdefault(variant_name, {})[qt] = cache_val
            except FutureTimeout:
                responses[i] = f"[Timeout] Query exceeded {_ABLATION_QUERY_TIMEOUT}s"
                if return_steps:
                    step_traces[i] = []
            except Exception as e:
                responses[i] = f"[Error] {e}"
                if return_steps:
                    step_traces[i] = []

    if return_steps:
        return responses, step_traces
    return responses


def _warmup_llm():
    """Warm up the LLM + NLI + Embedding models so cold starts do not slow down the first query."""
    logger = get_logger("ablation")
    from concurrent.futures import ThreadPoolExecutor

    def _load_llm():
        from src.agent.base_agent import FixedChatOpenAI
        from src.config import Config
        kwargs = dict(
            model=Config.LLM_MODEL,
            temperature=0.0,
            max_tokens=10,
            openai_api_key=Config.LLM_API_KEY,
            openai_api_base=Config.LLM_BASE_URL,
            timeout=Config.API_TIMEOUT,
        )
        if Config.LLM_EXTRA_BODY:
            kwargs["extra_body"] = Config.LLM_EXTRA_BODY
        llm = FixedChatOpenAI(**kwargs)
        llm.invoke("Hi")

    def _load_nli():
        from src.evaluation.selfcheck import SelfCheckGPT
        from src.config import Config as _Cfg
        sc = SelfCheckGPT(device=_Cfg.NLI_DEVICE)
        sc._load_model()

    def _load_embedding():
        from src.rag.poultry_db import create_embedding_model
        create_embedding_model()

    logger.info("Pre-loading models (LLM + NLI + Embedding)...")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {
            pool.submit(_load_llm): "LLM",
            pool.submit(_load_nli): "NLI",
            pool.submit(_load_embedding): "Embedding",
        }
        from concurrent.futures import as_completed
        for f in as_completed(futs):
            name = futs[f]
            try:
                f.result()
                logger.info("  %s ready", name)
            except Exception as e:
                logger.warning("  %s failed (non-critical): %s", name, e)


def _run_single_react(queries: List[dict], tools=None, verbose: bool = False,
                       variant_name: str = "", model: str = None,
                       base_url: str = None, api_key: str = None,
                       extra_body: dict = None):
    """Single ReAct Agent baseline"""
    from src.agent.react_agent import ReActAgent

    if tools is None:
        tools = get_all_tools()

    agent = ReActAgent(tools=tools, verbose=verbose, model=model,
                       base_url=base_url, api_key=api_key,
                       extra_body=extra_body)

    def _do(q):
        return agent.run(q["query"])  # (resp, steps)

    return _run_concurrent(queries, _do, "  Single ReAct",
                           variant_name=variant_name, return_steps=True)


def _run_multi_agent(queries: List[dict], tools=None, rag_enabled: bool = True,
                     verbose: bool = False, variant_name: str = "",
                     model: str = None, synthesis_enabled: bool = True,
                     base_url: str = None, api_key: str = None,
                     extra_body: dict = None,
                     validation_mode: str = "auto",
                     synthesis_mode: str = "synthesis",
                     single_pass: bool = False):
    """Full Multi-Agent architecture (Router → Specialists → Synthesis)"""
    from src.agent.multi_agent_engine import MultiAgentEngine

    if tools is None:
        tools = get_all_tools()

    engine = MultiAgentEngine(tools=tools, verbose=verbose, rag_enabled=rag_enabled,
                              model=model, synthesis_enabled=synthesis_enabled,
                              validation_mode=validation_mode,
                              synthesis_mode=synthesis_mode,
                              single_pass=single_pass,
                              base_url=base_url, api_key=api_key,
                              extra_body=extra_body)

    def _do(q):
        result = engine.answer(q["query"])
        return (result["response"], result.get("steps", []))

    return _run_concurrent(queries, _do, "  Multi-Agent",
                           variant_name=variant_name, return_steps=True)


def _run_multi_agent_no_rag(queries: List[dict], tools=None, verbose: bool = False,
                              variant_name: str = "", model: str = None,
                              synthesis_enabled: bool = True,
                              base_url: str = None, api_key: str = None,
                              extra_body: dict = None):
    """Multi-Agent without RAG (Dimension 4 ablation)"""
    return _run_multi_agent(queries, tools=tools, rag_enabled=False, verbose=verbose,
                             variant_name=variant_name, model=model,
                             synthesis_enabled=synthesis_enabled,
                             base_url=base_url, api_key=api_key,
                             extra_body=extra_body)


def _run_multi_agent_no_tools(queries: List[dict], tools=None, verbose: bool = False,
                              variant_name: str = "", model: str = None,
                              synthesis_enabled: bool = True,
                              base_url: str = None, api_key: str = None,
                              extra_body: dict = None):
    """Multi-Agent without tools: specialists answer from parametric knowledge only,
    validating the marginal contribution of routing + synthesis.

    This is NOT the no-ReAct variant — it removes tools entirely. It used to be
    mislabelled "Multi-Agent (no ReAct)", which also happened to be dispatched
    to the full pipeline because of a case-sensitive string check (see dispatch validation).
    """
    del tools  # unused — no tools means no tool access at all
    from src.agent.multi_agent_engine import MultiAgentEngine

    engine = MultiAgentEngine(tools=[], verbose=verbose, rag_enabled=True,
                              model=model, synthesis_enabled=synthesis_enabled,
                              base_url=base_url, api_key=api_key,
                              extra_body=extra_body)

    def _do(q):
        result = engine.answer(q["query"])
        return (result["response"], result.get("steps", []))

    return _run_concurrent(queries, _do, "  Multi-Agent (no tools)",
                           variant_name=variant_name, return_steps=True)


def _run_multi_agent_single_pass(queries: List[dict], tools=None,
                                 verbose: bool = False,
                                 variant_name: str = "", model: str = None,
                                 synthesis_enabled: bool = True,
                                 base_url: str = None, api_key: str = None,
                                 extra_body: dict = None):
    """Multi-Agent true no-ReAct: each specialist makes exactly one tool call, then answers directly.

    This is the true "no iterative reasoning" control: the specialist may call
    exactly one tool, receives that observation, and must answer without any
    further tool access. It replaces the previously reported (and invalid)
    "Multi-Agent (no ReAct)" row.
    """
    from src.agent.multi_agent_engine import MultiAgentEngine

    if tools is None:
        tools = get_all_tools()

    engine = MultiAgentEngine(tools=tools, verbose=verbose, rag_enabled=True,
                              model=model, synthesis_enabled=synthesis_enabled,
                              single_pass=True,
                              base_url=base_url, api_key=api_key,
                              extra_body=extra_body)

    def _do(q):
        result = engine.answer(q["query"])
        return (result["response"], result.get("steps", []))

    return _run_concurrent(queries, _do, "  Multi-Agent (single-pass)",
                           variant_name=variant_name, return_steps=True)


# ── Factorial contrasts (E1 architecture / E2 synthesis) ─────

def _run_standalone_llm(queries: List[dict], tools=None, verbose: bool = False,
                        variant_name: str = "", model: str = None,
                        base_url: str = None, api_key: str = None,
                        extra_body: dict = None):
    """Plain LLM: no tools, no RAG, no agent loop (data-access control)."""
    from langchain_core.messages import HumanMessage
    from src.agent.base_agent import FixedChatOpenAI

    kwargs = dict(
        model=model or Config.LLM_MODEL,
        temperature=0.0,
        max_tokens=Config.MAX_TOKENS,
        openai_api_key=api_key or Config.LLM_API_KEY,
        openai_api_base=base_url or Config.LLM_BASE_URL,
        timeout=Config.API_TIMEOUT,
    )
    eb = extra_body or Config.LLM_EXTRA_BODY
    if eb:
        kwargs["extra_body"] = eb
    llm = FixedChatOpenAI(**kwargs)

    def _do(q):
        resp = llm.invoke([HumanMessage(content=q["query"])])
        return ((resp.content or "").strip(), [])

    return _run_concurrent(queries, _do, "  Standalone LLM",
                           variant_name=variant_name, return_steps=True)


def _run_single_react_rag(queries: List[dict], tools=None, verbose: bool = False,
                          variant_name: str = "", model: str = None,
                          base_url: str = None, api_key: str = None,
                          extra_body: dict = None):
    """Single ReAct agent with all tools plus RAG context prepended."""
    from src.agent.react_agent import ReActAgent
    from src.rag.poultry_db import PoultryDB

    if tools is None:
        tools = get_all_tools()
    agent = ReActAgent(tools=tools, verbose=verbose, model=model,
                       base_url=base_url, api_key=api_key,
                       extra_body=extra_body)
    db = PoultryDB()

    def _do(q):
        try:
            docs = db.search(q["query"], k=Config.RAG_K)
        except Exception:
            docs = []
        ctx = "\n\n".join(
            f"[{d.get('source', '')}] {d.get('content', '')}" for d in docs
        )
        augmented = (
            f"Reference context:\n{ctx}\n\nQuestion: {q['query']}"
            if ctx else q["query"]
        )
        return agent.run(augmented)

    return _run_concurrent(queries, _do, "  Single ReAct + RAG",
                           variant_name=variant_name, return_steps=True)


def _run_single_react_rag_synthesis(queries: List[dict], tools=None,
                                    verbose: bool = False, variant_name: str = "",
                                    model: str = None, base_url: str = None,
                                    api_key: str = None, extra_body: dict = None):
    """Single ReAct + RAG, followed by one synthesis rewrite (no decomposition)."""
    from src.agent.base_agent import AgentResult
    from src.agent.synthesis_agent import SynthesisAgent

    if tools is None:
        tools = get_all_tools()
    # Reuse the RAG runner to obtain responses, then rewrite each one.
    responses, traces = _run_single_react_rag(
        queries, tools=tools, verbose=verbose,
        variant_name=variant_name, model=model, base_url=base_url,
        api_key=api_key, extra_body=extra_body,
    )
    synth = SynthesisAgent(model=model, base_url=base_url, api_key=api_key,
                           extra_body=extra_body)
    rewritten = []
    for q, resp in zip(queries, responses):
        try:
            rewritten.append(
                synth.synthesize(
                    q["query"],
                    [AgentResult(agent_name="single_react", response=resp)],
                    rag_context="",
                    validation_mode="off",
                )
            )
        except Exception as e:
            rewritten.append(f"[Error] {e}")
    return rewritten, traces


def _run_multi_agent_validation_off(queries: List[dict], tools=None,
                                    verbose: bool = False, variant_name: str = "",
                                    model: str = None, base_url: str = None,
                                    api_key: str = None, extra_body: dict = None):
    return _run_multi_agent(queries, tools, verbose=verbose,
                            variant_name=variant_name, model=model,
                            validation_mode="off",
                            base_url=base_url, api_key=api_key,
                            extra_body=extra_body)


def _run_multi_agent_validation_on(queries: List[dict], tools=None,
                                   verbose: bool = False, variant_name: str = "",
                                   model: str = None, base_url: str = None,
                                   api_key: str = None, extra_body: dict = None):
    return _run_multi_agent(queries, tools, verbose=verbose,
                            variant_name=variant_name, model=model,
                            validation_mode="always",
                            base_url=base_url, api_key=api_key,
                            extra_body=extra_body)


def _run_multi_agent_validation_concat(queries: List[dict], tools=None,
                                       verbose: bool = False, variant_name: str = "",
                                       model: str = None, base_url: str = None,
                                       api_key: str = None, extra_body: dict = None):
    """Deterministic validation + concatenation (no synthesis LLM call)."""
    return _run_multi_agent(queries, tools, verbose=verbose,
                            variant_name=variant_name, model=model,
                            validation_mode="always", synthesis_mode="concat",
                            base_url=base_url, api_key=api_key,
                            extra_body=extra_body)


def _run_multi_agent_shuffled_validation(queries: List[dict], tools=None,
                                         verbose: bool = False,
                                         variant_name: str = "", model: str = None,
                                         base_url: str = None, api_key: str = None,
                                         extra_body: dict = None):
    """Synthesis with the same warnings in random order (order control)."""
    return _run_multi_agent(queries, tools, verbose=verbose,
                            variant_name=variant_name, model=model,
                            validation_mode="shuffled",
                            base_url=base_url, api_key=api_key,
                            extra_body=extra_body)


def _run_multi_agent_query_only_synthesis(queries: List[dict], tools=None,
                                          verbose: bool = False,
                                          variant_name: str = "", model: str = None,
                                          base_url: str = None, api_key: str = None,
                                          extra_body: dict = None):
    """Extra-call control: one synthesis call with no specialist evidence."""
    from src.agent.synthesis_agent import SynthesisAgent

    del tools  # deliberately no tool access
    synth = SynthesisAgent(model=model, base_url=base_url, api_key=api_key,
                           extra_body=extra_body)

    def _do(q):
        return (synth.synthesize(q["query"], [], rag_context="",
                                 validation_mode="off"), [])

    return _run_concurrent(queries, _do, "  Query-only synthesis",
                           variant_name=variant_name, return_steps=True)


def _run_multi_agent_no_specialist(queries: List[dict], exclude: str,
                                    tools=None, verbose: bool = False,
                                    variant_name: str = "", model: str = None,
                                    synthesis_enabled: bool = True,
                                    base_url: str = None, api_key: str = None,
                                    extra_body: dict = None):
    """Multi-Agent with the given specialist removed (Dimension 2 ablation)

    Only queries related to that specialist are run (matched by category);
    unrelated queries directly reuse the cached results of Full (4 Specialists).
    """
    from src.agent.multi_agent_engine import MultiAgentEngine

    # exclude → category prefix mapping (includes the Chinese category mapping)
    _EXCLUDE_CAT_MAP = {
        "feed": "Feed Ingredient Bank",
        "nutrition": "Nutrient Requirements",
        "performance": "Performance Management",
        "definitions": "Definitions",
    }
    _CN_CAT_MAP = {
        "Chinese White-Feathered Broiler Performance": "Performance Management",
        "Chinese Yellow-Feathered Broiler Performance": "Performance Management",
        "Chinese Broiler Nutrient Requirements": "Nutrient Requirements",
        "Cross-Breed Comparison (Broiler)": "Performance Management",
        "Chinese Layer Performance": "Performance Management",
        "Chinese Layer Nutrient Requirements": "Nutrient Requirements",
    }
    target_prefix = _EXCLUDE_CAT_MAP.get(exclude, "")

    if tools is None:
        tools = get_all_tools()

    engine = MultiAgentEngine(tools=tools, verbose=verbose, rag_enabled=True,
                              model=model, synthesis_enabled=synthesis_enabled,
                              base_url=base_url, api_key=api_key,
                              extra_body=extra_body)
    if exclude in engine.specialists:
        del engine.specialists[exclude]

    def _do(q):
        result = engine.answer(q["query"])
        return (result["response"], result.get("steps", []))

    # Filter: only run queries matching the excluded specialist's domain
    msg = f"  -{exclude}"
    if target_prefix:
        relevant = []
        for i, q in enumerate(queries):
            cat = q.get("category", "")
            base_cat = _CN_CAT_MAP.get(cat, cat)
            if base_cat.startswith(target_prefix):
                relevant.append(i)

        if not relevant:
            return _run_concurrent(queries, _do, msg, variant_name=variant_name, return_steps=True)

        # Try to fill irrelevant slots from "Full (4 Specialists)" cache
        with _ABLATION_CACHE_LOCK:
            full_cache = _ABLATION_RESPONSE_CACHE.get("Full (4 Specialists)", {})

        if full_cache:
            responses: List[str] = [""] * len(queries)
            step_traces: List[list] = [[] for _ in queries]

            for i, q in enumerate(queries):
                if i not in relevant:
                    cached = full_cache.get(q["query"])
                    if cached:
                        if isinstance(cached, dict) and "resp" in cached:
                            responses[i] = cached["resp"]
                            step_traces[i] = cached.get("trace", [])
                        else:
                            responses[i] = cached
                        # Sync to this variant's cache so _save_responses_json picks it up
                        with _ABLATION_CACHE_LOCK:
                            _ABLATION_RESPONSE_CACHE.setdefault(variant_name, {})[q["query"]] = cached

            relevant_queries = [queries[i] for i in relevant]
            r_resp, r_traces = _run_concurrent(
                relevant_queries, _do, msg, variant_name=variant_name, return_steps=True)
            for idx, resp, traces in zip(relevant, r_resp, r_traces):
                responses[idx] = resp
                step_traces[idx] = traces

            # Print actual query count
            print(f"    ({len(relevant)}/{len(queries)} queries relevant to {exclude})")
            return responses, step_traces

    # Fallback: no Full cache — run all queries
    return _run_concurrent(queries, _do, msg, variant_name=variant_name, return_steps=True)


def _run_multi_agent_keyword_router(queries: List[dict], tools=None,
                                    verbose: bool = False,
                                    variant_name: str = "",
                                    model: str = None,
                                    synthesis_enabled: bool = True,
                                    base_url: str = None, api_key: str = None,
                                    extra_body: dict = None) -> List[str]:
    """Multi-Agent using keyword routing instead of LLM routing (Dimension 3 ablation)"""
    from src.agent.multi_agent_engine import MultiAgentEngine
    from src.agent.router_agent import RouterAgent

    if tools is None:
        tools = get_all_tools()

    engine = MultiAgentEngine(tools=tools, verbose=verbose, rag_enabled=True,
                              model=model, synthesis_enabled=synthesis_enabled,
                              base_url=base_url, api_key=api_key,
                              extra_body=extra_body)

    # Create a router that only uses keyword fallback
    class KeywordOnlyRouter(RouterAgent):
        def route(self, query):
            return self._keyword_fallback(query)

    engine.router = KeywordOnlyRouter()

    def _do(q):
        result = engine.answer(q["query"])
        return (result["response"], result.get("steps", []))

    return _run_concurrent(queries, _do, "  Keyword Router",
                           variant_name=variant_name, return_steps=True)


def _run_multi_agent_broadcast(queries: List[dict], tools=None,
                                verbose: bool = False,
                                variant_name: str = "",
                                model: str = None,
                                synthesis_enabled: bool = True,
                                base_url: str = None, api_key: str = None,
                                extra_body: dict = None):
    """Multi-Agent broadcast mode: no routing, invoke all 4 specialists (Dimension 3 ablation)"""
    from src.agent.multi_agent_engine import MultiAgentEngine
    from src.agent.base_agent import RouterDecision

    if tools is None:
        tools = get_all_tools()

    engine = MultiAgentEngine(tools=tools, verbose=verbose, rag_enabled=True,
                              model=model, synthesis_enabled=synthesis_enabled,
                              base_url=base_url, api_key=api_key,
                              extra_body=extra_body)

    class BroadcastRouter:
        def route(self, query):
            return RouterDecision(
                query=query, intent="multi",
                target_agents=["feed", "nutrition", "performance", "definitions"],
                requires_cross_validation=True,
            )

    engine.router = BroadcastRouter()

    def _do(q):
        result = engine.answer(q["query"])
        return (result["response"], result.get("steps", []))

    return _run_concurrent(queries, _do, "  Broadcast",
                           variant_name=variant_name, return_steps=True)


def _run_multi_agent_no_synthesis(queries: List[dict], tools=None,
                                   verbose: bool = False,
                                   variant_name: str = "",
                                   model: str = None,
                                   base_url: str = None, api_key: str = None,
                                   extra_body: dict = None):
    """Multi-Agent skipping the Synthesis layer and concatenating specialist outputs directly.

    Used in the synthesis-strategy ablation dimension to measure the marginal
    contribution of the SynthesisAgent's cross-validation + merging step.
    """
    return _run_multi_agent(queries, tools=tools, rag_enabled=True, verbose=verbose,
                            variant_name=variant_name, model=model,
                            synthesis_enabled=False,
                            base_url=base_url, api_key=api_key,
                            extra_body=extra_body)


# ── Architecture variant dispatch ────────────────────────────
# Explicit name -> runner mapping. The previous implementation used
# ``elif "no ReAct" in name.lower()``, which never matched (capital R in the
# literal), so the "no ReAct" variant silently re-ran the full pipeline.
# Never dispatch ablation variants by substring matching again.
_ARCHITECTURE_RUNNERS = {
    "Single ReAct": _run_single_react,
    "Multi-Agent (single-pass tool call)": _run_multi_agent_single_pass,
    "Multi-Agent (no tools)": _run_multi_agent_no_tools,
    "Multi-Agent (full)": _run_multi_agent,
}

# "no ReAct" name is deprecated; it now maps to the true single-pass variant.
_ARCHITECTURE_ALIASES = {
    "Multi-Agent (no ReAct)": "Multi-Agent (single-pass tool call)",
}


def _resolve_architecture_runner(name: str):
    """Return the runner for an architecture variant.

    Raises ValueError for unknown names so a typo cannot silently fall back to
    the full pipeline (the exact failure mode behind the dispatch issue).
    """
    canonical = _ARCHITECTURE_ALIASES.get(name, name)
    if canonical not in _ARCHITECTURE_RUNNERS:
        raise ValueError(
            f"Unknown architecture variant {name!r}. "
            f"Known variants: {sorted(_ARCHITECTURE_RUNNERS)}"
        )
    return _ARCHITECTURE_RUNNERS[canonical]


# ── Factorial dimensions (E1 architecture / E2 synthesis) ───
# Same explicit-mapping discipline as the architecture dimension: a typo must
# raise, never silently fall back to the full pipeline.
_ARCHITECTURE_FACTORIAL_RUNNERS = {
    "Standalone LLM": _run_standalone_llm,
    "Single ReAct + tools": _run_single_react,
    "Single ReAct + tools + RAG": _run_single_react_rag,
    "Single ReAct + tools + RAG + synthesis": _run_single_react_rag_synthesis,
    "Multi-Agent (validation off)": _run_multi_agent_validation_off,
    "Multi-Agent (validation on)": _run_multi_agent_validation_on,
}

_SYNTHESIS_FACTORIAL_RUNNERS = {
    "Concatenation": _run_multi_agent_no_synthesis,
    "Validation + concatenation": _run_multi_agent_validation_concat,
    "Synthesis (validation off)": _run_multi_agent_validation_off,
    "Synthesis (shuffled warnings)": _run_multi_agent_shuffled_validation,
    "Synthesis (correct warnings)": _run_multi_agent_validation_on,
    "Query-only synthesis": _run_multi_agent_query_only_synthesis,
}


def _resolve_factorial_runner(name: str, mapping: dict, dimension: str):
    if name not in mapping:
        raise ValueError(
            f"Unknown {dimension} variant {name!r}. "
            f"Known variants: {sorted(mapping)}"
        )
    return mapping[name]


# ── Evaluation ──────────────────────────────────────────────

def _run_scoring(queries: List[dict], all_responses: Dict[str, List[str]],
                 evaluator_llm=None, evaluator_llm_bulk=None, selfcheck=None,
                 n_selfcheck: int = 5,
                 n_selfcheck_values: Optional[List[int]] = None,
                 step_traces: Optional[Dict[str, List[List[dict]]]] = None,
                 enable_tolerance: bool = True, enable_flesch: bool = True,
                 inconsistency_ratio: float = 1.0) -> Dict[str, dict]:
    """Run EvaluatorLLM + tolerance + Flesch + SelfCheck + Faithfulness.

    Delegates to run_evaluation_pipeline() to avoid code duplication.
    SelfCheck is skipped by default in ablations (too slow); it is enabled only when selfcheck_fn is passed.
    step_traces is used for Faithfulness evaluation.
    evaluator_llm_bulk (score_bulk) is preferred — absolute scoring, batched.
    evaluator_llm (score_responses) is the fallback.
    """
    from src.evaluation.evaluator import run_evaluation_pipeline

    reports = run_evaluation_pipeline(
        queries=queries,
        nutrichat_responses=[],
        baseline_responses=all_responses,
        evaluator_llm=evaluator_llm,
        evaluator_llm_bulk=evaluator_llm_bulk,
        selfcheck=selfcheck,
        n_selfcheck_samples=n_selfcheck,
        n_selfcheck_values=n_selfcheck_values if n_selfcheck_values else None,
        step_traces=step_traces,
        enable_tolerance=enable_tolerance,
        enable_flesch=enable_flesch,
        inconsistency_ratio=inconsistency_ratio,
    )
    # Remove empty NutriCHAT entry added by the pipeline
    reports.pop("NutriCHAT", None)
    return {name: r.summary() for name, r in reports.items()}


def _try_get_scorers():
    """Create EvaluatorLLM and SelfCheckGPT (graceful on failure).

    Returns (evaluator_llm, evaluator_llm_bulk, selfcheck_fn).
    evaluator_llm_bulk (score_bulk) is the preferred path — absolute scoring,
    batched by query. evaluator_llm (score_responses) is the fallback.
    """
    evaluator_fn = None
    evaluator_bulk_fn = None
    selfcheck_fn = None
    try:
        from src.evaluation.evaluator_llm import EvaluatorLLM
        ev = EvaluatorLLM()
        evaluator_fn = ev.score_responses
        evaluator_bulk_fn = ev.score_bulk
    except Exception as e:
        print(f"  [WARN] EvaluatorLLM not available: {e}")

    try:
        from src.evaluation.selfcheck import SelfCheckGPT
        sc = SelfCheckGPT(device=Config.NLI_DEVICE,
                             sample_model=Config.SELFCHECK_MODEL,
                             sample_base_url=Config.SELFCHECK_BASE_URL,
                             sample_api_key=Config.SELFCHECK_API_KEY,
                             sample_extra_body=Config.SELFCHECK_EXTRA_BODY)
        selfcheck_fn = sc.compute_score
    except Exception as e:
        print(f"  [WARN] SelfCheckGPT not available: {e}")

    return evaluator_fn, evaluator_bulk_fn, selfcheck_fn


# ── Table output ────────────────────────────────────────────

def _val(s: dict, key: str):
    """Extract (mean, std, n) from a report summary dict."""
    entry = s.get(key, {})
    mean = entry.get("mean", None)
    std = entry.get("std", None)
    n = entry.get("n", 0)
    return (mean, std, n)


def _fmt(mean, std=None, n=0, decimals: int = 2):
    if mean is None:
        return "N/A"
    s = f"{mean:.{decimals}f}"
    if std is not None:
        s += f"±{std:.{decimals}f}"
    if n > 0:
        s += f" (n={n})"
    return s


def _fmt_pct(mean, std=None, n=0):
    if mean is None:
        return "N/A"
    s = f"{mean*100:.1f}%"
    if std is not None:
        s += f"±{std*100:.1f}%"
    if n > 0:
        s += f" (n={n})"
    return s


def _print_table(title: str, results: Dict[str, dict], metrics: List[str],
                  headers: List[str], note: str = ""):
    """Generic formatted table printer.
    Auto-detects inconsistency_n5/n10/n20 keys and expands columns.
    """
    # Check if multi-N inconsistency keys exist
    sample = next(iter(results.values()), {})
    multi_n_keys = sorted(
        [k for k in sample if k.startswith("inconsistency_n")],
        key=lambda x: int(x.replace("inconsistency_n", "")),
    )
    has_multi_n = len(multi_n_keys) >= 1

    # Build display metrics & headers
    display_metrics = []
    display_headers = []
    for m, h in zip(metrics, headers):
        if m == "inconsistency" and has_multi_n:
            for nk in multi_n_keys:
                display_metrics.append(nk)
                n = nk.replace("inconsistency_n", "")
                display_headers.append(f"Incon(N={n})")
        else:
            display_metrics.append(m)
            display_headers.append(h)

    width = max(32, len(max(results.keys(), key=len)) + 2) if results else 32
    col_w = 20

    print(f"\n{'=' * (width + col_w * len(display_metrics) + 4)}")
    print(f"  {title}")
    print(f"{'=' * (width + col_w * len(display_metrics) + 4)}")

    hdr = f"  {'Variant':<{width}}"
    for h in display_headers:
        hdr += f" {h:<{col_w}}"
    print(hdr)
    sep = f"  {'-'*width}"
    for _ in display_headers:
        sep += f" {'-'*col_w}"
    print(sep)

    for name, s in results.items():
        row = f"  {name:<{width}}"
        for m in display_metrics:
            mean, std, n = _val(s, m)
            if m.startswith("tolerance_p"):
                row += f" {_fmt_pct(mean, std, n):<{col_w}}"
            elif m.startswith("inconsistency"):
                row += f" {_fmt(mean, std, n, 4):<{col_w}}"
            else:
                row += f" {_fmt(mean, std, n, 2):<{col_w}}"
        print(row)

    if note:
        print(f"  {'-' * (width + col_w * len(display_metrics) + 4)}")
        print(f"  {note}")


# ── YAML configuration loading ──────────────────────────────

def _load_ablation_config() -> dict:
    """Load ablation.yaml, merge config.local.yaml scoring + variant overrides."""
    from src.schemas import validate_config
    from src.config import _LOCAL_OVERRIDE, deep_merge
    import yaml
    path = os.path.join(_PROJECT_ROOT, "config", "ablation.yaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Merge local overrides from config.local.yaml
    if _LOCAL_OVERRIDE:
        # 1. Scoring: scoring_ablation > scoring (experiment-specific > global override)
        base = _LOCAL_OVERRIDE.get("scoring_ablation") or _LOCAL_OVERRIDE.get("scoring")
        if base:
            data["scoring"] = deep_merge(data.get("scoring", {}), base)

        # 2. Ablation dimension variants: name-based merge + append
        # Two formats are supported:
        #   dict format (legacy): ablation.llm_backbone.variants: [...]
        #   list format (flat):   ablation.llm_backbone: [...]  ← recommended
        local_abl = _LOCAL_OVERRIDE.get("ablation")
        if local_abl and isinstance(local_abl, dict):
            for dim_key, dim_cfg in local_abl.items():
                if dim_key == "dimensions":
                    continue  # dimension-filter parameter, not an ablation dimension
                if dim_key not in data:
                    raise ValueError(
                        f"Unknown ablation dimension '{dim_key}' in config.local.yaml. "
                        f"Available: {list(data.keys())}"
                    )
                if isinstance(dim_cfg, list):
                    # Flat format: fully replace the variants list (config.local.yaml has full control)
                    local_variants = [
                        {"name": v, "model_ref": v} if isinstance(v, str) else v
                        for v in dim_cfg
                    ]
                    data[dim_key]["variants"] = local_variants
                elif isinstance(dim_cfg, dict):
                    # Legacy format: dim_cfg is {variants: [...], ...}
                    # Merge non-variant fields (enabled, meta)
                    for k, v in dim_cfg.items():
                        if k == "variants":
                            continue
                        if isinstance(v, dict) and isinstance(data[dim_key].get(k), dict):
                            data[dim_key][k] = deep_merge(data[dim_key][k], v)
                        else:
                            data[dim_key][k] = v
                    local_variants = dim_cfg.get("variants", [])
                    # Normalize strings → {name, model_ref}
                    local_variants = [
                        {"name": v, "model_ref": v} if isinstance(v, str) else v
                        for v in local_variants
                    ]
                    # Merge variants by name (override existing, append new)
                    if local_variants and "variants" in data[dim_key]:
                        name_map = {v["name"]: v for v in data[dim_key]["variants"]}
                        for lv in local_variants:
                            if lv["name"] in name_map:
                                name_map[lv["name"]].update(lv)
                            else:
                                data[dim_key]["variants"].append(lv)
                else:
                    continue

    data = validate_config("ablation", data)

    # 3. Resolve model_ref → {model, base_url, api_key, ...} from models.yaml baselines
    from src.config import Config as _Config
    for dim_key, dim_cfg in data.items():
        if not isinstance(dim_cfg, dict):
            continue
        variants = dim_cfg.get("variants", [])
        for v in variants:
            if not isinstance(v, dict):
                continue
            ref = v.get("model_ref")
            if ref:
                resolved = _Config.resolve_model_ref(ref)
                if resolved:
                    v.update(resolved)
                    v.pop("model_ref", None)
                else:
                    _avail = sorted(
                        m.get("name", "") for m in _Config.BASELINE_MODELS
                    )
                    raise ValueError(
                        f"model_ref '{ref}' not found in models.yaml baselines.\n"
                        f"  Available names: {_avail}"
                    )

    return data


# ── Dimension definitions (global constants; control execution and report order) ──
_DIM_KEYS = [
    "architecture",
    "synthesis",
    "specialist",
    "router",
    "rag",
    "llm_backbone",
    "language",
    "architecture_factorial",
    "synthesis_factorial",
]


# ── Per-dimension execution (variant configs read from YAML) ─

def _run_variant_single(v, dim_key, queries, tools, verbose):
    """Run a single ablation variant. Returns (name, responses, step_traces, elapsed_s)."""
    name = v["name"]
    model = v.get("model")
    base_url = v.get("base_url")
    api_key = v.get("api_key")
    extra_body = v.get("extra_body")
    _llm_kw = dict(model=model, base_url=base_url, api_key=api_key, extra_body=extra_body)
    t0 = time.time()
    try:
        if dim_key == "architecture":
            runner = _resolve_architecture_runner(name)
            resp, traces = runner(queries, tools, verbose=verbose,
                                  variant_name=name, **_llm_kw)

        elif dim_key == "architecture_factorial":
            runner = _resolve_factorial_runner(
                name, _ARCHITECTURE_FACTORIAL_RUNNERS, dim_key
            )
            resp, traces = runner(queries, tools, verbose=verbose,
                                  variant_name=name, **_llm_kw)

        elif dim_key == "synthesis_factorial":
            runner = _resolve_factorial_runner(
                name, _SYNTHESIS_FACTORIAL_RUNNERS, dim_key
            )
            resp, traces = runner(queries, tools, verbose=verbose,
                                  variant_name=name, **_llm_kw)

        elif dim_key == "specialist":
            exclude = v.get("exclude")
            if exclude:
                resp, traces = _run_multi_agent_no_specialist(queries, exclude=exclude, tools=tools, verbose=verbose, variant_name=name, **_llm_kw)
            else:
                resp, traces = _run_multi_agent(queries, tools, verbose=verbose, variant_name=name, **_llm_kw)

        elif dim_key == "router":
            strategy = v.get("strategy", "llm")
            if strategy == "keyword":
                resp, traces = _run_multi_agent_keyword_router(queries, tools, verbose, variant_name=name, **_llm_kw)
            elif strategy == "broadcast":
                resp, traces = _run_multi_agent_broadcast(queries, tools, verbose, variant_name=name, **_llm_kw)
            else:
                resp, traces = _run_multi_agent(queries, tools, verbose=verbose, variant_name=name, **_llm_kw)

        elif dim_key == "rag":
            rag_enabled = v.get("rag_enabled", True)
            if rag_enabled:
                resp, traces = _run_multi_agent(queries, tools, verbose=verbose, variant_name=name, **_llm_kw)
            else:
                resp, traces = _run_multi_agent_no_rag(queries, tools, verbose, variant_name=name, **_llm_kw)

        elif dim_key == "synthesis":
            synthesis_enabled = v.get("synthesis_enabled", True)
            if synthesis_enabled:
                kw = dict(**_llm_kw, synthesis_enabled=True)
                resp, traces = _run_multi_agent(queries, tools, verbose=verbose, variant_name=name, **kw)
            else:
                # _run_multi_agent_no_synthesis does not accept synthesis_enabled, so do not pass **kw (extra keys raise TypeError)
                resp, traces = _run_multi_agent_no_synthesis(queries, tools, verbose, variant_name=name, **_llm_kw)

        elif dim_key == "llm_backbone":
            resp, traces = _run_multi_agent(queries, tools, verbose=verbose, variant_name=name, **_llm_kw)

        elif dim_key == "language":
            resp, traces = _run_multi_agent(queries, tools, verbose=verbose, variant_name=name, **_llm_kw)

        else:
            resp, traces = _run_multi_agent(queries, tools, verbose=verbose, variant_name=name, **_llm_kw)

        elapsed = time.time() - t0
        _save_disk_cache()
        return name, resp, traces, round(elapsed, 1), None
    except Exception as e:
        import traceback
        elapsed = time.time() - t0
        logger = get_logger("ablation")
        logger.error("Variant '%s' failed: %s", name, e)
        traceback.print_exc()
        error_msg = f"[Error] {e}"
        return name, [error_msg] + [""] * (len(queries) - 1), [], round(elapsed, 1), str(e)


def _resolve_variant_queries(v: dict, base_queries: List[dict]) -> List[dict]:
    """Filter base queries per variant's query_set override (for language dimension).

    Uses language field when available, falls back to id prefix.
    """
    qs = v.get("query_set")
    if not qs or qs == "all":
        return base_queries
    if qs == "chinese":
        return [q for q in base_queries if q.get("language") == "zh" or q.get("id", "").startswith("CN-")]
    elif qs == "western":
        return [q for q in base_queries if q.get("language") == "en" and not q.get("id", "").startswith("CN-")]
    return base_queries


def _run_dimension_variants(dim_key: str, variants: list, queries, evaluator_fn, evaluator_bulk_fn, selfcheck_fn, verbose, meta: dict = None,
                            scoring_cfg: Optional[dict] = None):
    """Run all variants for a given ablation dimension, driven by config.

    Each variant runs independently in parallel. As each completes, scoring
    results are printed immediately — no waiting for slow variants.

    Each variant dict has at minimum {'name': str}, with dimension-specific fields:
      - architecture: name only (maps to _run_single_react / _run_multi_agent / _run_multi_agent_no_react)
      - specialist:   {'exclude': str} → _run_multi_agent_no_specialist
      - router:       {'strategy': str} → llm/keyword/broadcast
      - rag:          {'rag_enabled': bool} → _run_multi_agent
    """
    meta = meta or {}
    scoring_cfg = scoring_cfg or {}
    enable_selfcheck = scoring_cfg.get("selfcheck", False)
    # Read n_selfcheck_values from the merged ablation config, overriding the settings.yaml default
    _sc_n_selfcheck = scoring_cfg.get("n_selfcheck_values") or Config.EVAL_N_SELFCHECK_VALUES

    print(f"\n{'─' * 60}")
    print(f"  Dimension: {meta.get('title', dim_key)}")
    if meta.get('subtitle'):
        print(f"  {meta['subtitle']}")
    print(f"{'─' * 60}")

    tools = get_all_tools()
    all_responses: Dict[str, List[str]] = {}
    all_step_traces: Dict[str, List[List[dict]]] = {}
    variant_times: Dict[str, float] = {}
    results: Dict[str, dict] = {}

    n_workers = min(len(variants), 3)  # variant parallelism; avoids 429 rate limits from nested thread pools

    def _score_variant(name: str, resp: List[str], traces: List[list], v_queries=None):
        """Score a single variant with optional per-variant queries (language dimension)."""
        q = v_queries if v_queries is not None else queries
        r = _run_scoring(q, {name: resp}, evaluator_llm=evaluator_fn,
                         evaluator_llm_bulk=evaluator_bulk_fn,
                         selfcheck=None,
                         n_selfcheck_values=_sc_n_selfcheck,
                         step_traces={name: traces} if traces else None)
        return r

    # Specialist dimension: run Full serially first (cache written to disk), then run the other variants reusing the cache
    if dim_key == "specialist":
        full_variants = [v for v in variants if v.get("exclude") is None]
        other_variants = [v for v in variants if v.get("exclude") is not None]
        # Run Full first
        for v in full_variants:
            name, resp, traces, elapsed, err = _run_variant_single(v, dim_key, queries, tools, verbose)
            all_responses[name] = resp
            all_step_traces[name] = traces
            variant_times[name] = elapsed
            _save_disk_cache()
            status = f"FAILED ({err})" if err else f"{elapsed:.0f}s, {len(resp)} responses"
            tqdm.write(f"\n  >>> [{name}] {status}")
            if not err:
                results.update(_score_variant(name, resp, traces))
                c_mean, c_std, c_n = _val(results[name], "correctness")
                tqdm.write(f"    {name}: correctness={_fmt(c_mean, c_std, c_n)}")
        variants = other_variants  # only run remaining in parallel

    print(f"  Running {len(variants)} variants in parallel...")
    with ThreadPoolExecutor(max_workers=min(len(variants), _ABLATION_MAX_WORKERS)) as pool:
        def _submit_with_queries(v):
            v_queries = _resolve_variant_queries(v, queries)
            result = _run_variant_single(v, dim_key, v_queries, tools, verbose)
            return result, v_queries, v["name"]

        futures = {pool.submit(_submit_with_queries, v): v["name"]
                   for v in variants}

        for f in as_completed(futures):
            (name, resp, traces, elapsed, err), v_queries, _ = f.result()
            all_responses[name] = resp
            all_step_traces[name] = traces
            variant_times[name] = elapsed

            status = f"FAILED ({err})" if err else f"{elapsed:.0f}s, {len(resp)} responses"
            tqdm.write(f"\n  >>> [{name}] {status}")

            # Score this variant immediately
            if not err:
                variant_results = _score_variant(name, resp, traces, v_queries)
                results.update(variant_results)
            else:
                results[name] = {m: {"mean": 0, "std": 0, "n": 0}
                                 for m in ["correctness", "precision", "scientific_depth",
                                           "tolerance_p2", "tolerance_p5", "flesch_reading_ease",
                                            "inconsistency", "faithfulness", "faithfulness_semantic"]}

            # Print interim result for this variant
            if results and not err:
                c_mean, c_std, c_n = _val(results[name], "correctness")
                p_mean, p_std, p_n = _val(results[name], "precision")
                d_mean, d_std, d_n = _val(results[name], "scientific_depth")
                tqdm.write(f"    {name}: correctness={_fmt(c_mean, c_std, c_n)}, "
                          f"precision={_fmt(p_mean, p_std, p_n)}, "
                          f"depth={_fmt(d_mean, d_std, d_n)}")

    # Validation
    for name, resp in all_responses.items():
        expected = len(_resolve_variant_queries({"query_set": ""}, queries))
        if len(resp) != expected:
            # Try to determine actual expected count per variant
            for v in variants:
                if v["name"] == name:
                    expected = len(_resolve_variant_queries(v, queries))
                    break
            if len(resp) != expected:
                print(f"  [WARN] {name}: {len(resp)} responses, expected {expected}")

    # ── Final comparison table (SelfCheck runs separately below) ──
    _print_table(
        f"Dimension: {meta.get('title', dim_key)}",
        results,
        metrics=["correctness", "precision", "scientific_depth", "tolerance_p2", "tolerance_p5", "flesch_reading_ease", "faithfulness", "faithfulness_semantic"],
        headers=["Correctness", "Precision", "Depth", "Tol±2%", "Tol±5%", "FRES", "Faith(Num)", "Faith(Sem)"],
        note=meta.get("note", ""),
    )

    # ── SelfCheck (independent, runs after main table) ──
    if enable_selfcheck and selfcheck_fn:
        # Collect per-variant skip_selfcheck flags (set via config.local.yaml)
        skip_sc = {v["name"] for v in variants if v.get("skip_selfcheck")}
        sc_variants = [v for v in variants if v["name"] not in skip_sc]
        sc_ratio = float(scoring_cfg.get("hallucination_ratio", 1.0))

        if not sc_variants:
            print(f"\n  [Skip] SelfCheck skipped: all variants have skip_selfcheck=true")
        else:
            has_query_set = any(v.get("query_set") for v in sc_variants)
            print(f"\n  SelfCheckGPT (inconsistency_ratio={sc_ratio}) ...")

            if has_query_set:
                # Per-variant SelfCheck for dimensions with per-variant query_sets
                for v in sc_variants:
                    name = v["name"]
                    if name not in all_responses:
                        continue
                    v_queries = _resolve_variant_queries(v, queries)
                    sc_r = _run_scoring(v_queries, {name: all_responses[name]}, None,
                                        selfcheck=selfcheck_fn,
                                        n_selfcheck_values=_sc_n_selfcheck,
                                        enable_tolerance=False, enable_flesch=False,
                                        inconsistency_ratio=sc_ratio)
                    for _, sr in sc_r.items():
                        for k, val in sr.items():
                            if k.startswith("inconsistency"):
                                results.setdefault(name, {})[k] = val
            else:
                # Standard batch SelfCheck over all eligible variants
                sc_filtered = {n: r for n, r in all_responses.items() if n not in skip_sc}
                sc_results = _run_scoring(queries, sc_filtered, None,
                                          selfcheck=selfcheck_fn,
                                          n_selfcheck_values=_sc_n_selfcheck,
                                          enable_tolerance=False, enable_flesch=False,
                                        inconsistency_ratio=sc_ratio)
                for name, sr in sc_results.items():
                    if name in results:
                        results[name].update({k: v for k, v in sr.items()
                                              if k.startswith("inconsistency")})

            has_halluc = any(
                k.startswith("inconsistency") and v.get("mean") is not None
                for r in results.values() for k, v in r.items()
            )
            if has_halluc:
                _print_table(
                    f"Inconsistency (SelfCheckGPT)",
                    results,
                    metrics=["inconsistency"],
                    headers=["Incon."],
                    note="",
                )
            else:
                print("  [WARN] SelfCheckGPT returned no valid inconsistency scores "
                      "(check NLI model / API availability)")
    return results, variant_times


# ── Saving results ──────────────────────────────────────────

def _save_report(all_results: Dict[str, dict], log_dir: str, dim_meta: Dict[str, dict] = None,
                 variant_times: Optional[Dict[str, Dict[str, float]]] = None,
                 dim_suffix: str = ""):
    """Save all dimension results to JSON and Markdown.

    Automatically merges existing results when reusing a directory, avoiding overwrites.
    """
    dim_meta = dim_meta or {}
    _suffix = f"_{dim_suffix}" if dim_suffix else ""
    # JSON — merge existing results (keep earlier dimension data when reusing a directory)
    json_path = os.path.join(log_dir, f"ablation_report{_suffix}.json")
    merged = dict(all_results)
    if os.path.exists(json_path):
        try:
            import json as _json
            with open(json_path, "r", encoding="utf-8") as f:
                existing = _json.load(f)
            # Merge existing dimension + timing data
            for k, v in existing.items():
                if k not in merged:
                    merged[k] = v
            # Merge existing timing data
            if "_timing" in existing and variant_times:
                existing_timing = existing["_timing"]
                for dk, dv in variant_times.items():
                    existing_timing[dk] = dv
                merged["_timing"] = existing_timing
            elif "_timing" in existing:
                merged["_timing"] = existing["_timing"]
        except Exception:
            pass
    if variant_times:
        merged["_timing"] = variant_times
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    # Timing lookup
    timing = merged.get("_timing", {})

    # Markdown
    md_path = os.path.join(log_dir, f"ablation_report{_suffix}.md")
    base_metrics = [
        ("correctness", "Correctness"),
        ("precision", "Precision"),
        ("scientific_depth", "Depth"),
        ("tolerance_p2", "Tol±2%"),
        ("tolerance_p5", "Tol±5%"),
        ("flesch_reading_ease", "FRES"),
        ("faithfulness", "Faith(Num)"),
        ("faithfulness_semantic", "Faith(Sem)"),
    ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# NutriCHAT Ablation Study Report\n\n")
        f.write("Focus: Multi-Agent Architecture (Router → Specialists → Synthesis)\n\n")
        f.write("SelfCheckGPT-NLI scores reported at N=5, 10, 20 to validate stability.\n\n")
        f.write("Faithfulness: fraction of response numeric claims supported by tool-retrieved data.\n\n")

        for dim_key in _DIM_KEYS:
            if dim_key not in merged:
                continue
            label = dim_meta.get(dim_key, {}).get("title", dim_key)
            note = dim_meta.get(dim_key, {}).get("subtitle", "")
            f.write(f"---\n\n## {label}\n\n")
            f.write(f"{note}\n\n")

            results = merged[dim_key]
            sample = next(iter(results.values()), {})

            # Build metrics with multi-N hallucination if available
            multi_n_keys = sorted(
                [k for k in sample if k.startswith("inconsistency_n")],
                key=lambda x: int(x.replace("inconsistency_n", "")),
            )
            has_multi = len(multi_n_keys) >= 1

            display_metrics = list(base_metrics)
            if has_multi:
                for nk in multi_n_keys:
                    n = nk.replace("inconsistency_n", "")
                    display_metrics.append((nk, f"Halluc(N={n})"))
            else:
                display_metrics.append(("hallucination", "Incon."))

            # Table header
            header = f"| {'Variant':<30} |"
            sep = f"| {'-'*30} |"
            for _, label in display_metrics:
                header += f" {label:<20} |"
                sep += f" {'-'*20} |"
            header += " Time |"
            sep += " ---- |"
            f.write(header + "\n")
            f.write(sep + "\n")

            # Table rows
            dim_timing = timing.get(dim_key, {})
            for name, s in results.items():
                row = f"| {name:<30} |"
                for m_key, _ in display_metrics:
                    mean, std, n = _val(s, m_key)
                    if mean is None:
                        cell = "N/A"
                    elif m_key.startswith("hallucination"):
                        cell = f"{mean:.4f}±{std:.4f}" if std is not None else f"{mean:.4f}"
                    elif m_key.startswith("tolerance_p"):
                        cell = f"{mean*100:.1f}%±{std*100:.1f}%" if std is not None else f"{mean*100:.1f}%"
                    else:
                        cell = f"{mean:.2f}±{std:.2f}" if std is not None else f"{mean:.2f}"
                    if n > 0:
                        cell += f" (n={n})"
                    row += f" {cell:<22} |"
                # Timing column
                t = dim_timing.get(name, None)
                if t is not None:
                    row += f" {t:>5.0f}s |"
                else:
                    row += f" {'N/A':>5} |"
                f.write(row + "\n")
            f.write("\n")

        # Footer with total time
        total_sec = sum(
            t for dim_t in timing.values() for t in dim_t.values()
        ) if timing else 0
        if total_sec >= 60:
            footer_time = f"{total_sec / 60:.0f} min {total_sec % 60:.0f} s"
        else:
            footer_time = f"{total_sec:.0f} s"
        f.write("---\n")
        f.write(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | Total time: {footer_time}*\n")

    print(f"\n[Ablation] Report saved to: {log_dir}/")
    print(f"  ablation_report{_suffix}.json")
    print(f"  ablation_report{_suffix}.md")


def _save_responses_json(log_dir: str, queries: list):
    """Save responses.json in eval/compare format: {responses: {variant: [resp, ...]}, queries: [...]}."""
    import json
    cache_path = os.path.join(log_dir, ".ablation_cache.json")
    if not os.path.exists(cache_path):
        return
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

        # Build ordered response lists matching queries order
        responses_by_var = {}
        for var, q_map in cache.items():
            resp_list = []
            for q in queries:
                qt = q["query"]
                val = q_map.get(qt, "")
                if isinstance(val, dict) and "resp" in val:
                    resp_list.append(val["resp"])
                else:
                    resp_list.append(val if isinstance(val, str) else "")
            responses_by_var[var] = resp_list

        doc = {
            "generated": __import__("datetime").datetime.now().isoformat(),
            "num_queries": len(queries),
            "models": list(responses_by_var.keys()),
            "queries": [
                {"id": q.get("id", f"Q{i+1:03d}"), "category": q.get("category", ""),
                 "type": q.get("type", ""), "query": q["query"],
                 "expected_value": q.get("expected_value")}
                for i, q in enumerate(queries)
            ],
            "responses": responses_by_var,
        }

        out_path = os.path.join(log_dir, "responses.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ── Config saving (consistent with --evaluate/--compare) ─────


def _strip_api_keys(obj):
    """Recursively strip api_key / api_key_env fields from a dict."""
    if isinstance(obj, dict):
        return {k: _strip_api_keys(v) for k, v in obj.items()
                if k not in ("api_key", "api_key_env")}
    elif isinstance(obj, list):
        return [_strip_api_keys(i) for i in obj]
    return obj


def _save_ablation_config(log_dir: str, query_set: str, sample_ratio: Optional[float],
                          enabled_dims: dict, merged_cfg: dict, ablation_dim: Optional[str] = None,
                          n_selfcheck_values: Optional[list] = None):
    """Save config.yaml matching evaluate/compare format + ablation merged view.

    Format is compatible with --from-config loader (main.py).
    """
    import yaml
    from src.config import Config, _NUTRI_CHAT_MODEL, _EVALUATOR_MODEL, _SELFCHECK_CONFIG
    try:
        summary = {
            # Shared fields (same as --evaluate/--compare)
            "species": getattr(Config, "SPECIES", "broiler"),
            "agent_mode": "multi",
            "query_set": query_set,
            "eval_sample_ratio": sample_ratio,
            "eval_n_selfcheck_values": n_selfcheck_values or getattr(Config, "EVAL_N_SELFCHECK_VALUES", None),
            "llm_model": getattr(Config, "LLM_MODEL", None),
            "evaluator_model": getattr(Config, "EVALUATOR_MODEL", None),
            "embedding_provider": getattr(Config, "EMBEDDING_PROVIDER", None),
            "nli_device": getattr(Config, "NLI_DEVICE", None),
            # Ablation-specific fields
            "ablation_dim": ablation_dim,
            "enabled_dimensions": list(enabled_dims.keys()),
            "num_variants": sum(len(v.get("variants", [])) for v in enabled_dims.values()),
        }
        doc = {"config": dict(sorted(summary.items()))}
        if _NUTRI_CHAT_MODEL:
            doc["nutrichat"] = {k: v for k, v in _NUTRI_CHAT_MODEL.items()
                                if k not in ("api_key", "api_key_env") and v is not None}
        if _EVALUATOR_MODEL:
            doc["evaluator"] = {k: v for k, v in _EVALUATOR_MODEL.items()
                                if k not in ("api_key", "api_key_env") and v is not None}
        if _SELFCHECK_CONFIG:
            doc["selfcheck"] = {k: v for k, v in _SELFCHECK_CONFIG.items()
                                if k not in ("api_key", "api_key_env") and v is not None}
        doc["scoring"] = merged_cfg.get("scoring", {})
        doc["ablation_merged"] = _strip_api_keys(merged_cfg)
        with open(os.path.join(log_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(doc, f, default_flow_style=False, allow_unicode=True)
        print(f"  [Config] config.yaml saved")
    except Exception:
        pass


# ── Main entry point ────────────────────────────────────────

def run_ablation(
    dim: Optional[str] = None,
    verbose: bool = False,
    query_set: str = "all",
    sample_ratio: Optional[float] = None,
):
    """Run ablation study, driven by config/ablation.yaml.

    Args:
        dim: None (all enabled dimensions), or one of "architecture", "specialist", "router",
             "rag", "synthesis", "llm_backbone", "language"
        sample_ratio: Optional stratified sampling ratio (0.0-1.0).
    """
    print("\n" + "=" * 60)
    print("  NutriCHAT Ablation Study")
    print("  Focus: Multi-Agent Architecture")
    print("  (Router → Specialists → Synthesis)")
    print("=" * 60)

    # Load config first (needed for scoring_ablation.sample_ratio)
    cfg = _load_ablation_config()

    # Use scoring_ablation's sample_ratio if set, else fall back to parameter
    _abl_scoring = cfg.get("scoring", {})
    _sr = _abl_scoring.get("sample_ratio")
    if _sr is not None:
        sample_ratio = _sr

    # Load queries
    from src.utils.query_loader import load_queries
    queries = load_queries(query_set=query_set, sample_ratio=sample_ratio)
    print(f"[Ablation] Loaded {len(queries)} queries")
    dims_config = {k: cfg.get(k, {}) for k in _DIM_KEYS}
    enabled_dims = {k: v for k, v in dims_config.items() if v.get("enabled", True)}

    # Filter dimensions by ablation.dimensions in config.local.yaml (CLI --ablation-dim takes priority)
    if not dim:  # only use config filtering when the CLI did not specify a dimension
        _cfg_dims = cfg.get("dimensions")
        if _cfg_dims and isinstance(_cfg_dims, list):
            enabled_dims = {k: v for k, v in enabled_dims.items() if k in _cfg_dims}
            print(f"[Ablation] Filtered by config: {list(enabled_dims.keys())}")

    if not enabled_dims:
        print("[Ablation] No enabled dimensions found in config/ablation.yaml")
        return

    # Create log dir (uses the _create_log_dir format uniformly, including config.yaml + config_snapshot)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dim_suffix = f"_{dim}" if dim else ""
    sp = getattr(Config, "SPECIES", None)
    sp_suffix = f"_{sp}" if sp and sp != "broiler" else ""
    log_dir = os.path.join(_PROJECT_ROOT, "logs", f"ablation{dim_suffix}{sp_suffix}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Save compact config.yaml (same format as --evaluate/--compare)
    _n_sc_vals = (cfg.get("scoring") or {}).get("n_selfcheck_values")
    _save_ablation_config(log_dir, query_set, sample_ratio, enabled_dims, cfg,
                          ablation_dim=dim, n_selfcheck_values=_n_sc_vals)

    # Initialize the disk cache (persisted across runs)
    # Prefer loading the cache from the most recent ablation log to support incremental skipping
    global _ABLATION_CACHE_FILE
    _ABLATION_CACHE_FILE = os.path.join(log_dir, ".ablation_cache.json")
    if not os.path.exists(_ABLATION_CACHE_FILE):
        _logs_dir = os.path.join(_PROJECT_ROOT, "logs")
        _prev_caches = sorted(
            [d for d in os.listdir(_logs_dir) if d.startswith("ablation_") and d != os.path.basename(log_dir)],
            key=lambda d: os.path.getmtime(os.path.join(_logs_dir, d)),
            reverse=True,
        )
        for _prev in _prev_caches:
            _prev_cache = os.path.join(_logs_dir, _prev, ".ablation_cache.json")
            if os.path.exists(_prev_cache):
                import shutil
                shutil.copy2(_prev_cache, _ABLATION_CACHE_FILE)
                print(f"  [Cache] Loaded from previous run: {_prev}")
                break
    _load_disk_cache()

    # Extract metadata from config
    dim_meta = {}
    for dim_key in _DIM_KEYS:
        dim_cfg = enabled_dims.get(dim_key, {})
        dim_meta[dim_key] = dim_cfg.get("meta", {})

    # Per-run ablation cache only (don't wipe project-wide .cache/)

    # Extract scoring config (determines whether SelfCheck etc. run)
    ablation_scoring_cfg = cfg.get("scoring", {})
    if ablation_scoring_cfg.get("selfcheck", False):
        print(f"  [Config] SelfCheckGPT enabled (may 3-5x scoring time)")
    else:
        print(f"  [Config] SelfCheckGPT disabled (enable in config/ablation.yaml → scoring.selfcheck)")

    # Initialize scorers
    evaluator_fn, evaluator_bulk_fn, selfcheck_fn = _try_get_scorers()

    # Warm-up: trigger LLM and embedding loading so cold starts do not slow down the first query
    _warmup_llm()

    # Determine which dimensions to run
    dim_keys = [dim] if dim else list(enabled_dims.keys())

    all_results: Dict[str, dict] = {}
    all_variant_times: Dict[str, Dict[str, float]] = {}

    # Define an intermediate save function that writes to disk after each completed dimension
    def _partial_save():
        """Save whatever results we have so far into the merged report."""
        if all_results:
            _save_report(all_results, log_dir, dim_meta=dim_meta,
                         variant_times=all_variant_times, dim_suffix="")
            print(f"  [Save] Intermediate results saved ({len(all_results)} dimensions)")

    def _is_dimension_cached(variants, queries_list):
        """Check whether all variants of a dimension are already completed in the cache."""
        with _ABLATION_CACHE_LOCK:
            for v in variants:
                vname = v["name"]
                v_cache = _ABLATION_RESPONSE_CACHE.get(vname, {})
                if sum(1 for q in queries_list if q["query"] not in v_cache) > 0:
                    return False
            return True

    try:
        for dim_key in dim_keys:
            if dim_key not in enabled_dims:
                print(f"[Ablation] Dimension '{dim_key}' not found or disabled in config, skipping")
                continue
            dim_cfg = enabled_dims[dim_key]
            variants = dim_cfg.get("variants", [])
            if not variants:
                print(f"[Ablation] Dimension '{dim_key}' has no variants, skipping")
                continue

            # Incremental skip: every variant × every query is cached → restore responses from cache + re-score
            if _is_dimension_cached(variants, queries):
                title = dim_meta.get(dim_key, {}).get("title", dim_key)
                print(f"\n>>> {title} ({dim_key}) — SKIPPED (all cached, re-scoring)")
                # Restore responses from cache
                cached_responses: Dict[str, List[str]] = {}
                cached_traces: Dict[str, List[List[dict]]] = {}
                for v in variants:
                    vname = v["name"]
                    v_cache = _ABLATION_RESPONSE_CACHE.get(vname, {})
                    resp_list = []
                    trace_list = []
                    for q in queries:
                        val = v_cache.get(q["query"], "")
                        if isinstance(val, dict) and "resp" in val:
                            resp_list.append(val["resp"])
                            trace_list.append(val.get("trace", []))
                        else:
                            resp_list.append(val if isinstance(val, str) else "")
                            trace_list.append([])
                    cached_responses[vname] = resp_list
                    cached_traces[vname] = trace_list
                # Re-score
                dim_results = _run_scoring(
                    queries, cached_responses, evaluator_llm=evaluator_fn,
                    evaluator_llm_bulk=evaluator_bulk_fn,
                    selfcheck=None,
                    n_selfcheck_values=ablation_scoring_cfg.get("n_selfcheck_values") or Config.EVAL_N_SELFCHECK_VALUES,
                    step_traces=cached_traces,
                    enable_tolerance=ablation_scoring_cfg.get("tolerance", False),
                    enable_flesch=ablation_scoring_cfg.get("flesch", True),
                )
                all_results[dim_key] = dim_results
                continue

            title = dim_meta.get(dim_key, {}).get("title", dim_key)
            print(f"\n>>> {title} ({dim_key})")
            try:
                dim_results, dim_times = _run_dimension_variants(
                    dim_key, variants, queries, evaluator_fn, evaluator_bulk_fn, selfcheck_fn,
                    verbose, meta=dim_meta.get(dim_key),
                    scoring_cfg=ablation_scoring_cfg,
                )
                all_results[dim_key] = dim_results
                all_variant_times[dim_key] = dim_times
                _partial_save()  # save intermediate results after each completed dimension
            except KeyboardInterrupt:
                raise  # propagate to the outer try
            except Exception as e:
                print(f"  [ERROR] {dim_key} failed: {e}")
                import traceback
                traceback.print_exc()
                _partial_save()  # save whatever exists even for the failed dimension

    except KeyboardInterrupt:
        print("\n\n  [Interrupt] Ablation interrupted by user. Saving partial results...")
        _save_disk_cache()
        _partial_save()
        print("  Partial results saved. Run --ablation again to resume (disk cache loaded).")
        return all_results

    _save_disk_cache()
    _save_responses_json(log_dir, queries)
    if all_results:
        _save_report(all_results, log_dir, dim_meta=dim_meta,
                     variant_times=all_variant_times, dim_suffix="")

    print("\n" + "=" * 60)
    print("  Ablation Study Complete")
    print("=" * 60)

    return all_results
