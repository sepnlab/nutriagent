"""
NutriCHAT — CLI entry point.

Usage:
    python -m src.main --query "What is the calcium content of limestone?"
    python -m src.main --interactive
    python -m src.main --build-db
    python -m src.main --evaluate
    python -m src.main --compare
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import setup_logging, get_logger
from src.utils.cost_tracker import CostTracker
from src.utils.query_loader import load_queries

from src.config import Config
from src.tools import get_all_tools
from src.agent.hybrid_engine import NutriCHATEngine


_LOG_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
)


def _create_log_dir(**extra_cfg) -> str:
    """Create a timestamped log directory and save a minimal config.yaml for reproducibility."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_mode = extra_cfg.get("run_mode", "evaluate")
    amode = extra_cfg.get("agent_mode", "single")
    prefix = f"comp" if run_mode == "compare" else f"eval_{amode}"
    species = getattr(Config, "SPECIES", None)
    if species and species != "broiler":
        prefix += f"_{species}"
    log_dir = os.path.join(_LOG_DIR, f"{prefix}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    try:
        import yaml

        # Only keep the attributes that matter for reproducibility
        _KEY_INCLUDE_LIST = {
            # experiment setup
            "SPECIES", "agent_mode", "query_set", "categories",
            # sampling & evaluation
            "EVAL_SAMPLE_RATIO", "EVAL_N_SELFCHECK_VALUES", "EVAL_N_SELFCHECK_GEN_SAMPLES",
            "evaluator_llm", "selfcheck", "tolerance", "flesch",
            # model selection
            "LLM_MODEL", "EVALUATOR_MODEL", "SELFCHECK_MODEL", "SELFCHECK_BASE_URL",
            "EMBEDDING_PROVIDER", "LOCAL_EMBEDDING_PATH", "LOCAL_EMBEDDING_DEVICE",
            "SILICONFLOW_EMBEDDING_MODEL", "NLI_MODEL_PATH",
            # generation params
            "TEMPERATURE", "MAX_TOKENS", "RAG_K",
            "MAX_ITERATIONS", "EARLY_STOPPING",
            "ROUTER_TEMPERATURE", "ROUTER_MAX_TOKENS", "SYNTHESIS_TEMPERATURE",
            "SELFCHECK_N_SAMPLES",
            # API
            "API_TIMEOUT", "API_RETRY_MAX_ATTEMPTS", "API_RETRY_MIN_WAIT", "API_RETRY_MAX_WAIT",
            "API_BASELINE_TIMEOUT",
        }

        resolved = {}
        for attr in _KEY_INCLUDE_LIST:
            if hasattr(Config, attr):
                resolved[attr] = getattr(Config, attr)
            elif attr in extra_cfg:
                resolved[attr] = extra_cfg[attr]
        resolved.update({k: v for k, v in extra_cfg.items() if k not in resolved})
        # Normalize to lowercase for consistent naming
        resolved = {k.lower(): v for k, v in resolved.items()}

        # scoring subsection — grouped separately for --from-config
        scoring = {
            "evaluator_llm": extra_cfg.get("evaluator_llm", True),
            "selfcheck": extra_cfg.get("selfcheck", True),
            "hallucination_ratio": extra_cfg.get("hallucination_ratio"),
            "tolerance": extra_cfg.get("tolerance", True),
            "flesch": extra_cfg.get("flesch", True),
            "n_selfcheck_values": extra_cfg.get("n_selfcheck_values") or Config.EVAL_N_SELFCHECK_VALUES,
            "shuffle_responses": extra_cfg.get("shuffle_responses"),
        }
        for k in ("evaluator_llm", "selfcheck", "tolerance", "flesch",
                   "hallucination_ratio", "n_selfcheck_values", "shuffle_responses",
                   "breakdown"):
            resolved.pop(k, None)

        # Resolve full model configs for reproducibility
        from src.config import _NUTRI_CHAT_MODEL, _EVALUATOR_MODEL, _SELFCHECK_CONFIG

        def _safe_model(m: dict) -> dict:
            return {k: v for k, v in m.items()
                    if k not in ("api_key", "api_key_env") and v is not None}

        doc = {}
        if resolved:
            doc["config"] = dict(sorted(resolved.items()))
        if _NUTRI_CHAT_MODEL:
            doc["nutrichat"] = _safe_model(_NUTRI_CHAT_MODEL)
        if _EVALUATOR_MODEL:
            doc["evaluator"] = _safe_model(_EVALUATOR_MODEL)
        if _SELFCHECK_CONFIG:
            doc["selfcheck"] = _safe_model(_SELFCHECK_CONFIG)
        doc["scoring"] = {k: v for k, v in scoring.items() if v is not None}

        with open(os.path.join(log_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(doc, f, default_flow_style=False, allow_unicode=True)
    except Exception:
        pass

    return log_dir


def setup_engine(verbose=False, max_tokens=None, rag_enabled=True, mode="single"):
    tools = get_all_tools()
    return NutriCHATEngine(
        tools=tools, verbose=verbose,
        max_tokens=max_tokens or Config.MAX_TOKENS,
        rag_enabled=rag_enabled,
        mode=mode,
    )


def build_database(force=False):
    engine = setup_engine(verbose=True)
    count = engine.build_vector_db(force_rebuild=force)
    print(f"\nPoultryDB built with {count} documents")
    return count


def process_query(engine, query: str):
    print(f"\n{'='*60}\nQuery: {query}\n{'='*60}")
    t0 = time.time()
    result = engine.answer(query)
    elapsed = time.time() - t0
    print(f"\nResponse ({elapsed:.2f}s):\n{'-'*40}\n{result['response']}")
    print(f"\nTools used: {result['tools_used']}")
    print(f"RAG context: {'Yes' if result['rag_context'] else 'No'}")
    print(f"ReAct steps: {len(result['steps'])}")
    return result


def interactive_mode(engine):
    print("\n" + "=" * 60)
    print("  NutriCHAT - Interactive Mode")
    print("  Type 'quit' to exit, 'status' for system info")
    print("=" * 60)
    while True:
        try:
            query = input("\n[Poultry] Enter your poultry nutrition question: ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                break
            if query.lower() == "status":
                print(f"  Model: {Config.LLM_MODEL}")
                print(f"  Max tokens: {Config.MAX_TOKENS}")
                print(f"  Agent mode: {engine.mode.upper()}")
                print(f"  Tools: 4 expert-designed tools")
                print(f"  RAG: Enabled")
                continue
            process_query(engine, query)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


def _load_queries(filepath: str, query_set: str = "all") -> List[dict]:
    """Load queries, optionally filtered by set (all/chinese/western)."""
    from src.utils.query_loader import load_queries
    return load_queries(queries_file=filepath, query_set=query_set)


def _track_usage(engine, result: dict) -> dict:
    """Extract token usage from engine result, works for single and multi mode."""
    usage = {"model": Config.LLM_MODEL, "prompt_tokens": 0, "completion_tokens": 0}
    pt = result.get("prompt_tokens", 0)
    ct = result.get("completion_tokens", 0)
    if pt or ct:
        usage["prompt_tokens"] = pt
        usage["completion_tokens"] = ct
    elif engine.mode != "multi":
        agent_usage = getattr(engine, "_last_agent_usage", {}) or {}
        usage["prompt_tokens"] = agent_usage.get("prompt_tokens", 0)
        usage["completion_tokens"] = agent_usage.get("completion_tokens", 0)
    return usage


def _batch_timeout(n: int, n_workers: int, query_timeout: int) -> int:
    """Global safety timeout for a batch of n queries with n_workers slots.

    The previous implementation used ``query_timeout + 30`` regardless of batch
    size, which killed a 54-query run after ~3 minutes even though every query
    was still within its own timeout. Scale by the number of batches instead.
    """
    if n <= 0 or n_workers <= 0:
        return 60
    n_batches = (n + n_workers - 1) // n_workers
    return query_timeout * n_batches + 60


def _run_queries(engine, queries: List[dict], label: str,
                  max_concurrent: int = None) -> tuple:
    """运行查询，返回 (responses, cost_tracker, steps_data)。

    并行执行：每个线程创建独立 engine 实例（LangGraph 非线程安全），
    线程数 = min(查询数, max_concurrent)。每条查询超时上限 = Config.API_TIMEOUT × 5。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed as ac, TimeoutError as FutureTimeout

    query_timeout = Config.API_TIMEOUT * 5
    n = len(queries)
    if n == 0:
        return [], CostTracker(), []
    n_workers = min(n, max_concurrent or Config.API_MAX_CONCURRENT)

    # 每个线程创建独立 engine（共享 tools，独立 graph）
    _engines = []
    for _ in range(n_workers):
        _engines.append(setup_engine(
            verbose=engine.verbose,
            max_tokens=engine.max_tokens,
            rag_enabled=engine.rag_enabled,
            mode=engine.mode,
        ))

    def _safe_answer(q, worker_idx):
        query_str = q["query"] if isinstance(q, dict) else q
        t0 = time.time()
        eng = _engines[worker_idx % n_workers]

        # Per-query timeout: run answer() in a separate single-thread pool
        # so we can enforce a hard timeout even if graph.stream() loops.
        with ThreadPoolExecutor(max_workers=1) as timeout_pool:
            answer_future = timeout_pool.submit(eng.answer, query_str, use_cache=True)
            try:
                result = answer_future.result(timeout=query_timeout)
            except FutureTimeout:
                answer_future.cancel()
                raise FutureTimeout(f"Query exceeded {query_timeout}s")

        elapsed = result.get("elapsed_s") or (time.time() - t0)
        usage = _track_usage(eng, result)
        return result, usage, elapsed

    print(f"\n--- {label} (parallel={n_workers}) ---")
    tracker = CostTracker()
    responses_arr: List[str] = [""] * n
    steps_arr: List[dict] = [{} for _ in range(n)]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_safe_answer, q, i): i for i, q in enumerate(queries)}
        # Global safety timeout must scale with the number of batches. The old
        # `query_timeout + 30` killed slow 54-query runs after ~3 minutes even
        # though every individual query was still within its own timeout.
        global_timeout = _batch_timeout(n, n_workers, query_timeout)
        try:
            for f in tqdm(ac(futures, timeout=global_timeout), total=n,
                          desc=label.split(": ")[-1], unit="q"):
                i = futures[f]
                q = queries[i]
                _t0 = time.time()
                try:
                    result, usage, elapsed = f.result(timeout=0)
                    responses_arr[i] = result["response"]
                    tracker.record(
                        model=usage.get("model", Config.LLM_MODEL),
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        elapsed=elapsed,
                    )
                    steps_arr[i] = {
                        "id": q.get("id", f"q{i}"),
                        "query": q["query"],
                        "steps": result.get("steps", []),
                        "tools_used": result.get("tools_used", []),
                        "rag_context": (result.get("rag_context", "")[:200] if result.get("rag_context") else ""),
                        "react_domains": result.get("react_domains", []),
                        "specialists_used": result.get("specialists_used", []),
                        "elapsed_s": round(elapsed, 1),
                        "mode": result.get("mode", engine.mode),
                    }
                except FutureTimeout:
                    elapsed = time.time() - _t0
                    responses_arr[i] = f"[Timeout] Query exceeded {query_timeout}s"
                    steps_arr[i] = {
                        "id": q.get("id", f"q{i}"), "query": q["query"],
                        "steps": [], "error": f"[Timeout] {query_timeout}s",
                    }
                except Exception as e:
                    responses_arr[i] = f"[Error] {e}"
                    steps_arr[i] = {
                        "id": q.get("id", f"q{i}"), "query": q["query"],
                        "steps": [], "error": str(e),
                    }
        except FutureTimeout:
            # Global safety timeout: mark any remaining unfinished queries
            from src.logger import get_logger
            get_logger("compare").warning(
                f"as_completed global timeout — marking remaining queries as [Timeout]")
            for f, i in futures.items():
                if not f.done():
                    q = queries[i]
                    responses_arr[i] = f"[Timeout] Query exceeded {query_timeout}s"
                    steps_arr[i] = {
                        "id": q.get("id", f"q{i}"), "query": q["query"],
                        "steps": [], "error": f"[Timeout] {query_timeout}s (global)",
                    }

    return list(responses_arr), tracker, list(steps_arr)


def _save_step_traces(steps_data: List[dict], log_dir: str, suffix: str = ""):
    """保存中间步骤到日志目录。"""
    filename = f"step_traces{suffix}.json"
    filepath = os.path.join(log_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(steps_data, f, indent=2, ensure_ascii=False)
    print(f"[Steps] Saved {len(steps_data)} step traces to {filename}")


def _preload_models():
    """预加载 NLI + Embedding 模型（并行），避免评估时冷启动。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.logger import get_logger
    logger = get_logger("preload")

    def _load_nli():
        from src.evaluation.selfcheck import SelfCheckGPT
        from src.config import Config
        SelfCheckGPT(device=Config.NLI_DEVICE)._load_model()

    def _load_embedding():
        from src.rag.poultry_db import create_embedding_model
        create_embedding_model()

    logger.info("Pre-loading models...")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(_load_nli): "NLI", pool.submit(_load_embedding): "Embedding"}
        for f in as_completed(futs):
            name = futs[f]
            try:
                f.result()
                logger.info("  %s ready", name)
            except Exception as e:
                logger.warning("  %s failed: %s", name, e)


def run_evaluation(engine, queries_file=None, query_set="all", agent_mode="single",
                   detailed=False, categories=None, scoring=None, quick=False):
    """NutriCHAT evaluation: runs queries + EvaluatorLLM + Flesch + tolerance accuracy.

    Args:
        scoring: dict from evaluate.yaml scoring section, controls which metrics run
                 and provides n_selfcheck_values override.
    """
    print("\n" + "=" * 60)
    print("  NutriCHAT Evaluation Suite")
    print("=" * 60)
    _t0 = time.time()
    _preload_models()

    scoring = scoring or {}
    enable_eval_llm = scoring.get("evaluator_llm", True)
    enable_selfcheck = scoring.get("selfcheck", True)
    enable_tolerance = scoring.get("tolerance", True)
    enable_flesch = scoring.get("flesch", True)
    n_selfcheck_values = scoring.get("n_selfcheck_values") or Config.EVAL_N_SELFCHECK_VALUES

    from src.evaluation.evaluator import (
        run_evaluation_pipeline, print_report_table, save_report, save_raw_responses,
        print_cost_table, compute_breakdown, print_breakdown_tables, EvalReport,
        format_report_table,
    )

    log_dir = _create_log_dir(
        agent_mode=agent_mode, query_set=query_set, categories=categories,
        evaluator_llm=enable_eval_llm, selfcheck=enable_selfcheck,
        tolerance=enable_tolerance, flesch=enable_flesch,
        n_selfcheck_values=n_selfcheck_values,
        quick=quick,
    )
    queries = load_queries(
        queries_file=queries_file or Config.EVALUATION_QUERIES_FILE,
        query_set=query_set,
        sample_ratio=Config.EVAL_SAMPLE_RATIO,
        categories=categories,
    )
    print(f"[Evaluator] Loaded {len(queries)} queries (query_set={query_set})")

    # Create scorers
    from src.evaluation.evaluator_llm import EvaluatorLLM
    if enable_eval_llm:
        EvaluatorLLM.shuffle = scoring.get("shuffle_responses", False)
        ev_llm = EvaluatorLLM()
    else:
        ev_llm = None
    ev_llm_fn = ev_llm.score_responses if ev_llm else None

    from src.evaluation.selfcheck import SelfCheckGPT
    sc = SelfCheckGPT(device=Config.NLI_DEVICE,
                       sample_extra_body=Config.SELFCHECK_EXTRA_BODY) if enable_selfcheck else None
    sc_fn = sc.compute_score if sc else None

    # ── Run queries ──
    if agent_mode == "both":
        print("\n--- Modes: SINGLE + MULTI (parallel) ---")
        engine_single = setup_engine(
            verbose=engine.verbose, max_tokens=engine.max_tokens,
            rag_enabled=engine.rag_enabled, mode="single",
        )
        # both 模式下减半并发，避免 API 429 限流
        _half_concurrent = max(1, Config.API_MAX_CONCURRENT // 2)

        def _run_both_mode(mode_name: str, eng) -> tuple:
            resp, tracker, steps = _run_queries(eng, queries, f"{mode_name}",
                                                 max_concurrent=_half_concurrent)
            _save_step_traces(steps, log_dir, suffix=f"_{mode_name.lower()}")
            return mode_name, resp, tracker, steps

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(_run_both_mode, "SINGLE", engine_single): "SINGLE",
                pool.submit(_run_both_mode, "MULTI", engine): "MULTI",
            }
            both_results = {}
            for f in as_completed(futures):
                mode_name, resp, tracker, steps = f.result()
                tqdm.write(f"  >>> [{mode_name}] completed")
                both_results[mode_name] = (resp, tracker, steps)

        single_responses, single_tracker, single_steps = both_results["SINGLE"]
        multi_responses, multi_tracker, multi_steps = both_results["MULTI"]

        mname = Config.LLM_MODEL
        single_key = f"NutriCHAT-Single ({mname})"
        multi_key = f"NutriCHAT-Multi ({mname})"
        all_responses = {
            single_key: single_responses,
            multi_key: multi_responses,
        }
        save_raw_responses(queries, all_responses, log_dir)

        _step_traces = {
            single_key: [s["steps"] for s in single_steps],
            multi_key: [s["steps"] for s in multi_steps],
        }

        nutri_resp = single_responses
        bl_resp = {multi_key: multi_responses}
        methods = [single_key, multi_key]
        cost_data = {
            single_key: single_tracker.summary().get("_total", {}),
            multi_key: multi_tracker.summary().get("_total", {}),
        }
    else:
        nutrichat_responses, nutrichat_tracker, nutrichat_steps = _run_queries(
            engine, queries, f"Running NutriCHAT ({agent_mode})")
        _save_step_traces(nutrichat_steps, log_dir)

        nchat_name = f"NutriCHAT ({Config.LLM_MODEL})"
        all_responses = {nchat_name: nutrichat_responses}
        save_raw_responses(queries, all_responses, log_dir)

        _step_traces = {nchat_name: [s["steps"] for s in nutrichat_steps]}

        nutri_resp = nutrichat_responses
        bl_resp = {}
        methods = [nchat_name]
        cost_data = {nchat_name: nutrichat_tracker.summary().get("_total", {})}

    # ── Phase: Independent scoring (EvaluatorLLM + SelfCheck in parallel) ──
    reports: Dict[str, EvalReport] = {}

    _primary_name = methods[0]

    def _run_llm_scoring():
        """EvaluatorLLM + Flesch + Tolerance."""
        r = run_evaluation_pipeline(
            queries=queries,
            nutrichat_responses=nutri_resp,
            baseline_responses=bl_resp,
            evaluator_llm=ev_llm_fn,
            evaluator_llm_bulk=ev_llm.score_bulk if ev_llm else None,
            selfcheck=None,
            n_selfcheck_values=n_selfcheck_values,
            enable_tolerance=enable_tolerance,
            enable_flesch=enable_flesch,
            return_per_query=detailed,
            step_traces=_step_traces,
            nutrichat_name=_primary_name,
        )
        return r

    def _run_selfcheck_scoring():
        """SelfCheckGPT only."""
        return run_evaluation_pipeline(
            queries=queries,
            nutrichat_responses=nutri_resp,
            baseline_responses=bl_resp,
            evaluator_llm=None,
            selfcheck=sc_fn,
            n_selfcheck_values=n_selfcheck_values,
            enable_tolerance=False,
            enable_flesch=False,
            nutrichat_name=_primary_name,
        )

    scoring_tasks = {}
    bd = None
    with ThreadPoolExecutor(max_workers=2) as pool:
        if enable_eval_llm:
            scoring_tasks[pool.submit(_run_llm_scoring)] = "EvaluatorLLM"
        if enable_selfcheck and sc_fn:
            scoring_tasks[pool.submit(_run_selfcheck_scoring)] = "SelfCheckGPT"

        for f in as_completed(scoring_tasks):
            task_name = scoring_tasks[f]
            result = f.result()
            if not result:
                continue

            if task_name == "EvaluatorLLM":
                if detailed and isinstance(result, tuple):
                    result, per_query = result
                    bd = compute_breakdown(per_query, methods, reports=reports)
                    print_breakdown_tables(bd)

                _merge_eval_reports(reports, result)
                print_report_table(reports, cost_data=cost_data)

                # Statistical significance
                nutrichat_ref = next((n for n in reports if n.startswith("NutriCHAT")), None)
                if nutrichat_ref and len(reports) > 1:
                    from src.evaluation.evaluator import print_significance_table, compute_significance_test
                    comparisons = {}
                    for name, report in reports.items():
                        if name != nutrichat_ref:
                            comparisons[name] = compute_significance_test(reports[nutrichat_ref], report)
                    print_significance_table(nutrichat_ref, comparisons)

            elif task_name == "SelfCheckGPT":
                _merge_eval_reports(reports, result)

                has_halluc = any(r.hallucination_scores for r in reports.values())
                if has_halluc:
                    for line in format_report_table(
                        {m: reports[m] for m in methods if m in reports},
                        cost_data=cost_data,
                    ).split("\n"):
                        tqdm.write(line)

    # If no evaluator_llm, still run Flesch + Tolerance
    if not enable_eval_llm and not reports:
        reports = run_evaluation_pipeline(
            queries=queries,
            nutrichat_responses=nutri_resp,
            baseline_responses=bl_resp,
            evaluator_llm=None, selfcheck=None,
            enable_tolerance=enable_tolerance,
            enable_flesch=enable_flesch,
            nutrichat_name=methods[0],
        )

    # ── Final: cost + save ──
    total_elapsed = time.time() - _t0
    print_cost_table(cost_data)
    save_report(reports, log_dir, cost_data=cost_data, queries=queries, breakdown=bd,
                total_time_sec=total_elapsed)


def _merge_eval_reports(base, src):
    """Merge src reports into base (reports dict), keeping existing data."""
    from src.evaluation.evaluator import EvalReport
    for method, s in src.items():
        if method not in base:
            base[method] = EvalReport(method)
        t = base[method]
        for attr in ("correctness", "precision", "scientific_depth", "flesch_scores",
                      "hallucination_scores", "tolerance_p2", "tolerance_p5",
                      "faithfulness", "semantic_faithfulness"):
            src_list = getattr(s, attr, [])
            if src_list:
                getattr(t, attr).extend(src_list)
        # hallucination_multi: Dict[int, List[float]] — merge per-N
        for n, vals in s.hallucination_multi.items():
            t.hallucination_multi.setdefault(n, []).extend(vals)


def _quick_report(queries, responses, method_name, enable_flesch=True, enable_tolerance=True):
    """Compute Flesch reading ease + tolerance accuracy for a single model.

    Returns an EvalReport with only those two metrics populated.
    """
    from src.evaluation.evaluator import EvalReport, compute_flesch, evaluate_tolerance
    report = EvalReport(method_name)
    if enable_flesch:
        for resp in responses:
            try:
                report.add(flesch=compute_flesch(resp))
            except Exception:
                pass
    if enable_tolerance:
        for i, q in enumerate(queries):
            gt = q.get("expected_value")
            if gt is not None and i < len(responses) and responses[i]:
                try:
                    report.add(
                        tol2=evaluate_tolerance(responses[i], gt, 0.02),
                        tol5=evaluate_tolerance(responses[i], gt, 0.05),
                    )
                except Exception:
                    pass
    return report


def run_comparison(engine, queries_file=None, query_set="all", detailed=False,
                   categories=None, baselines=None, skip_nutrichat=False, scoring=None,
                   agent_mode="single", quick=False):
    """Multi-model comparison: NutriCHAT + baselines with streaming results.

    Baselines run in parallel with per-query timeout. As each model finishes,
    intermediate results (Flesch + tolerance) are shown immediately so you
    don't have to wait for all models to complete.

    Args:
        scoring: dict from evaluate.yaml scoring section, controls which metrics run
                 and provides n_selfcheck_values override.
    """
    from concurrent.futures import TimeoutError as FutureTimeout

    print("\n" + "=" * 60)
    print("  NutriCHAT Multi-Model Comparison")
    print("=" * 60)
    _preload_models()
    _t0 = time.time()

    scoring = scoring or {}
    enable_eval_llm = scoring.get("evaluator_llm", True)
    enable_selfcheck = scoring.get("selfcheck", True)
    enable_tolerance = scoring.get("tolerance", True)
    enable_flesch = scoring.get("flesch", True)
    n_selfcheck_values = scoring.get("n_selfcheck_values") or Config.EVAL_N_SELFCHECK_VALUES

    from src.evaluation.evaluator import (
        run_evaluation_pipeline, print_report_table, save_report, save_raw_responses,
        print_cost_table, compute_breakdown, print_breakdown_tables,
        format_report_table, EvalReport,
    )

    log_dir = _create_log_dir(
        agent_mode=agent_mode, run_mode="compare", query_set=query_set,
        categories=categories, baselines=baselines, skip_nutrichat=skip_nutrichat,
        evaluator_llm=enable_eval_llm, selfcheck=enable_selfcheck,
        tolerance=enable_tolerance, flesch=enable_flesch,
        n_selfcheck_values=n_selfcheck_values,
        quick=quick,
    )
    queries = load_queries(
        queries_file=queries_file or Config.EVALUATION_QUERIES_FILE,
        query_set=query_set,
        sample_ratio=Config.EVAL_SAMPLE_RATIO,
        categories=categories,
    )
    print(f"[Evaluator] Loaded {len(queries)} queries (query_set={query_set})")

    from src.evaluation.evaluator_llm import EvaluatorLLM
    from src.evaluation.baseline_runner import BaselineRunner

    # Conditionally create scorers
    if enable_eval_llm:
        EvaluatorLLM.shuffle = scoring.get("shuffle_responses", False)
        ev_llm = EvaluatorLLM()
    else:
        ev_llm = None
    ev_llm_fn = ev_llm.score_responses if ev_llm else None

    all_responses: Dict[str, List[str]] = {}
    all_step_traces: Dict[str, List[List[dict]]] = {}
    nutrichat_tracker = None
    nutrichat_responses = []
    _extra_nutri_trackers: Dict[str, CostTracker] = {}
    if not skip_nutrichat:
        if agent_mode == "both":
            print("\n--- Phase 1: NutriCHAT (SINGLE + MULTI) ---")
            engine_single = setup_engine(
                verbose=engine.verbose, max_tokens=engine.max_tokens,
                rag_enabled=engine.rag_enabled, mode="single",
            )
            # both 模式下减半并发，避免 API 429 限流
            _half_concurrent = max(1, Config.API_MAX_CONCURRENT // 2)

            def _run_both_mode(mode_label: str, eng) -> tuple:
                resp, tracker, steps = _run_queries(eng, queries, f"NutriCHAT-{mode_label}",
                                                     max_concurrent=_half_concurrent)
                _save_step_traces(steps, log_dir, suffix=f"_{mode_label.lower()}")
                return mode_label, resp, tracker, steps

            both_results = {}
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    pool.submit(_run_both_mode, "SINGLE", engine_single): "SINGLE",
                    pool.submit(_run_both_mode, "MULTI", engine): "MULTI",
                }
                for f in as_completed(futures):
                    mode_label, resp, tracker, steps = f.result()
                    tqdm.write(f"  >>> [NutriCHAT-{mode_label}] completed")
                    both_results[mode_label] = (resp, tracker, steps)

            mname = Config.LLM_MODEL
            single_key = f"NutriCHAT-Single ({mname})"
            multi_key = f"NutriCHAT-Multi ({mname})"
            single_resp, single_tracker, single_steps = both_results["SINGLE"]
            multi_resp, multi_tracker, multi_steps = both_results["MULTI"]
            all_responses[single_key] = single_resp
            all_responses[multi_key] = multi_resp
            if single_steps:
                all_step_traces[single_key] = [s["steps"] for s in single_steps]
            if multi_steps:
                all_step_traces[multi_key] = [s["steps"] for s in multi_steps]

            nchat_name = single_key
            nutrichat_responses = single_resp
            nutrichat_tracker = single_tracker
            _extra_nutri_trackers[multi_key] = multi_tracker
        else:
            nchat_name = f"NutriCHAT ({Config.LLM_MODEL})"
            print("\n--- Phase 1: NutriCHAT ---")
            nutrichat_responses, nutrichat_tracker, nutrichat_steps = _run_queries(engine, queries, nchat_name)
            all_responses[nchat_name] = nutrichat_responses
            if nutrichat_steps:
                all_step_traces[nchat_name] = [s["steps"] for s in nutrichat_steps]

        # Save NutriCHAT responses immediately (before baselines, so partial results survive)
        save_raw_responses(queries, all_responses, log_dir)

        print("\n--- Phase 2: Baseline Models ---")
    else:
        nchat_name = "NutriCHAT"
        print("\n--- Baseline Models Only (skipping NutriCHAT) ---")

    runner = BaselineRunner()
    runner.load_baselines(filter_names=baselines)
    model_names = list(runner.clients.keys())

    # Build SelfCheck map (method -> compute_score callable)
    from src.evaluation.selfcheck import SelfCheckGPT
    sc_map = {}
    if enable_selfcheck:
        sc_map[nchat_name] = SelfCheckGPT(
            device=Config.NLI_DEVICE,
            sample_extra_body=Config.SELFCHECK_EXTRA_BODY,
        ).compute_score
        # Add second NutriCHAT key for both-mode comparison
        if agent_mode == "both":
            _multi_sc_key = f"NutriCHAT-Multi ({Config.LLM_MODEL})"
            sc_map[_multi_sc_key] = SelfCheckGPT(
                device=Config.NLI_DEVICE,
                sample_extra_body=Config.SELFCHECK_EXTRA_BODY,
            ).compute_score
        for bl_entry in runner.baselines:
            sc_map[bl_entry["name"]] = SelfCheckGPT(
                device=Config.NLI_DEVICE,
                sample_model=bl_entry.get("model"),
                sample_base_url=bl_entry.get("base_url"),
                sample_api_key=os.getenv(bl_entry.get("api_key_env", ""), ""),
                sample_extra_body=bl_entry.get("extra_body"),
            ).compute_score

    baseline_trackers: Dict[str, CostTracker] = {}

    # Per-query timeout for baselines (same as NutriCHAT)
    query_timeout = Config.API_TIMEOUT * 5

    def _run_baseline_model(name: str):
        """Run all queries for a single baseline with per-query timeout.

        Queries run in parallel (up to 3 concurrent per model) for speed.
        Each query has a hard timeout (query_timeout).
        """
        from concurrent.futures import ThreadPoolExecutor as TPE, as_completed as ac2, TimeoutError as FT2
        import threading

        responses = [None] * len(queries)
        tracker = CostTracker()
        usage_lock = threading.Lock()
        bl_workers = min(len(queries), max(1, Config.API_MAX_CONCURRENT // 2))

        def _run_one(idx_q):
            idx, q = idx_q
            t0 = time.time()
            resp, usage = runner.run_query_with_usage(name, q["query"])
            elapsed = time.time() - t0
            return idx, resp, usage, elapsed

        with TPE(max_workers=bl_workers) as bl_pool:
            bl_futures = {bl_pool.submit(_run_one, (i, q)): i for i, q in enumerate(queries)}
            done = 0
            for f in ac2(bl_futures):
                i = bl_futures[f]
                try:
                    idx, resp, usage, elapsed = f.result(timeout=query_timeout)
                    responses[idx] = resp
                    with usage_lock:
                        tracker.record(
                            model=usage.get("_pricing_key") or usage.get("model") or name,
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0),
                            elapsed=elapsed,
                        )
                except (FT2, Exception) as e:
                    responses[idx] = f"[Error] {e}"
                done += 1

        # Fill any None slots (shouldn't happen, but safety)
        responses = [r if r is not None else "[Error] No response" for r in responses]
        return name, responses, tracker

    if model_names:
        # Quick metrics for NutriCHAT (computed once for interim display)
        interim_reports: Dict[str, EvalReport] = {}
        if not skip_nutrichat and (enable_flesch or enable_tolerance):
            interim_reports[nchat_name] = _quick_report(
                queries, nutrichat_responses, nchat_name,
                enable_flesch, enable_tolerance,
            )
            if agent_mode == "both":
                _multi_interim_key = f"NutriCHAT-Multi ({Config.LLM_MODEL})"
                _multi_interim_resp = all_responses[_multi_interim_key]
                interim_reports[_multi_interim_key] = _quick_report(
                    queries, _multi_interim_resp, _multi_interim_key,
                    enable_flesch, enable_tolerance,
                )

        print(f"\n  Running {len(model_names)} models in parallel "
              f"(timeout={query_timeout}s per query)...")
        n_workers = min(len(model_names), 4)
        # Build usage_by_model incrementally so partial reports include cost data
        usage_by_model = {}
        if not skip_nutrichat:
            usage_by_model[nchat_name] = nutrichat_tracker.summary().get("_total", {})

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_baseline_model, n): n for n in model_names}
            for f in as_completed(futures):
                name, responses, tracker = f.result()
                all_responses[name] = responses
                baseline_trackers[name] = tracker
                usage_by_model[name] = tracker.summary().get("_total", {})

                # Compute quick metrics for this baseline
                if enable_flesch or enable_tolerance:
                    interim_reports[name] = _quick_report(
                        queries, responses, name,
                        enable_flesch, enable_tolerance,
                    )

                # Incremental save: raw responses + partial report each time a model finishes
                save_raw_responses(queries, all_responses, log_dir)
                # Save partial report so results survive interruption
                _partial_reports = {}
                for _name, _rpt in interim_reports.items():
                    _partial_reports[_name] = _rpt
                if _partial_reports:
                    save_report(_partial_reports, log_dir, cost_data=usage_by_model, queries=queries)

                # Compact one-line interim: FRES + tolerance for this model
                report = interim_reports.get(name)
                if report:
                    f_res = report.flesch_scores
                    fres_str = f"{sum(f_res)/len(f_res):.0f}" if f_res else "-"
                    t2 = report.tolerance_p2
                    t2_str = f"{sum(1 for v in t2 if v and v > 0)/len(t2)*100:.0f}%" if t2 else "-"
                    tqdm.write(f"  >>> [{name}] done  FRES={fres_str}  Tol±2%={t2_str}  → {log_dir}/")
                else:
                    tqdm.write(f"  >>> [{name}] done  → {log_dir}/")
    else:
        print("  No baselines available (check API keys)")
        if not skip_nutrichat:
            save_raw_responses(queries, {nchat_name: nutrichat_responses}, log_dir)

    # Finalize usage_by_model (may already be partially built in the baseline loop)
    if 'usage_by_model' not in locals():
        usage_by_model = {}
    if nutrichat_tracker and nchat_name not in usage_by_model:
        usage_by_model[nchat_name] = nutrichat_tracker.summary().get("_total", {})
    for name, tracker in _extra_nutri_trackers.items():
        usage_by_model[name] = tracker.summary().get("_total", {})
    for name, tracker in baseline_trackers.items():
        if name not in usage_by_model:
            usage_by_model[name] = tracker.summary().get("_total", {})

    # ── Phase 3: Independent scoring (no data dependencies) ──
    all_methods = list(all_responses.keys())
    if not all_methods:
        print("  No responses to score.")
        return

    nutri_for_scoring = all_responses.get(nchat_name,
                                          all_responses[all_methods[0]])
    bl_for_scoring = {k: v for k, v in all_responses.items() if k != nchat_name}

    reports: Dict[str, EvalReport] = {}

    def _score_llm():
        """EvaluatorLLM + Flesch + Tolerance + Faithfulness."""
        r = run_evaluation_pipeline(
            queries=queries,
            nutrichat_responses=nutri_for_scoring,
            baseline_responses=bl_for_scoring,
            evaluator_llm=ev_llm_fn,
            evaluator_llm_bulk=ev_llm.score_bulk if ev_llm else None,
            selfcheck=None,
            n_selfcheck_values=n_selfcheck_values,
            enable_tolerance=enable_tolerance,
            enable_flesch=enable_flesch,
            return_per_query=detailed,
            step_traces=None,
            nutrichat_name=nchat_name,
        )
        return r

    def _score_selfcheck():
        """SelfCheckGPT only."""
        return run_evaluation_pipeline(
            queries=queries,
            nutrichat_responses=nutri_for_scoring,
            baseline_responses=bl_for_scoring,
            evaluator_llm=None,
            selfcheck=sc_map,
            n_selfcheck_values=n_selfcheck_values,
            enable_tolerance=False, enable_flesch=False,
            nutrichat_name=nchat_name,
        )

    def _score_faithfulness():
        """Faithfulness only."""
        return run_evaluation_pipeline(
            queries=queries,
            nutrichat_responses=nutri_for_scoring,
            baseline_responses=bl_for_scoring,
            evaluator_llm=None, selfcheck=None,
            enable_tolerance=False, enable_flesch=False,
            step_traces=all_step_traces,
            nutrichat_name=nchat_name,
        )

    scoring_tasks = {}
    bd = None
    with ThreadPoolExecutor(max_workers=3) as pool:
        if enable_eval_llm:
            scoring_tasks[pool.submit(_score_llm)] = "EvaluatorLLM"
        if enable_selfcheck and sc_map:
            scoring_tasks[pool.submit(_score_selfcheck)] = "SelfCheckGPT"
        if all_step_traces:
            scoring_tasks[pool.submit(_score_faithfulness)] = "Faithfulness"

        per_query_data = None
        for f in as_completed(scoring_tasks):
            task_name = scoring_tasks[f]
            result = f.result()
            if not result:
                continue

            # Unpack per_query_data from EvaluatorLLM (the only task that returns it)
            if task_name == "EvaluatorLLM" and detailed and isinstance(result, tuple):
                result, per_query_data = result

            _merge_eval_reports(reports, result)

            tqdm.write(f"\n  >>> [{task_name}] completed")

            # Print appropriate results table per task
            if task_name == "EvaluatorLLM":
                if detailed and per_query_data:
                    bd = compute_breakdown(per_query_data, all_methods, reports=reports)
                    print_breakdown_tables(bd)
                print_report_table(reports, cost_data=usage_by_model)
                # Statistical significance
                nutrichat_ref = next((n for n in reports if n.startswith("NutriCHAT")), None)
                if nutrichat_ref and len(reports) > 1:
                    from src.evaluation.evaluator import print_significance_table, compute_significance_test
                    comparisons = {}
                    for name, report in reports.items():
                        if name != nutrichat_ref:
                            comparisons[name] = compute_significance_test(reports[nutrichat_ref], report)
                    print_significance_table(nutrichat_ref, comparisons)
            elif task_name == "SelfCheckGPT":
                has_halluc = any(r.hallucination_scores for r in reports.values())
                if has_halluc:
                    for line in format_report_table(
                        {m: reports[m] for m in all_methods if m in reports},
                        cost_data=usage_by_model,
                    ).split("\n"):
                        tqdm.write(line)
            elif task_name == "Faithfulness":
                has_faith = any(r.faithfulness for r in reports.values())
                if has_faith:
                    for line in format_report_table(
                        {m: reports[m] for m in all_methods if m in reports},
                        cost_data=usage_by_model,
                    ).split("\n"):
                        tqdm.write(line)

    # ── Final: Cost & save ──
    total_elapsed = time.time() - _t0
    print_cost_table(usage_by_model)
    save_report(reports, log_dir, cost_data=usage_by_model, queries=queries, breakdown=bd,
                total_time_sec=total_elapsed)


def _run_batch_scoring(log_dir: str):
    """Load saved responses and batch-score all metrics (EvaluatorLLM + Flesch + tolerance)."""
    import json

    responses_path = os.path.join(log_dir, "responses.json")
    if not os.path.exists(responses_path):
        print(f"[Error] No responses.json found at {responses_path}")
        return

    with open(responses_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = data["queries"]
    all_responses = data["responses"]
    if not all_responses:
        print("[Error] No responses found")
        return

    print(f"\n{'='*60}")
    print(f"  Batch Scoring: {len(queries)} queries, {len(all_responses)} model(s)")
    print(f"{'='*60}")

    from src.evaluation.evaluator import (
        EvalReport, run_evaluation_pipeline, print_report_table, save_report,
    )
    from src.evaluation.evaluator_llm import EvaluatorLLM

    EvaluatorLLM._cache_path = os.path.join(log_dir, "evaluator_cache.json")
    ev_llm = EvaluatorLLM()

    methods = list(all_responses.keys())
    nchat_key = next((n for n in methods if n.startswith("NutriCHAT")), methods[0])
    nutrichat_resp = all_responses[nchat_key]
    baseline_resp = {m: all_responses[m] for m in methods if m != nchat_key}

    # SelfCheck (skip with SCORE_SKIP_SELFCHECK=1 for quick ablation scoring)
    _skip_sc = os.environ.get("SCORE_SKIP_SELFCHECK", "0") == "1"
    sc_map = None
    if not _skip_sc:
        from src.evaluation.selfcheck import SelfCheckGPT
        _bl_by_name = {}
        try:
            from src.config import Config as _cfg
            for b in _cfg.BASELINE_MODELS:
                _bl_by_name[b["name"]] = b
        except Exception:
            pass
        sc_map = {}
        for method in all_responses:
            if method.startswith("NutriCHAT"):
                sc_map[method] = SelfCheckGPT(device=Config.NLI_DEVICE).compute_score
            else:
                bl_entry = _bl_by_name.get(method, {})
                sc_map[method] = SelfCheckGPT(
                    device=Config.NLI_DEVICE,
                    sample_model=bl_entry.get("model"),
                    sample_base_url=bl_entry.get("base_url"),
                    sample_api_key=os.getenv(bl_entry.get("api_key_env", ""), ""),
                    sample_extra_body=bl_entry.get("extra_body"),
                ).compute_score

    reports, per_query_data = run_evaluation_pipeline(
        queries=[{**q, "query": q["query"]} for q in queries],
        nutrichat_responses=nutrichat_resp,
        baseline_responses=baseline_resp,
        evaluator_llm=ev_llm.score_responses,
        evaluator_llm_bulk=ev_llm.score_bulk,
        selfcheck=sc_map,
        n_selfcheck_values=Config.EVAL_N_SELFCHECK_VALUES,
        enable_tolerance=True,
        enable_flesch=True,
        nutrichat_name=nchat_key,
        return_per_query=True,
    )

    print_report_table(reports)
    from src.evaluation.evaluator import compute_breakdown
    breakdown = compute_breakdown(per_query_data, list(all_responses.keys()), reports)
    save_report(reports, log_dir, queries=queries, breakdown=breakdown)


def _show_latest_report(json_mode: bool = False):
    """Print the most recent log directory's report table.

    Args:
        json_mode: If True, print report.json to stdout directly.
    """
    log_dirs = sorted(
        [d for d in os.listdir(_LOG_DIR)
         if os.path.isdir(os.path.join(_LOG_DIR, d))],
        reverse=True,
    )
    if not log_dirs:
        msg = "No log directories found."
        sys.exit(1) if json_mode else print(msg)
        return

    latest = os.path.join(_LOG_DIR, log_dirs[0])
    fpath = os.path.join(latest, "report.json")
    if not os.path.exists(fpath):
        msg = f"No report.json in {log_dirs[0]}"
        sys.exit(1) if json_mode else print(msg)
        return

    if json_mode:
        import json as _json
        with open(fpath, encoding="utf-8") as f:
            data = _json.load(f)
        _json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
        return

    # Human-readable mode
    print(f"Latest: {log_dirs[0]}")

    for fname in ["report_table.md", "report.json"]:
        fpath = os.path.join(latest, fname)
        if not os.path.exists(fpath):
            continue
        print(f"\n--- {fname} ---")
        if fname.endswith(".txt"):
            with open(fpath, encoding="utf-8") as f:
                print(f.read())
        elif fname.endswith(".json"):
            import json as _json
            with open(fpath, encoding="utf-8") as f:
                data = _json.load(f)
            # Single-line per model for quick scanning
            header = f"  {'Method':<30} {'Overall':>8} {'C':>6} {'P':>6} {'D':>6} {'Halluc':>7}  {'FRES':>5}"
            sep = "  " + "-" * (len(header) - 2)
            print(sep)
            print(header)
            print(sep)
            for method, metrics in data.items():
                if method.startswith("_"):
                    continue
                overall = metrics.get("overall_weighted_score")
                c = (metrics.get("correctness") or {}).get("mean")
                p = (metrics.get("precision") or {}).get("mean")
                d = (metrics.get("scientific_depth") or {}).get("mean")
                hall = (metrics.get("hallucination") or {}).get("mean")
                f_res = (metrics.get("flesch") or {}).get("mean")
                def _fmt(v):
                    if v is None: return "  N/A"
                    return f"{v:6.2f}"
                print(f"  {method:<30} {_fmt(overall):>8} {_fmt(c):>6} {_fmt(p)} {_fmt(d)} {_fmt(hall)} {_fmt(f_res):>7}")
            print(sep)


def _validate_all_configs():
    """Validate all config files, resources, and environment for running experiments."""

    # ── Helpers ────────────────────────────────────────────────
    def _yaml_load(path):
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _print(name, status, msg=""):
        icons = {"ok": " [OK]  ", "fail": "[FAIL] ", "warn": "[WARN] ", "skip": "[SKIP] "}
        icon = icons.get(status, "       ")
        if status == "ok":
            print(f"  {icon}{name}")
        else:
            print(f"  {icon}{name}{'  ' + msg if msg else ''}")

    _project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    _config_dir = os.path.join(_project_root, "config")
    all_ok = True
    _has_warn = False

    # ── 1. YAML 解析 + Schema 校验 ─────────────────────────────
    print("\n  ── Config files ────────────────────────────────────")
    import yaml
    from pydantic import ValidationError
    from src.schemas import validate_config

    schema_map = {
        "settings": "settings.yaml",
        "models": "models.yaml",
        "embeddings": "embeddings.yaml",
        "evaluate": "evaluate.yaml",
        "ablation": "ablation.yaml",
        "pricing": "pricing.yaml",
    }
    for schema, filename in schema_map.items():
        path = os.path.join(_config_dir, filename)
        if not os.path.exists(path):
            _print(filename, "skip", "not found")
            continue
        try:
            data = _yaml_load(path)
            validate_config(schema, data)
            _print(filename, "ok")
        except ValidationError as e:
            all_ok = False
            _print(filename, "fail")
            for err in e.errors():
                loc = " → ".join(str(x) for x in err.get("loc", ()))
                print(f"         {loc}: {err.get('msg', '')}")
        except yaml.YAMLError as e:
            all_ok = False
            _print(filename, "fail", f"YAML parse error: {e}")

    # config.local.yaml — parse only, no schema
    local_path = os.path.join(_config_dir, "config.local.yaml")
    local_data = {}
    if os.path.exists(local_path):
        try:
            local_data = _yaml_load(local_path)
            _print("config.local.yaml", "ok")
        except yaml.YAMLError as e:
            all_ok = False
            _print("config.local.yaml", "fail", f"YAML parse error — ALL local overrides ignored!\n         {e}")
    else:
        _print("config.local.yaml", "skip")

    # ── 2. API Keys ────────────────────────────────────────────
    print("\n  ── API Keys ─────────────────────────────────────────")
    from dotenv import load_dotenv
    load_dotenv()
    all_providers = {}
    try:
        models_data = _yaml_load(os.path.join(_config_dir, "models.yaml"))
        for p in models_data.get("providers", []):
            all_providers[p["provider"]] = p.get("api_key_env", "")
    except Exception:
        all_providers = {}

    # Collect which providers are actually in use
    active_providers = set()
    try:
        if local_data.get("nutrichat"):
            active_providers.add(local_data["nutrichat"].get("provider", ""))
        if local_data.get("evaluator"):
            active_providers.add(local_data["evaluator"].get("provider", ""))
    except Exception:
        pass
    if not active_providers:
        try:
            active_providers.add(models_data.get("nutrichat", {}).get("provider", ""))
            active_providers.add(models_data.get("evaluator", {}).get("provider", ""))
        except Exception:
            pass
    # Also include baselines from evaluate.yaml + local
    try:
        eval_data = _yaml_load(os.path.join(_config_dir, "evaluate.yaml"))
        if local_data:
            import copy
            from src.config import deep_merge
            eval_data = deep_merge(eval_data, local_data)
        if eval_data.get("baselines"):
            for m in models_data.get("baselines", {}).get("models", []):
                if m.get("name") in eval_data["baselines"]:
                    active_providers.add(m.get("provider", ""))
    except Exception:
        pass

    for prov in ["siliconflow", "deepseek", "aliyun", "volcano", "tencent"]:
        env_var = all_providers.get(prov, f"{prov.upper()}_API_KEY")
        val = os.getenv(env_var, "")
        if prov in active_providers:
            if val:
                _print(f"{env_var}", "ok", f"({prov})")
            else:
                all_ok = False
                _print(f"{env_var}", "fail", f"({prov}) — required but not set!")
        elif val:
            _print(f"{env_var}", "ok", f"({prov}, configured)")
        else:
            _print(f"{env_var}", "skip", f"({prov})")

    # ── 3. Data files ──────────────────────────────────────────
    print("\n  ── Data files ───────────────────────────────────────")
    data_checks = [
        ("Feed ingredient bank", "corpus/shared/feed_ingredient_bank.csv"),
        ("Amino acids", "corpus/shared/feed_ingredient_amino_acids.csv"),
        ("Digestibility", "corpus/shared/feed_ingredient_digestibility.csv"),
        ("Definitions", "corpus/shared/definitions.csv"),
        ("Nutrient requirements (broiler)", "corpus/broiler/nutrient_requirements.csv"),
        ("Performance management (broiler)", "corpus/broiler/performance_management.csv"),
        ("Queries (broiler)", "queries/broiler/evaluation_queries.json"),
        ("Queries (layer)", "queries/layer/evaluation_queries.json"),
        ("Chinese queries", "queries/broiler/evaluation_queries_chinese.json"),
    ]
    for name, rel_path in data_checks:
        full = os.path.join(_project_root, rel_path)
        if os.path.exists(full):
            sz = os.path.getsize(full)
            _print(name, "ok", f"({sz:,} bytes)")
        else:
            _print(name, "fail", f"NOT FOUND: {rel_path}")
            all_ok = False

    # ── 4. Model files ─────────────────────────────────────────
    print("\n  ── Model files ──────────────────────────────────────")
    from src.config import Config
    emb_path = Config.LOCAL_EMBEDDING_PATH
    if os.path.isdir(emb_path):
        _print("Qwen3 Embedding", "ok", f"({emb_path})")
    else:
        _print("Qwen3 Embedding", "fail", f"NOT FOUND: {emb_path}")
        _has_warn = True

    nli_path = Config.NLI_MODEL_PATH
    if os.path.isdir(nli_path):
        _print("NLI (DeBERTa-v3)", "ok", f"({nli_path})")
    else:
        _print("NLI (DeBERTa-v3)", "fail", f"NOT FOUND: {nli_path}")
        _has_warn = True

    # ── 5. Vector DB ────────────────────────────────────────────
    print("\n  ── Vector database ──────────────────────────────────")
    for species in ["broiler", "layer"]:
        db = os.path.join(_project_root, "vector_db", f"chroma_{species}", "chroma.sqlite3")
        if os.path.exists(db):
            _print(f"chroma_{species}", "ok")
        else:
            _print(f"chroma_{species}", "warn", "not built — will auto-build on first query")
            _has_warn = True

    # ── 6. Pricing coverage ─────────────────────────────────────
    print("\n  ── Pricing ──────────────────────────────────────────")
    from src.utils.cost_tracker import _resolve_pricing_key
    missing_pricing = []
    try:
        models_data = _yaml_load(os.path.join(_config_dir, "models.yaml"))
        # NutriAgent base model
        base_model = Config.LLM_MODEL
        if _resolve_pricing_key(base_model) is None:
            missing_pricing.append(f"nutrichat: {base_model}")
        # Baseline models
        for m in models_data.get("baselines", {}).get("models", []):
            key = f"{m['provider']}::{m['model']}"
            if _resolve_pricing_key(key) is None:
                missing_pricing.append(f"baseline {m['name']}: {key}")
    except Exception:
        pass

    if missing_pricing:
        _has_warn = True
        _print(f"{len(missing_pricing)} model(s) missing from pricing.yaml", "warn")
        for m in missing_pricing:
            print(f"         {m}")
    else:
        _print("All models covered", "ok")

    # ── 7. Active overrides ─────────────────────────────────────
    print("\n  ── Local overrides (config.local.yaml) ──────────────")
    if local_data:
        _dump_overrides(local_data, indent=7)
    else:
        print("         (none)")

    # ── 8. Effective config summary ─────────────────────────────
    print(f"  ── Effective config ──────────────────────────────")
    print(f"     Base model:     {Config.LLM_MODEL}")
    print(f"     Evaluator:      {Config.EVALUATOR_MODEL}")
    print(f"     Sample ratio:   {Config.EVAL_SAMPLE_RATIO or 'full'}")
    print(f"     N SelfCheck:    {Config.EVAL_N_SELFCHECK_VALUES}")
    print(f"     Hall. ratio:    {Config.HALLUCINATION_RATIO}")
    print(f"  ──────────────────────────────────────────────────")

    print()
    if all_ok:
        print("配置文件、资源文件、API key 校验全部通过。")
    elif _has_warn:
        print("存在警告，但不影响运行。建议修复后重试。")
    else:
        print("存在错误，请修复后重试。")
    return all_ok


def _dump_overrides(data: dict, indent: int = 4):
    """Pretty-print active config.local.yaml overrides."""
    prefix = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            for k, v in value.items():
                if isinstance(v, dict):
                    print(f"{prefix}  {k}: ...")
                else:
                    print(f"{prefix}  {k}: {v}")
        elif isinstance(value, list):
            print(f"{prefix}{key}: {[str(x) for x in value]}")
        else:
            print(f"{prefix}{key}: {value}")


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="NutriCHAT - Poultry Nutrition AI Agent")
    parser.add_argument("--query", "-q", type=str, help="Single query to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--build-db", action="store_true", help="Build/rebuild PoultryDB")
    parser.add_argument("--force-db", action="store_true", help="Force rebuild PoultryDB")
    parser.add_argument("--evaluate", "-e", action="store_true", help="Run NutriCHAT evaluation (EvaluatorLLM + metrics)")
    parser.add_argument("--compare", "-c", action="store_true", help="Multi-model comparison (NutriCHAT + baselines)")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study (Section 3.6 framework)")
    parser.add_argument("--ablation-dim", type=str, default=None,
                        choices=["architecture", "specialist", "router", "rag",
                                 "synthesis", "llm_backbone", "language",
                                 "architecture_factorial", "synthesis_factorial"],
                        help="Specific ablation dimension (default: all)")
    parser.add_argument("--ablation-to-compare", action="store_true",
                        help="End-to-end: ablation → auto-tune → compare baselines")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="With --ablation-to-compare: reuse latest ablation results")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --ablation-to-compare: tune only, skip compare")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens for response")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG component")
    parser.add_argument("--agent-mode", choices=["single", "multi", "both"], default=None,
                        help="Agent architecture: single (ReAct), multi (Router+Specialist+Synthesis), both (compare)")
    parser.add_argument("--query-set", choices=["all", "chinese", "western"], default=None,
                        help="Query set to evaluate: all, chinese (CN-*), western (non-CN-*)")
    parser.add_argument("--query-file", type=str, default=None,
                        help="Path to a custom query JSON file (overrides the default species queries file)")
    parser.add_argument("--breakdown", "-b", action="store_true",
                        help="Print detailed breakdown by query type & tool category (with --evaluate)")
    parser.add_argument("--score-from", type=str, default=None,
                        help="Path to a log dir with responses.json to run batch scoring on saved results")
    parser.add_argument("--baselines-only", action="store_true",
                        help="Skip NutriCHAT, only run baseline models (with --compare)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick dev mode: minimal sampling + skip SelfCheck")
    parser.add_argument("--latest", action="store_true",
                        help="Print the most recent log directory's report")
    parser.add_argument("--json", action="store_true",
                        help="Output --latest report as JSON to stdout")
    parser.add_argument("--from-config", type=str, default=None, metavar="LOG_DIR",
                        help="Load config from a previous log dir's config.yaml and re-run --evaluate")
    parser.add_argument("--check-config", action="store_true",
                        help="Validate all YAML config files and exit")

    args = parser.parse_args()

    # ── Config validation (--check-config) ──────────────────────
    if args.check_config:
        _validate_all_configs()
        return

    # ── Pricing coverage warning (once at startup) ─────────────
    try:
        Config._check_pricing()
    except Exception:
        pass  # non-critical

    # ── Load evaluate config (config/evaluate.yaml) when needed ──
    _EVAL_CONFIG_PATH = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "evaluate.yaml")
    )

    def _load_eval_cfg() -> dict:
        try:
            import yaml
            with open(_EVAL_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            from src.schemas import validate_config
            from src.config import inherit_experiment_config, deep_merge, _LOCAL_OVERRIDE

            # Apply config.local.yaml overrides (already loaded by Config module)
            if _LOCAL_OVERRIDE:
                _EVAL_KEYS = {"evaluation", "agent_mode", "query_set", "baselines",
                              "breakdown", "scoring", "categories", "thinking_baselines"}
                relevant_overrides = {k: v for k, v in _LOCAL_OVERRIDE.items() if k in _EVAL_KEYS}
                data = deep_merge(data, relevant_overrides)
                # scoring_evaluate 合并进 scoring（与 config.py 保持一致）
                sc_ev = _LOCAL_OVERRIDE.get("scoring_evaluate")
                if sc_ev:
                    data["scoring"] = deep_merge(
                        data.get("scoring", {}), sc_ev
                    )

            from pydantic import ValidationError
            try:
                cfg = validate_config("evaluate", data)
            except ValidationError as e:
                logger = get_logger(__name__)
                logger.warning("Config validation failed (check config.local.yaml): %s", e)
                return {}
            return inherit_experiment_config(cfg, "evaluate")
        except Exception:
            return {}

    eval_cfg = _load_eval_cfg() if (args.evaluate or args.compare) else {}

    # CLI args override evaluate.yaml
    agent_mode = args.agent_mode or eval_cfg.get("agent_mode", "single")
    _query_set = args.query_set or eval_cfg.get("query_set", "all")
    _breakdown = args.breakdown or eval_cfg.get("breakdown", False)
    _baselines_only = args.baselines_only
    _eval_baselines = eval_cfg.get("baselines") if eval_cfg else None
    # Auto-include thinking_baselines in the comparison
    _thinking_bl = eval_cfg.get("thinking_baselines")
    if _thinking_bl:
        if _eval_baselines is None:
            _eval_baselines = list(_thinking_bl)
        else:
            for m in _thinking_bl:
                if m not in _eval_baselines:
                    _eval_baselines.append(m)
    _categories = eval_cfg.get("categories") if eval_cfg else None
    _scoring = eval_cfg.get("scoring", {}) if eval_cfg else {}
    # --compare 可单独覆盖评分开关（config.local.yaml → scoring_compare 段）
    if args.compare and eval_cfg:
        from src.config import deep_merge
        _scoring_cmp = eval_cfg.get("scoring_compare")
        if _scoring_cmp:
            _merged = deep_merge(dict(_scoring), _scoring_cmp)
            from src.schemas import ScoringSettings
            ScoringSettings(**_merged)  # 校验：reject unknown fields
            _scoring = _merged
    # 每个命令可独立的 sample_ratio（从 scoring 段读取）
    _sample_ratio = _scoring.get("sample_ratio") if isinstance(_scoring, dict) else None

    # --latest: show most recent log and exit
    if args.latest:
        _show_latest_report(json_mode=args.json)
        return

    # --from-config: reload config from a previous log dir
    if args.from_config:
        import yaml
        cfg_path = os.path.join(args.from_config, "config.yaml")
        if not os.path.exists(cfg_path):
            print(f"Error: {cfg_path} not found")
            return
        with open(cfg_path, "r", encoding="utf-8") as f:
            saved = yaml.safe_load(f) or {}

        # Detect run mode from saved config
        _is_ablation = "ablation_merged" in saved or "enabled_dimensions" in saved.get("config", {})

        if _is_ablation:
            args.ablation = True
            if "scoring" in saved:
                _scoring = saved["scoring"]
            if "config" in saved:
                _sr_from = saved["config"].get("eval_sample_ratio") or saved["config"].get("sample_ratio")
                if _sr_from is not None:
                    Config.EVAL_SAMPLE_RATIO = _sr_from
                _qs = saved["config"].get("query_set")
                if _qs:
                    _query_set = _qs
                _dim_from = saved["config"].get("ablation_dim")
                args.ablation_dim = _dim_from  # None = all dimensions
                # Restore agent_mode if present
                _am = saved["config"].get("agent_mode")
                if _am:
                    agent_mode = _am
        else:
            args.evaluate = True
            if "scoring" in saved:
                _scoring = saved["scoring"]
            elif "config" in saved:
                sc_keys = ("evaluator_llm", "selfcheck", "tolerance", "flesch",
                           "hallucination_ratio", "n_selfcheck_values", "shuffle_responses")
                _scoring = {k: saved["config"][k] for k in sc_keys if k in saved["config"]}
            for key in ("agent_mode", "query_set", "categories"):
                if key in saved.get("config", {}):
                    val = saved["config"][key]
                    if key == "agent_mode":
                        agent_mode = val
                    elif key == "query_set":
                        _query_set = val
                    elif key == "categories":
                        _categories = val
            # Restore sample_ratio
            _sr_from = saved.get("config", {}).get("eval_sample_ratio")
            if _sr_from is not None:
                Config.EVAL_SAMPLE_RATIO = _sr_from

    from src.config import Config

    # --quick: dev mode, 使用 scoring.*.quick 段配置（config.local.yaml 可自定义）
    if args.quick:
        from src.config import deep_merge
        _quick_cfg = _scoring.get("quick") or {}
        _scoring = deep_merge(dict(_scoring), _quick_cfg)
        # sample_ratio 优先从 quick 段取，回退到合并后的 scoring 顶层
        _quick_sr = _quick_cfg.get("sample_ratio") or _scoring.get("sample_ratio")
        if _quick_sr is not None:
            Config.EVAL_SAMPLE_RATIO = _quick_sr
            _sample_ratio = _quick_sr

    # ── Config validation for experiment commands ──────────────
    if args.evaluate or args.compare or args.ablation or args.ablation_dim or args.score_from or args.ablation_to_compare:
        if not _validate_all_configs():
            print("配置校验失败，请修复后重试。")
            return

    if args.build_db or args.force_db:
        build_database(force=args.force_db)
        if not args.query and not args.interactive and not args.evaluate and not args.compare:
            return

    if agent_mode == "both":
        engine_mode = "multi"
    else:
        engine_mode = agent_mode

    engine = setup_engine(
        verbose=args.verbose,
        max_tokens=args.max_tokens,
        rag_enabled=not args.no_rag,
        mode=engine_mode,
    )

    if not args.build_db and not args.force_db:
        db_path = Config.CHROMA_PERSIST_DIR
        if not os.path.exists(os.path.join(db_path, "chroma.sqlite3")):
            logger = get_logger(__name__)
            logger.info("PoultryDB not found at %s, building...", db_path)
            try:
                engine.build_vector_db()
            except Exception as e:
                logger.error("Could not build PoultryDB: %s", e)
                logger.error("Run 'python main.py --build-db' to retry.")

    if args.query:
        process_query(engine, args.query)
    elif args.interactive:
        interactive_mode(engine)
    elif args.compare:
        if _sample_ratio is not None:
            Config.EVAL_SAMPLE_RATIO = _sample_ratio
        run_comparison(engine, queries_file=args.query_file, query_set=_query_set, detailed=_breakdown,
                       categories=_categories, baselines=_eval_baselines,
                       skip_nutrichat=_baselines_only, scoring=_scoring,
                       agent_mode=agent_mode, quick=args.quick)
    elif args.evaluate:
        if _sample_ratio is not None:
            Config.EVAL_SAMPLE_RATIO = _sample_ratio
        run_evaluation(engine, queries_file=args.query_file, query_set=_query_set, agent_mode=agent_mode,
                       detailed=_breakdown, categories=_categories, scoring=_scoring, quick=args.quick)
    elif args.ablation_to_compare:
        from src.ablation_to_compare import run_ablation_to_compare
        run_ablation_to_compare(skip_ablation=args.skip_ablation, dry_run=args.dry_run, quick=args.quick)

    elif args.ablation or args.ablation_dim:
        from src.evaluation.ablation import run_ablation
        from src.config import _LOCAL_OVERRIDE
        # Ablation 也支持 --quick（当时 _scoring 为空未能注入 sample_ratio，在此补上）
        if args.quick and _LOCAL_OVERRIDE:
            _quick_sr = (_LOCAL_OVERRIDE.get("scoring_ablation") or {}).get("quick", {}).get("sample_ratio")
            if _quick_sr is not None:
                Config.EVAL_SAMPLE_RATIO = _quick_sr
        _ablation_qs = args.query_set or _query_set
        run_ablation(dim=args.ablation_dim, verbose=args.verbose, query_set=_ablation_qs,
                     sample_ratio=Config.EVAL_SAMPLE_RATIO)
    elif args.score_from:
        _run_batch_scoring(args.score_from)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
