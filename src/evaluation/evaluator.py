"""NutriCHAT 评估框架 — 整合全部四种评估方法。

四种评估方法：
  ① EvaluatorLLM — LLM 自动评分（correctness / precision / depth）
  ② 容差精度 — 数值查询的 ±2%/±5% 准确率
  ③ Flesch Reading Ease — 可读性
  ④ SelfCheckGPT-NLI — 跨样本不一致性检测（cross-sample inconsistency）
"""
import json
import os
import re
import statistics
from typing import Callable, Dict, List, Optional, Tuple

from src.config import Config


class EvalReport:
    """单个评估运行的报告容器。"""

    def __init__(self, method_name: str):
        self.method_name = method_name
        self.correctness: List[Optional[float]] = []
        self.precision: List[Optional[float]] = []
        self.scientific_depth: List[Optional[float]] = []
        self.flesch_scores: List[float] = []
        self.hallucination_scores: List[float] = []
        self.hallucination_multi: Dict[int, List[float]] = {}  # N -> [scores]
        self.tolerance_p2: List[Optional[float]] = []  # ±2%
        self.tolerance_p5: List[Optional[float]] = []  # ±5%
        self.faithfulness: List[Optional[float]] = []  # tool-data faithfulness (numeric)
        self.semantic_faithfulness: List[Optional[float]] = []  # NLI-based semantic faithfulness

    def add(self, c=None, p=None, d=None, flesch=None, halluc=None,
            tol2=None, tol5=None, halluc_multi: Optional[Dict[int, float]] = None,
            faithfulness=None, semantic_faithfulness=None):
        if c is not None:
            self.correctness.append(c)
        if p is not None:
            self.precision.append(p)
        if d is not None:
            self.scientific_depth.append(d)
        if flesch is not None:
            self.flesch_scores.append(flesch)
        if halluc is not None:
            self.hallucination_scores.append(halluc)
        if halluc_multi:
            for n, score in halluc_multi.items():
                if score is not None:
                    self.hallucination_multi.setdefault(n, []).append(score)
        if tol2 is not None:
            self.tolerance_p2.append(tol2)
        if tol5 is not None:
            self.tolerance_p5.append(tol5)
        if faithfulness is not None:
            self.faithfulness.append(faithfulness)
        if semantic_faithfulness is not None:
            self.semantic_faithfulness.append(semantic_faithfulness)

    def _stats(self, vals: List[float]) -> dict:
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0,
            "n": len(vals),
        }

    def summary(self) -> dict:
        # Overall weighted score（三指标等权）
        c_mean = self._stats(self.correctness)["mean"] or 0
        p_mean = self._stats(self.precision)["mean"]
        d_mean = self._stats(self.scientific_depth)["mean"] or 0
        if p_mean is not None:
            overall = (c_mean + p_mean + d_mean) / 3
        else:
            overall = (c_mean + d_mean) / 2

        result = {
            "method": self.method_name,
            "total_queries": max(
                len(self.correctness), len(self.flesch_scores),
                len(self.hallucination_scores) if self.hallucination_scores else 0,
                *(len(v) for v in self.hallucination_multi.values()),
            ),
            "correctness": self._stats(self.correctness),
            "precision": self._stats(self.precision),
            "scientific_depth": self._stats(self.scientific_depth),
            "overall_weighted_score": overall,
            "flesch_reading_ease": self._stats(self.flesch_scores),
            "inconsistency": self._stats(self.hallucination_scores),
            "faithfulness": self._stats([v for v in self.faithfulness if v is not None]) if self.faithfulness else {"mean": None, "std": None, "n": 0},
            "faithfulness_semantic": self._stats([v for v in self.semantic_faithfulness if v is not None]) if self.semantic_faithfulness else {"mean": None, "std": None, "n": 0},
            "tolerance_p2": self._stats([v for v in self.tolerance_p2 if v is not None]) if self.tolerance_p2 else {"mean": None, "std": None, "n": 0},
            "tolerance_p5": self._stats([v for v in self.tolerance_p5 if v is not None]) if self.tolerance_p5 else {"mean": None, "std": None, "n": 0},
        }

        # Add multi-N inconsistency keys (e.g. inconsistency_n5, inconsistency_n10, …)
        for n in sorted(self.hallucination_multi):
            result[f"inconsistency_n{n}"] = self._stats(self.hallucination_multi[n])

        return result


def _has_cjk(text: str) -> bool:
    """Check if text contains CJK (Chinese/Japanese/Korean) characters."""
    return bool(re.search(r"[一-鿿㐀-䶿豈-﫿]", text))


def compute_flesch(text: str) -> Optional[float]:
    """Flesch Reading Ease Score (0–100). Returns None for non-English text."""
    if _has_cjk(text):
        return None
    try:
        from textstat import flesch_reading_ease
        return flesch_reading_ease(text) if len(text.strip()) > 10 else 0.0
    except ImportError:
        words = text.strip().split()
        if not words:
            return 0.0
        sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sents:
            return 0.0
        syllables = sum(_count_syllables(w) for w in words)
        return max(0, min(100, 206.835 - 1.015 * (len(words) / len(sents)) - 84.6 * (syllables / len(words))))


def _count_syllables(word: str) -> int:
    w = word.lower().strip(".,!?;:'\"()-")
    vowels = "aeiouy"
    c = sum(1 for i, ch in enumerate(w) if ch in vowels and (i == 0 or w[i - 1] not in vowels))
    return max(c, 1)


def parse_numeric(response: str) -> List[Tuple[float, bool]]:
    """从响应文本中提取数值。

    Returns:
        [(value, is_range), ...]  — is_range=True 表示是范围（取中点）
    """
    results = []
    text = response.replace(",", "")

    # 优先匹配范围: "20–25%", "10-15%", "between X and Y"
    range_pat = re.findall(r"(?:between\s+)?(\d+\.?\d*)\s*(?:-|–|to|and)\s*(\d+\.?\d*)", text, re.IGNORECASE)
    for lo, hi in range_pat:
        lo, hi = float(lo), float(hi)
        if 0 < lo < hi and hi / lo < 3:  # 合理范围
            results.append(((lo + hi) / 2, True))

    # 单值: 数字 + 可能的单位
    single_pat = re.findall(r"(?<![a-zA-Z])(\d+\.?\d*)\s*(?:\%|g|kg|mg|kcal|mm|N|IU)?", text)
    for v in single_pat:
        v_float = float(v)
        # Skip years (1900-2099) and common non-nutritional integers
        if 1900 <= v_float <= 2099:
            continue
        if 0 < v_float < 1e6:
            results.append((v_float, False))

    return results


def check_tolerance(predicted: float, ground_truth: float, tolerance: float = 0.05) -> bool:
    """Check whether the prediction is within relative error ``tolerance``.

    The implementation uses |p - g| / |g| <= tau. An earlier implementation
    used g in [p(1-tau), p(1+tau)], which is asymmetric and is no longer used.
    """
    if ground_truth == 0:
        return abs(predicted) <= tolerance
    return abs(predicted - ground_truth) / abs(ground_truth) <= tolerance


def evaluate_tolerance(
    response: str, ground_truth: float, tolerance: float = 0.05
) -> Optional[float]:
    """评估单条响应的容差精度。

    Returns: 1.0 (正确), 0.0 (错误), None (无法解析)
    """
    values = parse_numeric(response)
    if not values:
        return None

    # 取最接近 ground truth 的解析值
    best = min(values, key=lambda x: abs(x[0] - ground_truth))
    return 1.0 if check_tolerance(best[0], ground_truth, tolerance) else 0.0


def run_evaluation_pipeline(
    queries: List[dict],
    nutrichat_responses: List[str],
    baseline_responses: Dict[str, List[str]],
    evaluator_llm: Optional[Callable] = None,
    selfcheck: Optional[Callable] = None,
    n_selfcheck_samples: int = 5,
    n_selfcheck_values: Optional[List[int]] = None,
    enable_tolerance: bool = True,
    enable_flesch: bool = True,
    return_per_query: bool = False,
    step_traces: Optional[Dict[str, List[List[dict]]]] = None,
    nutrichat_name: str = "NutriCHAT",
    inconsistency_ratio: float = Config.HALLUCINATION_RATIO,
    evaluator_llm_bulk: Optional[Callable] = None,
) -> Dict[str, EvalReport]:
    """运行完整评估流水线。

    Args:
        queries: [{id, query, expected_value?, category?, type?}, ...]
        nutrichat_responses: NutriCHAT 的响应（与 queries 等长）
        baseline_responses: {baseline_name: [response_per_query, ...]}
        evaluator_llm: 可选，score_responses(query, {name: text}) -> {name: {metric: score}}
        selfcheck: 可选，compute_score(response, query) -> float
        n_selfcheck_samples: SelfCheckGPT 采样数（废弃，用 n_selfcheck_values 代替）
        n_selfcheck_values: 多 N 值列表，如 [5, 10, 20]
        return_per_query: True 则返回 (reports, per_query_data)
        step_traces: 可选，{method_name: [每条的 step 列表]}，用于 Faithfulness 评估
        inconsistency_ratio: SelfCheck 按比例采样 (0.0-1.0)，1.0=全部

    Returns:
        {method_name: EvalReport} 或 (reports, per_query_data)
    """
    reports: Dict[str, EvalReport] = {}
    all_methods = [nutrichat_name] + list(baseline_responses.keys())
    all_responses = {nutrichat_name: nutrichat_responses, **baseline_responses}

    for method in all_methods:
        reports[method] = EvalReport(method)

    # Per-query tracking for breakdown analysis
    per_query_data: List[dict] = [
        {"id": q.get("id", f"q{i}"), "category": q.get("category", ""),
         "type": q.get("type", ""), "expected_value": q.get("expected_value"),
         "flesch": {}, "tolerance_p2": {}, "tolerance_p5": {},
         "llm_scores": {}, "inconsistency": {}}
        for i, q in enumerate(queries)
    ]

    # 1. Flesch Reading Ease（所有方法）
    if enable_flesch:
        for method in all_methods:
            for i, resp in enumerate(all_responses[method]):
                score = compute_flesch(resp)
                reports[method].add(flesch=score)
                if i < len(per_query_data):
                    per_query_data[i]["flesch"][method] = score

    # 2. 容差精度（数值查询）
    if enable_tolerance:
        for i, q in enumerate(queries):
            gt = q.get("expected_value")
            if gt is None:
                continue
            for method in all_methods:
                if i >= len(all_responses[method]):
                    continue
                resp = all_responses[method][i]
                tol2 = evaluate_tolerance(resp, gt, 0.02)
                tol5 = evaluate_tolerance(resp, gt, 0.05)
                reports[method].add(tol2=tol2, tol5=tol5)
                if i < len(per_query_data):
                    per_query_data[i]["tolerance_p2"][method] = tol2
                    per_query_data[i]["tolerance_p5"][method] = tol5

    # 3. EvaluatorLLM（如果提供，优先批量评分，回退到逐条评分）
    if evaluator_llm is not None or evaluator_llm_bulk is not None:
        print("--- EvaluatorLLM Scoring ---")

        if evaluator_llm_bulk is not None:
            # 批量评分路径：每个 method 一次 API 调用评 N 条（batch_size=10）
            for method in all_methods:
                method_responses = all_responses[method]
                if not method_responses:
                    continue
                bulk_scores = evaluator_llm_bulk(queries, method_responses, batch_size=10)
                if bulk_scores:
                    for i, entry in enumerate(bulk_scores):
                        if i >= len(queries):
                            break
                        reports[method].add(
                            c=entry.get("correctness"),
                            p=entry.get("precision"),
                            d=entry.get("scientific_depth"),
                        )
                        if i < len(per_query_data):
                            per_query_data[i]["llm_scores"][method] = {
                                "correctness": entry.get("correctness"),
                                "precision": entry.get("precision"),
                                "scientific_depth": entry.get("scientific_depth"),
                            }
                    print(f"  {method}: {len(bulk_scores)}/{len(queries)} scored (bulk)")
                else:
                    # Bulk 失败，回退到逐条评分
                    print(f"  {method}: bulk failed, falling back to per-query")
                    for i, q in enumerate(queries):
                        batch = {method: method_responses[i]}
                        try:
                            scores = evaluator_llm(q["query"], batch)
                        except Exception:
                            scores = {method: {"correctness": None, "precision": None, "scientific_depth": None}}
                        s = scores.get(method, {})
                        reports[method].add(
                            c=s.get("correctness"), p=s.get("precision"), d=s.get("scientific_depth"),
                        )
                        if i < len(per_query_data):
                            per_query_data[i]["llm_scores"][method] = {
                                "correctness": s.get("correctness"),
                                "precision": s.get("precision"),
                                "scientific_depth": s.get("scientific_depth"),
                            }
        else:
            # 逐条评分路径（原有逻辑）
            from concurrent.futures import ThreadPoolExecutor, as_completed

            n_workers = min(len(queries), 4)

            def _score_one(i_q):
                i, q = i_q
                batch = {m: all_responses[m][i] for m in all_methods if i < len(all_responses[m])}
                try:
                    scores = evaluator_llm(q["query"], batch)
                except Exception as e:
                    scores = {m: {"correctness": None, "precision": None,
                                  "scientific_depth": None} for m in batch}
                return i, scores

            scored = [False] * len(queries)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_score_one, (i, q)) for i, q in enumerate(queries)]
                for f in as_completed(futures):
                    i, scores = f.result()
                    scored[i] = True
                    if scores:
                        for method in all_methods:
                            s = scores.get(method, {})
                            reports[method].add(
                                c=s.get("correctness"),
                                p=s.get("precision"),
                                d=s.get("scientific_depth"),
                            )
                            if i < len(per_query_data):
                                per_query_data[i]["llm_scores"][method] = {
                                    "correctness": s.get("correctness"),
                                    "precision": s.get("precision"),
                                    "scientific_depth": s.get("scientific_depth"),
                                }
                    if sum(scored) % 20 == 0:
                        print(f"  Progress: {sum(scored)}/{len(queries)}")

    # 4. SelfCheckGPT-NLI（支持按方法分配各自的采样模型 + inconsistency_ratio 按比例跳过）
    #    按方法分组并行：每组内查询并行，方法间也并行
    if selfcheck is not None:
        n_vals = n_selfcheck_values or ([n_selfcheck_samples] if n_selfcheck_samples > 0 else None)
        if not n_vals:
            print("SelfCheckGPT: no N values specified, skipping")
        else:
            # Apply inconsistency_ratio: randomly select a subset of queries
            import random
            _hr = max(0.0, min(1.0, inconsistency_ratio))
            n_sc = 0 if _hr <= 0 else max(1, int(round(len(queries) * _hr)))
            sc_indices = sorted(random.sample(range(len(queries)), n_sc)) if n_sc > 0 else []

            # 跳过定义查询：幻觉风险极低（直接从工具检索），节省 SelfCheck 开销
            _defn_indices = {i for i, q in enumerate(queries) if q.get("type") == "definition"}
            if _defn_indices:
                sc_indices = [i for i in sc_indices if i not in _defn_indices]

            if not sc_indices:
                print("SelfCheckGPT: no queries to check, skipping")
                return reports if not return_per_query else (reports, per_query_data)

            if n_sc < len(queries) or _defn_indices:
                print(f"--- SelfCheckGPT-NLI (N={n_vals}, queries={len(sc_indices)}/{len(queries)}, "
                      f"ratio={inconsistency_ratio}, skipped {len(_defn_indices)} definitions) ---")
            else:
                print(f"--- SelfCheckGPT-NLI (N={n_vals}) ---")

            # Resolve sc_map once (invariant across queries)
            sc_map: Dict[str, Callable] = {}
            if isinstance(selfcheck, dict):
                sc_map = selfcheck
            else:
                sc_map = {m: selfcheck for m in all_methods}

            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _score_method_all_queries(method):
                """Score all queries for a single method (query-level parallel)."""
                sc_fn = sc_map.get(method)
                if sc_fn is None:
                    return method, {}
                sc_instance = getattr(sc_fn, "__self__", None)
                has_multi = sc_instance is not None and hasattr(sc_instance, "compute_score_multi_n")

                def _score_one_query(i):
                    if i >= len(all_responses[method]):
                        return i, None, None
                    resp = all_responses[method][i]
                    if not resp or len(resp) <= 20:
                        return i, None, None
                    if has_multi:
                        scores = sc_instance.compute_score_multi_n(resp, queries[i]["query"], n_values=n_vals)
                        return i, scores.get(max(n_vals)), scores
                    else:
                        score = sc_fn(resp, queries[i]["query"], n_samples=n_vals[0])
                        return i, score, None

                results = {}
                q_workers = min(len(sc_indices), 3)
                with ThreadPoolExecutor(max_workers=q_workers) as qpool:
                    q_futures = {qpool.submit(_score_one_query, i): i for i in sc_indices}
                    for f in as_completed(q_futures):
                        i, hall, hall_multi = f.result()
                        if hall is not None:
                            results[i] = (hall, hall_multi)
                return method, results

            # Run all methods in parallel, each with query-level parallelism
            active_methods = [m for m in all_methods if m in sc_map]
            import threading as _sc_lock_mod
            _sc_data_lock = _sc_lock_mod.Lock()
            with ThreadPoolExecutor(max_workers=min(len(active_methods), 4)) as pool:
                futures = {pool.submit(_score_method_all_queries, m): m for m in active_methods}
                for f in as_completed(futures):
                    method, results = f.result()
                    for i, (hall, hall_multi) in results.items():
                        if hall_multi is not None:
                            reports[method].add(halluc=hall, halluc_multi=hall_multi)
                        else:
                            reports[method].add(halluc=hall)
                        if i < len(per_query_data):
                            with _sc_data_lock:
                                 per_query_data[i].setdefault("inconsistency", {})[method] = hall
                    print(f"  SelfCheck {method}: {len(results)} queries scored")

    # 5. Faithfulness — numeric + semantic
    if step_traces is not None:
        from src.evaluation.faithfulness import (
            compute_faithfulness_from_steps,
            compute_semantic_faithfulness,
        )
        print("--- Faithfulness (numeric + semantic NLI) ---")
        for method in all_methods:
            traces = step_traces.get(method)
            if not traces:
                continue
            for i, resp in enumerate(all_responses[method]):
                if i >= len(traces) or not resp:
                    continue
                score = compute_faithfulness_from_steps(resp, traces[i])
                reports[method].add(faithfulness=score)
                # Semantic faithfulness (NLI-based, best-effort)
                try:
                    sem_score = compute_semantic_faithfulness(resp, traces[i])
                    reports[method].add(semantic_faithfulness=sem_score)
                except Exception:
                    pass
        print(f"  Done")

    if return_per_query:
        return reports, per_query_data
    return reports


def format_report_table(reports: Dict[str, EvalReport],
                        cost_data: Optional[Dict[str, dict]] = None) -> str:
    """生成 Markdown 格式的对比表格（可直接渲染）。

    cost_data: 可选，提供后会在 Scored Metrics 表中追加 Avg Time(s) 列。
    """
    lines = []
    lines.append("# Evaluation Results\n")

    # ── Table 1: Correctness / Precision / Depth (with optional Time) ──
    has_time = cost_data is not None
    headers1 = ["Method", "Correctness", "Precision", "Depth", "Overall"]
    if has_time:
        headers1.append("Avg Time(s)")
    lines.append("## Scored Metrics\n")
    lines.append("| " + " | ".join(headers1) + " |")
    lines.append("| " + " | ".join("---" for _ in headers1) + " |")
    for name, r in reports.items():
        s = r.summary()
        c = s["correctness"]
        p = s["precision"]
        d = s["scientific_depth"]
        o = s["overall_weighted_score"]
        c_str = f"{c['mean']:.2f}±{c['std']:.2f} (N={c['n']})" if c["mean"] is not None else "N/A"
        p_str = f"{p['mean']:.2f}±{p['std']:.2f} (N={p['n']})" if p["mean"] is not None else "N/A"
        d_str = f"{d['mean']:.2f}±{d['std']:.2f} (N={d['n']})" if d["mean"] is not None else "N/A"
        o_str = f"{o:.2f}" if o else "N/A"
        row = f"| {name} | {c_str} | {p_str} | {d_str} | {o_str}"
        if has_time:
            avg_time = _lookup_avg_time(name, cost_data)
            row += f" | {avg_time}"
        row += " |"
        lines.append(row)

    # ── Table 2: FRES / Inconsistency / Faithfulness / Tolerance ──
    has_faith = any(
        r.summary().get("faithfulness", {}).get("mean") is not None
        for r in reports.values()
    )
    has_sem_faith = any(
        r.summary().get("faithfulness_semantic", {}).get("mean") is not None
        for r in reports.values()
    )
    lines.append("\n## Additional Metrics\n")
    headers2 = ["Method", "FRES", "Inconsistency"]
    if has_faith:
        headers2.append("Faithful(Num)")
    if has_sem_faith:
        headers2.append("Faithful(Sem)")
    headers2 += ["Tol±2%", "Tol±5%"]
    lines.append("| " + " | ".join(headers2) + " |")
    lines.append("| " + " | ".join("---" for _ in headers2) + " |")
    for name, r in reports.items():
        s = r.summary()
        fres = s["flesch_reading_ease"]
        hal = s["inconsistency"]
        faith = s.get("faithfulness", {})
        sem_faith = s.get("faithfulness_semantic", {})
        t2 = s["tolerance_p2"]
        t5 = s["tolerance_p5"]
        fres_str = f"{fres['mean']:.2f}±{fres['std']:.2f} (N={fres['n']})" if fres["mean"] is not None else "N/A"
        hal_str = f"{hal['mean']:.4f}±{hal['std']:.4f} (N={hal['n']})" if hal["mean"] is not None else "N/A"
        faith_str = f"{faith['mean']*100:.1f}% (N={faith['n']})" if faith.get("mean") is not None else "N/A"
        sem_faith_str = f"{sem_faith['mean']*100:.1f}% (N={sem_faith['n']})" if sem_faith.get("mean") is not None else "N/A"
        t2_str = f"{t2['mean']*100:.1f}% (N={t2['n']})" if t2["mean"] is not None else "N/A"
        t5_str = f"{t5['mean']*100:.1f}% (N={t5['n']})" if t5["mean"] is not None else "N/A"
        row = f"| {name} | {fres_str} | {hal_str}"
        if has_faith:
            row += f" | {faith_str}"
        if has_sem_faith:
            row += f" | {sem_faith_str}"
        row += f" | {t2_str} | {t5_str} |"
        lines.append(row)

    # Dynamic footnote: explain per-metric N differences (across all methods)
    if reports:
        total = max(
            r.summary().get("total_queries", 0) for r in reports.values()
        )
        diffs = []
        for key, label, reason in [
            ("precision", "Precision", "评分返回None"),
            ("inconsistency", "Inconsistency", "inconsistency_ratio抽样"),
            ("tolerance_p2", "Tolerance", "仅数值查询"),
            ("faithfulness", "Faithful(Num)", "仅调用工具查询"),
            ("faithfulness_semantic", "Faithful(Sem)", "仅调用工具查询"),
        ]:
            ns = set()
            for r in reports.values():
                v = r.summary().get(key, {}) or {}
                n = v.get("n", 0)
                if n > 0:
                    ns.add(n)
            if ns and min(ns) < total:
                if len(ns) == 1:
                    diffs.append(f"{label}={ns.pop()}({reason})")
                else:
                    diffs.append(f"{label}={min(ns)}-{max(ns)}({reason})")
        if diffs:
            lines.append(f"\n> N=样本数（共{total}条查询）。{'；'.join(diffs)}。\n")

    return "\n".join(lines)


def _lookup_avg_time(method_name: str, cost_data: Dict[str, dict]) -> str:
    """从 cost_data 中按方法名模糊匹配查找 Avg Time(s)。"""
    if not cost_data:
        return "N/A"
    # Exact match first
    if method_name in cost_data:
        d = cost_data[method_name]
        calls = d.get("calls", 0)
        total = d.get("total_time_s", 0) or 0
        avg = total / calls if calls else 0
        return f"{avg:.1f}"
    # Fuzzy match: method name may contain model suffix
    for key, d in cost_data.items():
        if key in method_name or method_name in key:
            calls = d.get("calls", 0)
            total = d.get("total_time_s", 0) or 0
            avg = total / calls if calls else 0
            return f"{avg:.1f}"
    return "N/A"


def print_report_table(reports: Dict[str, EvalReport],
                       cost_data: Optional[Dict[str, dict]] = None):
    """打印comparison table。"""
    print(format_report_table(reports, cost_data=cost_data))


_COST_HEADERS = ["Method", "Queries", "Total Tokens", "Avg Tokens",
                  "Avg Time(s)", "Cost/Query($)", "Total Cost($)"]


def _cost_data_rows(usage_by_method: Dict[str, dict]) -> list:
    rows = []
    for method, data in usage_by_method.items():
        calls = data.get("calls", 0)
        total_tok = data.get("total_tokens", 0) or data.get("prompt_tokens", 0) + data.get("completion_tokens", 0)
        total_time = data.get("total_time_s", 0)
        avg_time = total_time / calls if calls else 0
        avg_tok = total_tok // calls if calls else 0
        cpq = data.get("cost_usd", 0) / calls if calls else 0
        rows.append((method, calls, total_tok, avg_tok, avg_time, cpq, data.get("cost_usd", 0)))
    return rows


def _format_cost_table(usage_by_method: Dict[str, dict]) -> str:
    lines = ["## Cost & Inference Time Analysis\n"]
    lines.append("| " + " | ".join(_COST_HEADERS) + " |")
    lines.append("| " + " | ".join("---" for _ in _COST_HEADERS) + " |")
    for method, calls, total_tok, avg_tok, avg_time, cpq, total_cost in _cost_data_rows(usage_by_method):
        lines.append(
            f"| {method} | {calls} | {total_tok:,} | {avg_tok:,} "
            f"| {avg_time:.2f} | {cpq:.6f} | {total_cost:.6f} |"
        )
    return "\n".join(lines)


def print_cost_table(usage_by_method: Dict[str, dict]):
    print(_format_cost_table(usage_by_method))


def _save_cost_table(usage_by_method: Dict[str, dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(_format_cost_table(usage_by_method))


# ── Statistical significance ───────────────────────────────

def compute_significance_test(
    report_a: "EvalReport",
    report_b: "EvalReport",
    alpha: float = 0.05,
) -> dict:
    """Paired t-test between two methods' per-query scores.

    For each metric (correctness, precision, scientific_depth),
    performs a two-tailed paired t-test on the overlapping queries.

    Args:
        report_a: First method's EvalReport (e.g., "NutriCHAT")
        report_b: Second method's EvalReport (e.g., "DeepSeek-V4-Flash")
        alpha: Significance threshold (default 0.05)

    Returns:
        {metric_name: {"t_statistic": float, "p_value": float,
                        "significant": bool, "n": int, "mean_diff": float}}
    """
    from scipy import stats as _stats

    def _paired_test(a_vals, b_vals):
        pairs = [(a, b) for a, b in zip(a_vals, b_vals)
                 if a is not None and b is not None]
        if len(pairs) < 3:
            return {"t_statistic": None, "p_value": None,
                    "significant": None, "n": len(pairs), "mean_diff": None}
        a_clean, b_clean = zip(*pairs)
        diff = [a - b for a, b in zip(a_clean, b_clean)]
        mean_diff = sum(diff) / len(diff)
        t_stat, p_value = _stats.ttest_rel(a_clean, b_clean)
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < alpha),
            "n": len(pairs),
            "mean_diff": float(mean_diff),
        }

    return {
        "correctness": _paired_test(report_a.correctness, report_b.correctness),
        "precision": _paired_test(report_a.precision, report_b.precision),
        "scientific_depth": _paired_test(report_a.scientific_depth, report_b.scientific_depth),
    }


def format_significance_table(
    baseline_method: str,
    comparisons: dict,
) -> str:
    """Format paired t-test results into a readable table.

    Args:
        baseline_method: Name of the reference method (e.g., "NutriCHAT")
        comparisons: {method_name: significance_results_dict}
                     Each value is the output of compute_significance_test()

    Returns:
        Formatted table string.
    """
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append(f"  STATISTICAL SIGNIFICANCE (paired t-test vs {baseline_method})")
    lines.append("=" * 100)
    lines.append(f"  {'Method':<28} {'Correctness':<28} {'Precision':<28} {'Depth':<28}")
    lines.append(f"  {'-'*28} {'-'*28} {'-'*28} {'-'*28}")

    for method, tests in comparisons.items():
        row = f"  {method:<28}"
        for metric in ("correctness", "precision", "scientific_depth"):
            t = tests.get(metric, {})
            if t.get("p_value") is None:
                cell = "N/A"
            else:
                sig = "**" if t["significant"] else "ns"
                p = t["p_value"]
                d = t["mean_diff"]
                cell = f"p={p:.4f} {sig} Δ={d:+.3f}"
            row += f" {cell:<28}"
        lines.append(row)
    lines.append("=" * 100)
    return "\n".join(lines)


def print_significance_table(baseline_method: str, comparisons: dict):
    """Print paired t-test results table."""
    print(format_significance_table(baseline_method, comparisons))


# ── Breakdown analysis ─────────────────────────────────────


def compute_breakdown(
    per_query_data: List[dict],
    methods: List[str],
    reports: Optional[Dict[str, "EvalReport"]] = None,
) -> dict:
    """Compute per-group mean±std for all metrics grouped by query type and tool category.

    Args:
        per_query_data: list from run_evaluation_pipeline(return_per_query=True)
        methods: method names e.g. ["NutriCHAT", "GPT-4o"]

    Returns:
        {"by_type": {type: {method: {metric: {"mean", "std", "n"}}}},
         "by_category": {category: {method: {metric: {"mean", "std", "n"}}}}}
    """
    from collections import defaultdict
    import statistics

    # Inject inconsistency from reports (separate SelfCheck pipeline) into per_query_data
    if reports:
        for method in methods:
            r = reports.get(method)
            if r and r.hallucination_scores:
                # inconsistency_scores are sampled (only some indices have values)
                for i in range(min(len(per_query_data), len(r.hallucination_scores))):
                    if i < len(r.hallucination_scores) and r.hallucination_scores[i] is not None:
                        per_query_data[i].setdefault("inconsistency", {})[method] = r.hallucination_scores[i]

    def _stats(vals):
        clean = [v for v in vals if v is not None]
        if not clean:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": sum(clean) / len(clean),
            "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
            "n": len(clean),
        }

    def _group(key):
        groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for entry in per_query_data:
            g = entry.get(key, "Unknown")
            for method in methods:
                llm = entry.get("llm_scores", {}).get(method, {})
                for metric in ("correctness", "precision", "scientific_depth"):
                    v = llm.get(metric)
                    if v is not None:
                        groups[g][method][metric].append(v)
                fv = entry.get("flesch", {}).get(method)
                if fv is not None:
                    groups[g][method]["flesch"].append(fv)
                hv = entry.get("inconsistency", {}).get(method)
                if hv is not None:
                    groups[g][method]["inconsistency"].append(hv)
        return {g: {m: {mm: _stats(vv) for mm, vv in mmd.items()}
                     for m, mmd in md.items()}
                for g, md in groups.items()}

    return {"by_type": _group("type"), "by_category": _group("category")}


def print_breakdown_tables(breakdown: dict):
    """Print query-type and tool-category breakdown tables."""
    print(format_breakdown_text(breakdown))


def _fmt_metric(d: dict, fmt: str = ".2f") -> str:
    """Format a metric dict {mean, std, n} as 'mean±std' or 'N/A'."""
    if d is None or d.get("mean") is None:
        return "N/A"
    if d.get("n", 0) <= 1:
        return f"{d['mean']:{fmt}}"
    return f"{d['mean']:{fmt}}±{d['std']:{fmt}}"


def format_breakdown_text(breakdown: dict) -> str:
    """Format breakdown tables as a string for saving to file."""
    lines = []
    type_labels = {
        "single-value": "Value-type", "comparison": "Comparison",
        "open-ended": "Open-ended", "definition": "Definition",
        "multi-hop": "Multi-hop", "advanced": "Advanced",
    }

    by_type = breakdown.get("by_type", {})
    if by_type:
        lines.append("\n" + "=" * 80)
        lines.append("  BREAKDOWN BY QUERY TYPE")
        lines.append("=" * 80)
        for method in sorted(set(m for g in by_type.values() for m in g)):
            lines.append(f"\n  [{method}]")
            lines.append(f"  {'Type':<18} {'Correctness':<18} {'Precision':<18} {'Depth':<18} {'FRES':<14} {'Inconsistency':<16}")
            lines.append(f"  {'-'*18} {'-'*18} {'-'*18} {'-'*18} {'-'*14} {'-'*16}")
            for qt in ["single-value", "comparison", "open-ended", "definition", "multi-hop", "advanced"]:
                d = by_type.get(qt, {}).get(method, {})
                if d:
                    c = _fmt_metric(d.get("correctness"))
                    p = _fmt_metric(d.get("precision"))
                    dp = _fmt_metric(d.get("scientific_depth"))
                    fv = _fmt_metric(d.get("flesch"))
                    hv = _fmt_metric(d.get("inconsistency"), ".4f")
                    lines.append(f"  {type_labels.get(qt, qt):<18} {c:<18} {p:<18} {dp:<18} {fv:<14} {hv:<16}")

    by_cat = breakdown.get("by_category", {})
    if by_cat:
        lines.append("\n" + "=" * 80)
        lines.append("  BREAKDOWN BY TOOL CATEGORY")
        lines.append("=" * 80)
        for method in sorted(set(m for g in by_cat.values() for m in g)):
            lines.append(f"\n  [{method}]")
            lines.append(f"  {'Category':<32} {'Correctness':<18} {'Precision':<18} {'Depth':<18} {'FRES':<14} {'Inconsistency':<16}")
            lines.append(f"  {'-'*32} {'-'*18} {'-'*18} {'-'*18} {'-'*14} {'-'*16}")
            for cat in sorted(by_cat):
                d = by_cat[cat].get(method, {})
                if d:
                    c = _fmt_metric(d.get("correctness"))
                    p = _fmt_metric(d.get("precision"))
                    dp = _fmt_metric(d.get("scientific_depth"))
                    fv = _fmt_metric(d.get("flesch"))
                    hv = _fmt_metric(d.get("inconsistency"), ".4f")
                    lines.append(f"  {cat:<32} {c:<18} {p:<18} {dp:<18} {fv:<14} {hv:<16}")
    lines.append("")
    return "\n".join(lines)


def _save_report_txt(reports: Dict[str, EvalReport], path: str,
                     cost_data: Optional[Dict[str, dict]] = None):
    """将报告写入纯文本文件。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(format_report_table(reports, cost_data=cost_data))


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration string."""
    if seconds >= 3600:
        return f"{seconds / 3600:.0f} h {seconds % 3600 / 60:.0f} min"
    elif seconds >= 60:
        return f"{seconds / 60:.0f} min {seconds % 60:.0f} s"
    else:
        return f"{seconds:.0f} s"


def _append_total_time(path: str, total_time_sec: float):
    """Append total execution time footer to existing report_table.md."""
    dur = _format_duration(total_time_sec)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n---\n*Total execution time: {dur}*\n")


def save_raw_responses(queries: List[dict], responses: Dict[str, List[str]], log_dir: str):
    """保存所有模型的原始响应到 JSON。"""
    import datetime
    data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "num_queries": len(queries),
        "models": list(responses.keys()),
        "queries": [
            {
                "id": q.get("id", f"Q{i+1:03d}"),
                "category": q.get("category", ""),
                "type": q.get("type", ""),
                "query": q["query"],
                "expected_value": q.get("expected_value"),
            }
            for i, q in enumerate(queries)
        ],
        "responses": {
            name: texts for name, texts in responses.items()
        },
    }
    path = os.path.join(log_dir, "responses.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  responses.json       原始响应 ({len(queries)} 条 × {len(responses)} 个模型)")


def save_report(reports: Dict[str, EvalReport], filepath: str,
                cost_data: Optional[Dict[str, dict]] = None,
                queries: Optional[List[dict]] = None,
                breakdown: Optional[dict] = None,
                total_time_sec: Optional[float] = None):
    """导出评估结果，含可选的时间/成本数据和 per-query 评分明细。

    total_time_sec: 脚本总执行时间（秒），追加到 report_table.md 底部。
    """
    if os.path.isdir(filepath) or os.path.basename(os.path.dirname(filepath)).startswith("log_"):
        log_dir = filepath if os.path.isdir(filepath) else os.path.dirname(filepath)
        os.makedirs(log_dir, exist_ok=True)

        # 1. report.json — 汇总指标 + per-query 评分 + 时间成本
        json_path = os.path.join(log_dir, "report.json")
        data = {name: r.summary() for name, r in reports.items()}

        # Append per-query scores from EvalReport parallel arrays
        if queries:
            for name, r in reports.items():
                pq_list = []
                for i, q in enumerate(queries):
                    entry = {
                        "id": q.get("id"),
                        "category": q.get("category"),
                        "type": q.get("type"),
                        "expected_value": q.get("expected_value"),
                        "correctness": r.correctness[i] if i < len(r.correctness) else None,
                        "precision": r.precision[i] if i < len(r.precision) else None,
                        "scientific_depth": r.scientific_depth[i] if i < len(r.scientific_depth) else None,
                        "inconsistency": r.hallucination_scores[i] if i < len(r.hallucination_scores) else None,
                        "tolerance_p2": r.tolerance_p2[i] if i < len(r.tolerance_p2) else None,
                        "tolerance_p5": r.tolerance_p5[i] if i < len(r.tolerance_p5) else None,
                        "flesch": r.flesch_scores[i] if i < len(r.flesch_scores) else None,
                        "faithfulness": r.faithfulness[i] if i < len(r.faithfulness) else None,
                        "semantic_faithfulness": r.semantic_faithfulness[i] if i < len(r.semantic_faithfulness) else None,
                    }
                    # Multi-N inconsistency (e.g. N=5, N=10, N=20)
                    for n in sorted(r.hallucination_multi):
                        scores = r.hallucination_multi[n]
                        entry[f"inconsistency_n{n}"] = scores[i] if i < len(scores) else None
                    pq_list.append(entry)
                data[name]["per_query"] = pq_list

        if cost_data:
            data["_cost"] = cost_data
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # 2. report_table.md — 可读表格（Markdown），含运行时间
        txt_path = os.path.join(log_dir, "report_table.md")
        _save_report_txt(reports, txt_path, cost_data=cost_data)
        if total_time_sec is not None:
            _append_total_time(txt_path, total_time_sec)

        # 3. cost_table.md — 成本时间明细
        if cost_data:
            cost_path = os.path.join(log_dir, "cost_table.md")
            _save_cost_table(cost_data, cost_path)

        # 4. breakdown.csv — 按查询类型/工具类别分组（含 std + 全部指标）
        if breakdown:
            import csv
            _BD_METRICS = ["correctness", "precision", "scientific_depth", "flesch", "inconsistency"]
            bd_path = os.path.join(log_dir, "breakdown.csv")
            with open(bd_path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                header = ["group", "method", "category"]
                for m in _BD_METRICS:
                    header += [f"{m}_mean", f"{m}_std", f"{m}_n"]
                w.writerow(header)
                for group_key, group_label in [("by_type", "query_type"), ("by_category", "tool_category")]:
                    data = breakdown.get(group_key, {})
                    for cat in sorted(data):
                        for method in sorted(data[cat]):
                            row = [group_label, method, cat]
                            for m in _BD_METRICS:
                                d = data[cat][method].get(m, {}) or {}
                                row.append(f"{d['mean']:.4f}" if d.get("mean") is not None else "")
                                row.append(f"{d['std']:.4f}" if d.get("std") is not None else "")
                                row.append(d.get("n", ""))
                            w.writerow(row)

        print(f"\n[Evaluator] Results saved to {log_dir}/")
        print(f"  report.json         指标汇总")
        print(f"  report_table.md     可读表格")
        if cost_data:
            print(f"  cost_table.md       成本时间明细")
        if breakdown:
            print(f"  breakdown.csv       查询类型/工具类别分组")
    else:
        data = {name: r.summary() for name, r in reports.items()}
        if queries:
            for name, r in reports.items():
                pq_list = []
                for i, q in enumerate(queries):
                    entry = {
                        "id": q.get("id"), "category": q.get("category"),
                        "type": q.get("type"), "expected_value": q.get("expected_value"),
                        "correctness": r.correctness[i] if i < len(r.correctness) else None,
                        "precision": r.precision[i] if i < len(r.precision) else None,
                        "scientific_depth": r.scientific_depth[i] if i < len(r.scientific_depth) else None,
                        "inconsistency": r.hallucination_scores[i] if i < len(r.hallucination_scores) else None,
                        "tolerance_p2": r.tolerance_p2[i] if i < len(r.tolerance_p2) else None,
                        "tolerance_p5": r.tolerance_p5[i] if i < len(r.tolerance_p5) else None,
                        "flesch": r.flesch_scores[i] if i < len(r.flesch_scores) else None,
                    }
                    for n in sorted(r.hallucination_multi):
                        scores = r.hallucination_multi[n]
                        entry[f"inconsistency_n{n}"] = scores[i] if i < len(scores) else None
                    pq_list.append(entry)
                data[name]["per_query"] = pq_list
        if cost_data:
            data["_cost"] = cost_data
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n[Evaluator] Results saved to {filepath}")
