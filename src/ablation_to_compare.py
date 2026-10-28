"""Ablation → Compare 端到端管道。

流程:
  Phase 1: --ablation（完整消融，决策维 30% 采样，不开 SelfCheck）
  Phase 2: 解析结果 → 自动选配最优参数
  Phase 3: --compare（全量查询 + SelfCheck + 基线对比）

用法:
  python main.py --ablation-to-compare
  python main.py --ablation-to-compare --skip-ablation  # 复用已有消融结果
  python main.py --ablation-to-compare --dry-run        # 只选配不跑对比
"""
import json
import os
import shutil
import time
from typing import Dict, List, Optional, Tuple

from src.config import Config, _LOCAL_OVERRIDE, _PROJECT_ROOT, deep_merge
from src.evaluation.ablation import run_ablation

# ── 决策维度的参数映射 ────────────────────────────────────
# 消融变体名称 → 实际参数值（用于 config.local.yaml 更新）

DECISION_DIMS = {
    "llm_backbone": {
        "variants": {},  # 动态填充：variant_name -> {"model": ..., "provider": ...}
        "param_path": ["nutrichat", "model"],
    },
    "router": {
        "variant_map": {
            "LLM Router": "llm",
            "Keyword Router": "keyword",
            "Broadcast (All)": "broadcast",
        },
        "param_path": ["router_strategy"],
    },
    "synthesis": {
        "variant_map": {
            "With Synthesis": True,
            "Without Synthesis": False,
        },
        "param_path": ["synthesis_enabled"],
    },
    "rag": {
        "variant_map": {
            "Multi-Agent + RAG": True,
            "Multi-Agent (no RAG)": False,
        },
        "param_path": ["rag_enabled"],
    },
}

# ── 综合评分权重 ──────────────────────────────────────────
WEIGHTS = {
    "correctness": 0.40,
    "scientific_depth": 0.30,
    "precision": 0.20,
    "hallucination": -0.10,
}


def _composite_score(metrics: dict) -> float:
    """计算综合得分 = 0.4×correctness + 0.3×depth + 0.2×precision - 0.1×halluc.

    halluc 为 None 时跳过该项，权重归一化。
    """
    c = (metrics.get("correctness") or {}).get("mean") or 0
    d = (metrics.get("scientific_depth") or {}).get("mean") or 0
    p = (metrics.get("precision") or {}).get("mean") or 0
    h = (metrics.get("hallucination_n5") or metrics.get("hallucination") or {}).get("mean")

    if h is None:
        # 没有 halluc 时归一化到 1.0
        w_total = WEIGHTS["correctness"] + WEIGHTS["scientific_depth"] + WEIGHTS["precision"]
        return (c * WEIGHTS["correctness"] + d * WEIGHTS["scientific_depth"] + p * WEIGHTS["precision"]) / w_total
    return c * WEIGHTS["correctness"] + d * WEIGHTS["scientific_depth"] + p * WEIGHTS["precision"] + h * WEIGHTS["hallucination"]


def _load_ablation_report(log_dir: str) -> Optional[dict]:
    """从指定日志目录加载 ablation_report.json。"""
    path = os.path.join(log_dir, "ablation_report.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_latest_ablation_log() -> Optional[str]:
    """找到最近一次消融运行的日志目录。"""
    logs_dir = os.path.join(_PROJECT_ROOT, "logs")
    if not os.path.isdir(logs_dir):
        return None
    candidates = sorted(
        [d for d in os.listdir(logs_dir) if d.startswith("ablation_") and not d.startswith("ablation_compare_")],
        key=lambda d: os.path.getmtime(os.path.join(logs_dir, d)),
    )
    return os.path.join(logs_dir, candidates[-1]) if candidates else None


# ═══════════════════════════════════════════════════════════
# Phase 2: 消融结果解析 + 自动选配
# ═══════════════════════════════════════════════════════════

class AblationParser:
    """解析 ablation_report.json，计算综合得分，选出最优参数。"""

    def __init__(self, report: dict):
        self.report = report

    def _variant_scores(self, dim_key: str) -> List[Tuple[str, float, dict]]:
        """返回维度的变体列表: [(name, composite_score, raw_metrics), ...]"""
        dim_data = self.report.get(dim_key, {})
        results = []
        for vname, metrics in dim_data.items():
            score = _composite_score(metrics)
            results.append((vname, score, metrics))
        # 按综合得分降序排列
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def pick_best(self, dim_key: str) -> Tuple[str, float, dict]:
        """选出指定维度的最优变体。返回 (variant_name, score, metrics)."""
        ranked = self._variant_scores(dim_key)
        if not ranked:
            raise ValueError(f"Dimension '{dim_key}' not found in ablation report")
        return ranked[0]

    def pick_all_best(self) -> dict:
        """选出全部 4 个决策维的最优变体。"""
        result = {}
        for dim_key in ["llm_backbone", "router", "synthesis", "rag"]:
            name, score, metrics = self.pick_best(dim_key)
            result[dim_key] = {
                "name": name,
                "score": score,
                "metrics": {k: v for k, v in metrics.items()
                            if k in ("correctness", "precision", "scientific_depth")},
            }
            h = metrics.get("hallucination_n5") or metrics.get("hallucination")
            if h:
                result[dim_key]["hallucination"] = h.get("mean")
        return result

    def summarize(self, picks: dict) -> str:
        """生成选配摘要（供 Step 4 报告用）。"""
        lines = []
        for dim_key in ["llm_backbone", "router", "synthesis", "rag"]:
            pick = picks[dim_key]
            ranked = self._variant_scores(dim_key)
            lines.append(f"## {dim_key}")
            lines.append("")
            lines.append(f"| Variant | Correctness | Depth | Precision | Halluc | 综合得分 |")
            lines.append(f"|---------|------------|-------|-----------|--------|---------|")
            for vname, score, metrics in ranked:
                c = (metrics.get("correctness") or {}).get("mean")
                d = (metrics.get("scientific_depth") or {}).get("mean")
                p = (metrics.get("precision") or {}).get("mean")
                h = (metrics.get("hallucination_n5") or metrics.get("hallucination") or {}).get("mean")
                h_str = f"{h:.4f}" if h is not None else "N/A"
                marker = " ← **最优**" if vname == pick["name"] else ""
                lines.append(
                    f"| {vname} | {c:.2f} | {d:.2f} | {p:.2f} | {h_str} | {score:.3f}{marker} |"
                )
            lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# Phase 2: 配置更新
# ═══════════════════════════════════════════════════════════

def _resolve_backbone_config(variant_name: str) -> dict:
    """将 llm_backbone 变体名解析为模型的 provider+model 配置。

    搜索顺序：
      1. models.yaml baselines（支持 flat list 和 provider-grouped dict 两种格式）
      2. 从名称推断 provider（"provider/ModelName" 格式）
    """
    from src.config import _model_cfg

    models_section = _model_cfg.get("baselines", {}).get("models", []) if isinstance(_model_cfg, dict) else []

    # Format 1: flat list [{name, model, provider}, ...]
    if isinstance(models_section, list):
        for m in models_section:
            if isinstance(m, dict) and m.get("name") == variant_name:
                return {
                    "model": m.get("model", variant_name),
                    "provider": m.get("provider", "siliconflow"),
                }

    # Format 2: provider-grouped dict {provider: [{name, model}, ...], ...}
    if isinstance(models_section, dict):
        for provider, entries in models_section.items():
            if not isinstance(entries, list):
                continue
            for m in entries:
                if isinstance(m, dict) and m.get("name") == variant_name:
                    return {
                        "model": m.get("model", variant_name),
                        "provider": provider,
                    }

    # Fallback: infer from name
    for prefix in ("siliconflow/", "aliyun/", "volcano/", "tencent/", "deepseek/"):
        if variant_name.startswith(prefix):
            provider = prefix.rstrip("/")
            model = variant_name[len(prefix):]
            return {"model": model, "provider": provider}

    # Unknown format
    return {"model": variant_name, "provider": "siliconflow"}


def _update_config_local(picks: dict, dry_run: bool = False) -> Optional[str]:
    """更新 config.local.yaml 的最优参数。返回备份路径（dry_run 时返回 None）。"""
    local_path = os.path.join(_PROJECT_ROOT, "config", "config.local.yaml")
    if not os.path.exists(local_path):
        print("  [ERROR] config.local.yaml not found")
        return None

    # 读取当前配置
    with open(local_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 备份
    bak_path = local_path + ".bak"
    if not dry_run:
        shutil.copy2(local_path, bak_path)

    # 解析 backbone 配置
    bb_name = picks["llm_backbone"]["name"]
    bb_config = _resolve_backbone_config(bb_name)

    # 构建变更摘要
    changes = [
        ("nutrichat.model", bb_config["model"]),
        ("nutrichat.provider", bb_config["provider"]),
    ]

    # 使用 yaml 安全地修改（通过重写文件）
    import yaml
    try:
        with open(local_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        # 更新 nutrichat
        if "nutrichat" not in config:
            config["nutrichat"] = {}
        config["nutrichat"]["model"] = bb_config["model"]
        config["nutrichat"]["provider"] = bb_config["provider"]

        # 更新其他决策维（写入 ablation 段作为参考）
        if "ablation" not in config:
            config["ablation"] = {}
        config["ablation"]["_selected"] = {
            "router": picks["router"]["name"],
            "synthesis": picks["synthesis"]["name"],
            "rag": picks["rag"]["name"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if not dry_run:
            with open(local_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            print(f"  [Config] {local_path} updated")
            print(f"  [Config] Backup saved to {bak_path}")
        else:
            print(f"  [DRY-RUN] Would update {local_path}")
    except Exception as e:
        print(f"  [WARN] Config update failed: {e}")
        if not dry_run:
            shutil.copy2(bak_path, local_path)
            print(f"  [Config] Restored from backup")
        return bak_path if not dry_run else None

    return bak_path if not dry_run else None


def _print_selection_summary(picks: dict):
    """打印选配结果摘要。"""
    print(f"\n{'=' * 60}")
    print(f"  Phase 2: 最优配置")
    print(f"{'=' * 60}")
    w = max(len(p["name"]) for p in picks.values()) + 2

    for dim_key in ["llm_backbone", "router", "synthesis", "rag"]:
        pick = picks[dim_key]
        print(f"  {dim_key:16s} → {pick['name']:<{w}}  score={pick['score']:.3f}")
    print(f"{'=' * 60}\n")


# ═══════════════════════════════════════════════════════════
# Phase 3: 对比执行
# ═══════════════════════════════════════════════════════════

def _run_compare(skip_nutrichat: bool = False, quick: bool = False):
    """执行 --compare。

    Args:
        skip_nutrichat: 跳过 NutriCHAT，仅跑基线
        quick: 快速模式（小采样 + 跳过 SelfCheck）
    """
    from src.main import run_comparison, setup_engine
    from src.config import Config, _LOCAL_OVERRIDE

    scoring = _LOCAL_OVERRIDE.get("scoring_compare", {})
    agent_mode = _LOCAL_OVERRIDE.get("agent_mode", "multi")

    if quick:
        quick_cfg = scoring.get("quick") or {}
        if quick_cfg:
            scoring = {**scoring, **quick_cfg}
        sr = quick_cfg.get("sample_ratio")
        if sr is not None:
            Config.EVAL_SAMPLE_RATIO = sr

    engine_mode = "multi" if agent_mode == "both" else agent_mode
    engine = setup_engine(
        mode=engine_mode,
        verbose=False,
        max_tokens=Config.MAX_TOKENS,
        rag_enabled=True,
    )

    run_comparison(
        engine=engine,
        query_set="all",
        detailed=False,
        baselines=None,
        skip_nutrichat=skip_nutrichat,
        scoring=scoring,
        agent_mode=agent_mode,
    )


# ═══════════════════════════════════════════════════════════
# Phase 4: 管道报告
# ═══════════════════════════════════════════════════════════

def _write_pipeline_report(log_dir: str, picks: dict, summary_text: str,
                            ablation_log_dir: str, compare_log_dir: str):
    """写入 ablation_to_compare_summary.md。"""
    path = os.path.join(log_dir, "ablation_to_compare_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Ablation → Compare 管道报告\n\n")
        f.write(f"*生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write(f"## 最优配置\n\n")
        f.write(f"| 维度 | 选择 | 综合得分 |\n")
        f.write(f"|------|------|---------|\n")
        for dim_key in ["llm_backbone", "router", "synthesis", "rag"]:
            pick = picks[dim_key]
            f.write(f"| {dim_key} | {pick['name']} | {pick['score']:.3f} |\n")

        f.write(f"\n> Multi-Agent 最终配置：{picks['llm_backbone']['name']} + ")
        f.write(f"Router={picks['router']['name']} + ")
        f.write(f"Synthesis={picks['synthesis']['name']} + ")
        f.write(f"RAG={picks['rag']['name']}\n\n")

        f.write(f"## 消融结果\n\n")
        f.write(summary_text)
        f.write(f"\n## 日志\n\n")
        f.write(f"- 管道日志: `{log_dir}`\n")
        f.write(f"- 消融日志: `{ablation_log_dir}`\n")
        f.write(f"- 对比日志: `{compare_log_dir}`\n")

    print(f"  [Report] {path}")


def _consolidate_results(log_dir: str, ablation_log_dir: str, compare_log_dir: str):
    """将消融和对比的结果文件复制到管道日志目录，统一存放。"""
    import shutil

    # 冲突文件：两个目录都有同名的文件（responses.json, config.yaml）
    _CONFLICT_NAMES = {"responses.json", "config.yaml"}

    for f in ["ablation_report.json", "ablation_report.md", "responses.json", "config.yaml"]:
        src = os.path.join(ablation_log_dir, f)
        dst = os.path.join(log_dir, f)
        if os.path.isfile(src) and not os.path.exists(dst):
            if f in _CONFLICT_NAMES:
                dst = os.path.join(log_dir, f"ablation_{f}")
            shutil.copy2(src, dst)
            print(f"  [Copy] {os.path.basename(dst)}")

    if compare_log_dir and os.path.isdir(compare_log_dir):
        for f in os.listdir(compare_log_dir):
            src = os.path.join(compare_log_dir, f)
            dst = os.path.join(log_dir, f)
            if not os.path.isfile(src):
                continue
            if f in _CONFLICT_NAMES:
                dst = os.path.join(log_dir, f"compare_{f}")
            elif os.path.exists(dst):
                dst = os.path.join(log_dir, f"compare_{f}")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"  [Copy] {os.path.basename(dst)}")


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def run_ablation_to_compare(skip_ablation: bool = False, dry_run: bool = False, quick: bool = False):
    """一键消融→对比管道。

    Args:
        skip_ablation: True 时复用最近的消融结果，不重新跑
        dry_run: True 时只选配不跑对比、不写配置
        quick: 快速模式（小采样 + 跳过 SelfCheck）
    """
    t0 = time.time()

    # ── Phase 1: 消融 ──────────────────────────────────────
    ablation_log_dir = None
    if skip_ablation:
        ablation_log_dir = _find_latest_ablation_log()
        if not ablation_log_dir:
            print("  [Error] No previous ablation log found for --skip-ablation")
            return
        print(f"\n{'=' * 60}")
        print(f"  Phase 1: 复用消融结果")
        print(f"  Log: {ablation_log_dir}")
        print(f"{'=' * 60}")
    else:
        print(f"\n{'=' * 60}")
        print(f"  Phase 1: 消融{'（快速模式）' if quick else ''}")
        print(f"{'=' * 60}")
        if quick:
            quick_sr = (_LOCAL_OVERRIDE.get("scoring_ablation") or {}).get("quick", {}).get("sample_ratio")
        else:
            quick_sr = None
        run_ablation(dim=None, verbose=False, query_set="all", sample_ratio=quick_sr)
        ablation_log_dir = _find_latest_ablation_log()

    # ── Phase 2: 选配 ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Phase 2: 自动选配")
    print(f"{'=' * 60}")

    report = _load_ablation_report(ablation_log_dir)
    if not report:
        print(f"  [Error] No ablation_report.json in {ablation_log_dir}")
        return

    parser = AblationParser(report)
    picks = parser.pick_all_best()
    _print_selection_summary(picks)
    summary_text = parser.summarize(picks)

    if dry_run:
        print("  [DRY-RUN] Skipping config update and compare\n")
        print(summary_text)
        return

    # ── 创建管道日志目录 ──────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(_PROJECT_ROOT, "logs", f"ablation_compare_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # 更新配置
    bak_path = _update_config_local(picks)

    # ── Phase 3: 对比 ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Phase 3: 基线对比")
    print(f"{'=' * 60}")
    _run_compare(skip_nutrichat=False, quick=quick)

    # 找到最近的 compare 日志
    logs_dir = os.path.join(_PROJECT_ROOT, "logs")
    compare_logs = sorted(
        [d for d in os.listdir(logs_dir) if d.startswith("comp_")],
        key=lambda d: os.path.getmtime(os.path.join(logs_dir, d)),
    )
    compare_log_dir = os.path.join(logs_dir, compare_logs[-1]) if compare_logs else ""

    # ── Phase 4: 报告 ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Phase 4: 管道报告")
    print(f"{'=' * 60}")
    _write_pipeline_report(log_dir, picks, summary_text, ablation_log_dir, compare_log_dir)

    # 统一复制结果到管道目录
    _consolidate_results(log_dir, ablation_log_dir, compare_log_dir)

    # 恢复配置（如果不需要改动永久生效）
    if bak_path and os.path.exists(bak_path):
        local_path = os.path.join(_PROJECT_ROOT, "config", "config.local.yaml")
        try:
            shutil.copy2(bak_path, local_path)
            # Verify copy succeeded before removing backup
            if os.path.exists(local_path) and os.path.getsize(local_path) == os.path.getsize(bak_path):
                os.remove(bak_path)
                print(f"  [Config] Restored original config.local.yaml from backup")
            else:
                print(f"  [WARN] Config restore verification failed, backup kept at {bak_path}")
        except Exception as e:
            print(f"  [WARN] Config restore failed: {e}, backup kept at {bak_path}")

    elapsed = time.time() - t0
    print(f"\n  [Done] Ablation -> Compare 管道完成 ({elapsed:.0f}s)")
    print(f"  管道日志: {log_dir}")
