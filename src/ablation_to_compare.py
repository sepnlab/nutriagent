"""Ablation → Compare end-to-end pipeline.

Pipeline:
  Phase 1: --ablation (full ablation, 30% sampling on decision dimensions, SelfCheck off)
  Phase 2: parse results → automatically select the best configuration
  Phase 3: --compare (all queries + SelfCheck + baseline comparison)

Usage:
  python main.py --ablation-to-compare
  python main.py --ablation-to-compare --skip-ablation  # reuse existing ablation results
  python main.py --ablation-to-compare --dry-run        # select configuration only, skip the comparison
"""
import json
import os
import shutil
import time
from typing import Dict, List, Optional, Tuple

from src.config import Config, _LOCAL_OVERRIDE, _PROJECT_ROOT, deep_merge
from src.evaluation.ablation import run_ablation

# ── Parameter mapping for the decision dimensions ─────────
# ablation variant name → actual parameter value (used to update config.local.yaml)

DECISION_DIMS = {
    "llm_backbone": {
        "variants": {},  # filled dynamically: variant_name -> {"model": ..., "provider": ...}
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

# ── Composite score weights ───────────────────────────────
WEIGHTS = {
    "correctness": 0.40,
    "scientific_depth": 0.30,
    "precision": 0.20,
    "hallucination": -0.10,
}


def _composite_score(metrics: dict) -> float:
    """Compute the composite score = 0.4×correctness + 0.3×depth + 0.2×precision - 0.1×halluc.

    If halluc is None that term is skipped and the weights are renormalized.
    """
    c = (metrics.get("correctness") or {}).get("mean") or 0
    d = (metrics.get("scientific_depth") or {}).get("mean") or 0
    p = (metrics.get("precision") or {}).get("mean") or 0
    h = (metrics.get("hallucination_n5") or metrics.get("hallucination") or {}).get("mean")

    if h is None:
        # Normalize to 1.0 when there is no halluc
        w_total = WEIGHTS["correctness"] + WEIGHTS["scientific_depth"] + WEIGHTS["precision"]
        return (c * WEIGHTS["correctness"] + d * WEIGHTS["scientific_depth"] + p * WEIGHTS["precision"]) / w_total
    return c * WEIGHTS["correctness"] + d * WEIGHTS["scientific_depth"] + p * WEIGHTS["precision"] + h * WEIGHTS["hallucination"]


def _load_ablation_report(log_dir: str) -> Optional[dict]:
    """Load ablation_report.json from the given log directory."""
    path = os.path.join(log_dir, "ablation_report.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_latest_ablation_log() -> Optional[str]:
    """Find the log directory of the most recent ablation run."""
    logs_dir = os.path.join(_PROJECT_ROOT, "logs")
    if not os.path.isdir(logs_dir):
        return None
    candidates = sorted(
        [d for d in os.listdir(logs_dir) if d.startswith("ablation_") and not d.startswith("ablation_compare_")],
        key=lambda d: os.path.getmtime(os.path.join(logs_dir, d)),
    )
    return os.path.join(logs_dir, candidates[-1]) if candidates else None


# ═══════════════════════════════════════════════════════════
# Phase 2: ablation result parsing + automatic configuration selection
# ═══════════════════════════════════════════════════════════

class AblationParser:
    """Parse ablation_report.json, compute composite scores, and select the best parameters."""

    def __init__(self, report: dict):
        self.report = report

    def _variant_scores(self, dim_key: str) -> List[Tuple[str, float, dict]]:
        """Return the dimension's variant list: [(name, composite_score, raw_metrics), ...]"""
        dim_data = self.report.get(dim_key, {})
        results = []
        for vname, metrics in dim_data.items():
            score = _composite_score(metrics)
            results.append((vname, score, metrics))
        # Sort by composite score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def pick_best(self, dim_key: str) -> Tuple[str, float, dict]:
        """Select the best variant of the given dimension. Returns (variant_name, score, metrics)."""
        ranked = self._variant_scores(dim_key)
        if not ranked:
            raise ValueError(f"Dimension '{dim_key}' not found in ablation report")
        return ranked[0]

    def pick_all_best(self) -> dict:
        """Select the best variant across all 4 decision dimensions."""
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
        """Generate the configuration summary (used for the Step 4 report)."""
        lines = []
        for dim_key in ["llm_backbone", "router", "synthesis", "rag"]:
            pick = picks[dim_key]
            ranked = self._variant_scores(dim_key)
            lines.append(f"## {dim_key}")
            lines.append("")
            lines.append(f"| Variant | Correctness | Depth | Precision | Halluc | Composite Score |")
            lines.append(f"|---------|------------|-------|-----------|--------|---------|")
            for vname, score, metrics in ranked:
                c = (metrics.get("correctness") or {}).get("mean")
                d = (metrics.get("scientific_depth") or {}).get("mean")
                p = (metrics.get("precision") or {}).get("mean")
                h = (metrics.get("hallucination_n5") or metrics.get("hallucination") or {}).get("mean")
                h_str = f"{h:.4f}" if h is not None else "N/A"
                marker = " ← **best**" if vname == pick["name"] else ""
                lines.append(
                    f"| {vname} | {c:.2f} | {d:.2f} | {p:.2f} | {h_str} | {score:.3f}{marker} |"
                )
            lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# Phase 2: configuration update
# ═══════════════════════════════════════════════════════════

def _resolve_backbone_config(variant_name: str) -> dict:
    """Resolve an llm_backbone variant name to the model's provider+model configuration.

    Search order:
      1. models.yaml baselines (supports both a flat list and a provider-grouped dict)
      2. Infer the provider from the name (the "provider/ModelName" format)
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
    """Update config.local.yaml with the best parameters. Returns the backup path (None in dry_run)."""
    local_path = os.path.join(_PROJECT_ROOT, "config", "config.local.yaml")
    if not os.path.exists(local_path):
        print("  [ERROR] config.local.yaml not found")
        return None

    # Read the current configuration
    with open(local_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Backup
    bak_path = local_path + ".bak"
    if not dry_run:
        shutil.copy2(local_path, bak_path)

    # Resolve the backbone configuration
    bb_name = picks["llm_backbone"]["name"]
    bb_config = _resolve_backbone_config(bb_name)

    # Build the change summary
    changes = [
        ("nutrichat.model", bb_config["model"]),
        ("nutrichat.provider", bb_config["provider"]),
    ]

    # Modify safely through yaml (by rewriting the file)
    import yaml
    try:
        with open(local_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        # Update nutrichat
        if "nutrichat" not in config:
            config["nutrichat"] = {}
        config["nutrichat"]["model"] = bb_config["model"]
        config["nutrichat"]["provider"] = bb_config["provider"]

        # Update the other decision dimensions (written to the ablation section for reference)
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
    """Print a summary of the selected configuration."""
    print(f"\n{'=' * 60}")
    print(f"  Phase 2: Best configuration")
    print(f"{'=' * 60}")
    w = max(len(p["name"]) for p in picks.values()) + 2

    for dim_key in ["llm_backbone", "router", "synthesis", "rag"]:
        pick = picks[dim_key]
        print(f"  {dim_key:16s} → {pick['name']:<{w}}  score={pick['score']:.3f}")
    print(f"{'=' * 60}\n")


# ═══════════════════════════════════════════════════════════
# Phase 3: comparison execution
# ═══════════════════════════════════════════════════════════

def _run_compare(skip_nutrichat: bool = False, quick: bool = False):
    """Execute --compare.

    Args:
        skip_nutrichat: skip NutriCHAT and run baselines only
        quick: quick mode (small sample + skip SelfCheck)
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
# Phase 4: pipeline report
# ═══════════════════════════════════════════════════════════

def _write_pipeline_report(log_dir: str, picks: dict, summary_text: str,
                            ablation_log_dir: str, compare_log_dir: str):
    """Write ablation_to_compare_summary.md."""
    path = os.path.join(log_dir, "ablation_to_compare_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Ablation → Compare Pipeline Report\n\n")
        f.write(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write(f"## Best configuration\n\n")
        f.write(f"| Dimension | Choice | Composite Score |\n")
        f.write(f"|-----------|--------|-----------------|\n")
        for dim_key in ["llm_backbone", "router", "synthesis", "rag"]:
            pick = picks[dim_key]
            f.write(f"| {dim_key} | {pick['name']} | {pick['score']:.3f} |\n")

        f.write(f"\n> Final Multi-Agent configuration: {picks['llm_backbone']['name']} + ")
        f.write(f"Router={picks['router']['name']} + ")
        f.write(f"Synthesis={picks['synthesis']['name']} + ")
        f.write(f"RAG={picks['rag']['name']}\n\n")

        f.write(f"## Ablation results\n\n")
        f.write(summary_text)
        f.write(f"\n## Logs\n\n")
        f.write(f"- Pipeline log: `{log_dir}`\n")
        f.write(f"- Ablation log: `{ablation_log_dir}`\n")
        f.write(f"- Comparison log: `{compare_log_dir}`\n")

    print(f"  [Report] {path}")


def _consolidate_results(log_dir: str, ablation_log_dir: str, compare_log_dir: str):
    """Copy the ablation and comparison result files into the pipeline log directory for unified storage."""
    import shutil

    # Conflicting files: both directories can contain the same file names (responses.json, config.yaml)
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
# Main entry point
# ═══════════════════════════════════════════════════════════

def run_ablation_to_compare(skip_ablation: bool = False, dry_run: bool = False, quick: bool = False):
    """One-shot ablation → comparison pipeline.

    Args:
        skip_ablation: if True, reuse the most recent ablation results instead of re-running
        dry_run: if True, only select the configuration without running the comparison or writing config
        quick: quick mode (small sample + skip SelfCheck)
    """
    t0 = time.time()

    # ── Phase 1: ablation ──────────────────────────────────
    ablation_log_dir = None
    if skip_ablation:
        ablation_log_dir = _find_latest_ablation_log()
        if not ablation_log_dir:
            print("  [Error] No previous ablation log found for --skip-ablation")
            return
        print(f"\n{'=' * 60}")
        print(f"  Phase 1: Reusing ablation results")
        print(f"  Log: {ablation_log_dir}")
        print(f"{'=' * 60}")
    else:
        print(f"\n{'=' * 60}")
        print(f"  Phase 1: Ablation{' (quick mode)' if quick else ''}")
        print(f"{'=' * 60}")
        if quick:
            quick_sr = (_LOCAL_OVERRIDE.get("scoring_ablation") or {}).get("quick", {}).get("sample_ratio")
        else:
            quick_sr = None
        run_ablation(dim=None, verbose=False, query_set="all", sample_ratio=quick_sr)
        ablation_log_dir = _find_latest_ablation_log()

    # ── Phase 2: configuration selection ───────────────────
    print(f"\n{'=' * 60}")
    print(f"  Phase 2: Automatic configuration selection")
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

    # ── Create the pipeline log directory ──────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(_PROJECT_ROOT, "logs", f"ablation_compare_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Update the configuration
    bak_path = _update_config_local(picks)

    # ── Phase 3: comparison ────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Phase 3: Baseline comparison")
    print(f"{'=' * 60}")
    _run_compare(skip_nutrichat=False, quick=quick)

    # Find the most recent compare log
    logs_dir = os.path.join(_PROJECT_ROOT, "logs")
    compare_logs = sorted(
        [d for d in os.listdir(logs_dir) if d.startswith("comp_")],
        key=lambda d: os.path.getmtime(os.path.join(logs_dir, d)),
    )
    compare_log_dir = os.path.join(logs_dir, compare_logs[-1]) if compare_logs else ""

    # ── Phase 4: report ────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Phase 4: Pipeline report")
    print(f"{'=' * 60}")
    _write_pipeline_report(log_dir, picks, summary_text, ablation_log_dir, compare_log_dir)

    # Copy all results into the pipeline directory
    _consolidate_results(log_dir, ablation_log_dir, compare_log_dir)

    # Restore the configuration (if the change should not persist)
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
    print(f"\n  [Done] Ablation -> Compare pipeline finished ({elapsed:.0f}s)")
    print(f"  Pipeline log: {log_dir}")
