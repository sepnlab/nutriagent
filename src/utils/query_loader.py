"""Unified evaluation query loader.

Consolidates the three duplicated query-loading implementations in main.py and ablation.py.
"""
import json
import os
from typing import Dict, List, Optional

from src.config import Config

# Chinese-category → base English-category mapping (merged in integer-sampling mode so Chinese categories are not treated as separate strata)
_CN_CATEGORY_MAP: Dict[str, str] = {
    "Chinese White-Feathered Broiler Performance": "Performance Management",
    "Chinese Yellow-Feathered Broiler Performance": "Performance Management",
    "Chinese Broiler Nutrient Requirements": "Nutrient Requirements",
    "Cross-Breed Comparison (Broiler)": "Performance Management",
    "Chinese Layer Performance": "Performance Management",
    "Chinese Layer Nutrient Requirements": "Nutrient Requirements",
}


def _load_single(filepath: str) -> List[dict]:
    """Load queries from a JSON file and flatten the category field.

    Language is inferred automatically for queries that lack a language field:
    - id starting with CN- → zh
    - query contains Chinese characters → zh
    - otherwise → en
    """
    import re
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = []
    for cat, qlist in data.items():
        for q in qlist:
            q["category"] = cat
            # Auto-detect language if not present
            if "language" not in q:
                qid = q.get("id", "")
                qtext = q.get("query", "")
                if qid.startswith("CN-") or re.search(r'[一-鿿]', qtext):
                    q["language"] = "zh"
                else:
                    q["language"] = "en"
            queries.append(q)
    return queries


def _normalize_categories(queries: List[dict]) -> None:
    """In-place mapping of Chinese category names to base English categories (used for integer sampling)."""
    for q in queries:
        mapped = _CN_CATEGORY_MAP.get(q.get("category", ""))
        if mapped is not None:
            q["category"] = mapped


def load_queries(
    queries_file: Optional[str] = None,
    query_set: str = "all",
    sample_ratio: Optional[float] = None,
    categories: Optional[List[str]] = None,
) -> List[dict]:
    """Load the evaluation query set, supporting Chinese merging + stratified sampling.

    Args:
        queries_file: JSON file path. Read from Config.EVALUATION_QUERIES_FILE when None.
        query_set: "all" / "chinese" / "western". chinese keeps only queries whose id starts with CN-.
        sample_ratio: passed to stratified_sample; None=no sampling, <1=proportional mode, >=1=N per category.
        categories: optional whitelist; only keep the specified categories.

    Returns:
        A list of query dicts; each dict contains query, category, id, expected_value etc.
    """
    filepath = queries_file or Config.EVALUATION_QUERIES_FILE

    # 1. Load the specified file
    queries = _load_single(filepath)

    # 2. Filter by query_set (prefer the language field, fall back to the id prefix)
    if query_set == "chinese":
        queries = [q for q in queries if q.get("language") == "zh" or q.get("id", "").startswith("CN-")]
    elif query_set == "western":
        queries = [q for q in queries if q.get("language") == "en" and not q.get("id", "").startswith("CN-")]

    # 4. In integer-sampling mode, merge Chinese categories into base English categories
    if sample_ratio is not None and sample_ratio >= 1:
        _normalize_categories(queries)

    # 5. Stratified sampling + category filtering
    if sample_ratio is not None or categories is not None:
        from src.utils.sampling import stratified_sample
        queries = stratified_sample(queries, ratio=sample_ratio, categories=categories)

    return queries
