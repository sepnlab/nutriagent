"""统一的评估查询加载器。

整合 main.py 和 ablation.py 中的三份重复查询加载逻辑。
"""
import json
import os
from typing import Dict, List, Optional

from src.config import Config

# 中文品类 → 基础英文品类映射（整数采样模式下归并，避免中文品类被当作独立分层）
_CN_CATEGORY_MAP: Dict[str, str] = {
    "Chinese White-Feathered Broiler Performance": "Performance Management",
    "Chinese Yellow-Feathered Broiler Performance": "Performance Management",
    "Chinese Broiler Nutrient Requirements": "Nutrient Requirements",
    "Cross-Breed Comparison (Broiler)": "Performance Management",
    "Chinese Layer Performance": "Performance Management",
    "Chinese Layer Nutrient Requirements": "Nutrient Requirements",
}


def _load_single(filepath: str) -> List[dict]:
    """从 JSON 文件加载查询并展平 category 字段。

    自动为缺少 language 字段的查询推断语言：
    - id 以 CN- 开头 → zh
    - 查询含中文字符 → zh
    - 其他 → en
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
    """就地将中文品类名映射到基础英文品类（整数采样用）。"""
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
    """加载评估查询集，支持中文合并 + 分层采样。

    Args:
        queries_file: JSON 文件路径。为 None 时从 Config.EVALUATION_QUERIES_FILE 读取。
        query_set: "all" / "chinese" / "western"。chinese 仅保留 id 以 CN- 开头的查询。
        sample_ratio: 传给 stratified_sample，None=不采样，<1=比例模式，>=1=每类 N 条。
        categories: 可选白名单，只保留指定 category。

    Returns:
        查询 dict 列表，每个 dict 包含 query、category、id、expected_value 等字段。
    """
    filepath = queries_file or Config.EVALUATION_QUERIES_FILE

    # 1. 加载指定文件
    queries = _load_single(filepath)

    # 2. 按 query_set 过滤（优先用 language 字段，回退到 id 前缀）
    if query_set == "chinese":
        queries = [q for q in queries if q.get("language") == "zh" or q.get("id", "").startswith("CN-")]
    elif query_set == "western":
        queries = [q for q in queries if q.get("language") == "en" and not q.get("id", "").startswith("CN-")]

    # 4. 整数采样模式下，将中文品类归并到基础英文品类
    if sample_ratio is not None and sample_ratio >= 1:
        _normalize_categories(queries)

    # 5. 分层采样 + category 过滤
    if sample_ratio is not None or categories is not None:
        from src.utils.sampling import stratified_sample
        queries = stratified_sample(queries, ratio=sample_ratio, categories=categories)

    return queries
