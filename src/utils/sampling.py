"""Stratified sampling utilities for evaluation query selection."""

import random
from collections import defaultdict
from typing import List, Optional


def stratified_sample(
    queries: List[dict],
    ratio: Optional[float] = None,
    categories: Optional[List[str]] = None,
    seed: int = 42,
) -> List[dict]:
    """Stratified sample queries by category, maintaining distribution.

    Args:
        queries: List of query dicts, each with a ``category`` key.
        ratio:
            - ``None``: return all (no sampling)
            - ``>= 1``: count mode — ``int(ratio)`` queries per category
            - ``0 < ratio < 1``: ratio mode — ``round(n * ratio)`` per category
            - ``0``: return empty
        categories: Optional whitelist; only categories in this list are kept.
        seed: Deterministic seed for reproducibility.

    Returns:
        Sampled (and optionally category-filtered) list of query dicts,
        shuffled to avoid category-block ordering.
    """
    # Step 1: category whitelist filter
    pool = queries
    if categories is not None:
        cats = set(categories)
        pool = [q for q in queries if q.get("category") in cats]

    # Step 2: no sampling requested
    if ratio is None:
        return pool
    if ratio == 0:
        return []

    # Step 3: stratified sampling
    rng = random.Random(seed)
    groups: dict[str, list] = defaultdict(list)
    for q in pool:
        groups[q.get("category", "Unknown")].append(q)

    count_mode = ratio >= 1
    result: List[dict] = []
    for cat, group in groups.items():
        if count_mode:
            n = int(ratio)
        else:
            n = max(1, round(len(group) * ratio))

        actual = min(n, len(group))
        if actual < n:
            print(f"  [Sampling] {cat}: requested {n}, only {len(group)} available, using {actual}")
        sampled = rng.sample(group, actual)
        result.extend(sampled)

    # Shuffle to avoid category-block ordering
    rng.shuffle(result)

    mode = "count" if count_mode else "ratio"
    print(
        f"[Sampling] {mode} sample: {len(pool)} -> {len(result)} "
        f"queries ({'per category' if count_mode else 'ratio'}: {ratio}, seed={seed})"
    )

    return result
