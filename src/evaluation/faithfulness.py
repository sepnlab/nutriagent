"""Faithfulness evaluation for NutriCHAT.

Two complementary metrics:
1. **Numeric faithfulness** — Are numeric values in the response supported by tool data?
2. **Semantic faithfulness** — Are semantic claims in the response entailed by tool observations?

Semantic faithfulness reuses SelfCheckGPT's class-level CrossEncoder NLI model,
avoiding duplicate GPU memory usage.
"""
import re
from typing import Dict, List, Optional, Tuple


# ── Shared CrossEncoder (reuse SelfCheckGPT's instance) ──────────────

_SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import CrossEncoder
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


def _get_nli_model():
    """Get CrossEncoder NLI model, reusing SelfCheckGPT's class-level cache.

    If SelfCheckGPT already loaded the model, we reuse it (zero extra GPU
    memory). Otherwise we load one and cache it back for future reuse.
    """
    from src.evaluation.selfcheck import SelfCheckGPT

    if SelfCheckGPT._shared_nli_model is not None:
        return SelfCheckGPT._shared_nli_model

    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        return None

    from src.config import Config

    model_candidates = [
        Config.NLI_MODEL_PATH,
        "cross-encoder/nli-deberta-v3-base",
        "cross-encoder/nli-MiniLM2-L6",
    ]

    for model_name in model_candidates:
        import os as _os
        import warnings as _warnings
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                if _os.path.isdir(str(model_name)):
                    pass
                model = CrossEncoder(model_name, device=Config.NLI_DEVICE)
                SelfCheckGPT._shared_nli_model = model
                SelfCheckGPT._shared_nli_device = Config.NLI_DEVICE
                return model
        except Exception:
            continue
    return None


# ── Observation → natural-language evidence ──────────────────────

_KEY_VALUE_PAT = re.compile(
    r"(?:^|\n)\s*([A-Za-z_][A-Za-z0-9_()\s/%]+?)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
    re.MULTILINE,
)


def _observation_to_sentences(observation: str) -> List[str]:
    """Convert structured tool output (Key: Value) into readable sentences.

    "CP: 8.5"  →  "The CP is 8.5."
    "ME_kcal_kg: 3350"  →  "The ME kcal/kg is 3350."
    "Ca: 0.02%"  →  "The Ca is 0.02 percent."
    """
    sentences = []
    seen_vals = set()  # Deduplicate identical values
    for m in _KEY_VALUE_PAT.finditer(observation):
        key = m.group(1).strip()
        val = m.group(2)

        val_dedup = f"{key}={val}"
        if val_dedup in seen_vals:
            continue
        seen_vals.add(val_dedup)

        # Normalise key for readability
        readable = key.replace("_", " ").strip()
        readable = re.sub(r"\s+", " ", readable)

        if len(readable) < 2:
            continue

        # Check what comes after the value for unit hints
        end = m.end()
        rest = observation[end:end + 10].strip()
        if rest and rest.startswith("%"):
            sentences.append(f"The {readable} is {val} percent.")
        elif rest and re.match(r'^(g|kg|mg|kcal|mm|N|IU)\b', rest):
            unit = rest.split()[0] if rest else ""
            sentences.append(f"The {readable} is {val} {unit}.")
        else:
            sentences.append(f"The {readable} is {val}.")

    return sentences


# ── Response → verifiable claims ─────────────────────────────────

_NUMERIC_WORD_RATIO = 0.5  # If >=50% tokens are purely numeric, skip


def _split_claims(text: str) -> List[str]:
    """Split response into sentence-level claims, filtering out numeric-heavy ones.

    Numeric-heavy sentences (e.g. "CP: 8.5, Ca: 0.96, ME: 3025") are
    already covered by numeric faithfulness and are skipped here.
    """
    text = re.sub(r"\s+", " ", text.strip())
    raw = re.split(r"(?<=[.!?])\s+", text)
    claims = []
    for s in raw:
        s = s.strip()
        if len(s) < 15:
            continue
        # Skip bullet lists / itemised numbers
        words = s.split()
        numeric_count = sum(1 for w in words if re.match(r"^[0-9,.%±≥≤<>/]+$", w))
        if words and numeric_count / len(words) >= _NUMERIC_WORD_RATIO:
            continue
        claims.append(s)
    return claims


# ── Numeric faithfulness (existing) ──────────────────────────────

_NUMBER_PAT = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def _normalize_param(name: str) -> str:
    n = name.strip().lower()
    n = re.sub(r"[_\s\-]+", "", n)
    n = re.sub(r"[%()/]", "", n)
    return n


def _parse_tool_observations(steps: List[dict]) -> List[Tuple[str, float]]:
    pairs: List[Tuple[str, float]] = []
    for step in steps:
        obs = step.get("observation", "")
        if not obs or step.get("action", "") in ("unknown", ""):
            continue
        for m in _KEY_VALUE_PAT.finditer(obs):
            name = m.group(1).strip()
            try:
                val = float(m.group(2))
            except ValueError:
                continue
            if 0 < val < 1e6:
                pairs.append((name, val))
    return pairs


def _parse_response_values(text: str) -> List[Tuple[str, float]]:
    results: List[Tuple[str, float]] = []
    for m in _KEY_VALUE_PAT.finditer(text):
        name = m.group(1).strip()
        try:
            val = float(m.group(2))
        except ValueError:
            continue
        if 0 < val < 1e6:
            results.append((name, val))

    for m in _NUMBER_PAT.finditer(text):
        val = float(m.group(0))
        if not (0 < val < 1e6) or (1900 <= val <= 2099):
            continue
        start = max(0, m.start() - 30)
        end = min(len(text), m.end() + 30)
        snippet = text[start:end].strip()
        already_captured = any(abs(val - v) < 0.001 for _, v in results)
        if not already_captured:
            results.append((snippet, val))

    return results


def _values_match(v1: float, v2: float, tolerance: float = 0.05) -> bool:
    if v1 == v2:
        return True
    if v1 == 0 or v2 == 0:
        return False
    ratio = abs(v1 - v2) / max(abs(v1), abs(v2))
    return ratio <= tolerance


def compute_faithfulness_from_steps(response: str, steps: List[dict]) -> Optional[float]:
    """Numeric faithfulness: what fraction of numeric claims are supported by tool data?"""
    tool_pairs = _parse_tool_observations(steps)
    if not tool_pairs:
        return None

    tool_vals = [v for _, v in tool_pairs]
    response_pairs = _parse_response_values(response)

    if not response_pairs:
        return None

    matched = 0
    total = 0
    for ctx, rval in response_pairs:
        total += 1
        supported = any(_values_match(rval, tv) for tv in tool_vals)
        if supported:
            matched += 1

    return matched / total if total > 0 else None


def compute_faithfulness_batch(
    responses: List[str],
    step_traces: List[List[dict]],
) -> List[Optional[float]]:
    return [
        compute_faithfulness_from_steps(resp, steps)
        for resp, steps in zip(responses, step_traces)
    ]


# ── Semantic faithfulness (NLI-based, new) ───────────────────────

def compute_semantic_faithfulness(
    response: str,
    steps: List[dict],
    entail_threshold: float = 0.7,
    contradict_threshold: float = 0.7,
) -> Optional[float]:
    """Semantic faithfulness: do tool observations NLI-entail response claims?

    For each non-numeric claim (sentence) in the response, checks against
    all tool observation sentences using the CrossEncoder NLI model:

      - Entailment prob ≥ *entail_threshold*     → 1.0  (supported)
      - Contradiction prob ≥ *contradict_threshold*  → 0.0  (contradicted)
      - Otherwise                                → 0.5  (neutral / unverifiable)

    The overall score is the average across all claims.

    Returns None if no tool observations, no verifiable claims, or the
    NLI model is unavailable.
    """
    nli_model = _get_nli_model()
    if nli_model is None:
        return None

    # Extract observations
    observations = [
        s.get("observation", "") for s in steps
        if s.get("observation") and s.get("action", "") not in ("unknown", "")
    ]
    if not observations:
        return None

    # Convert to natural-language evidence sentences
    evidence = []
    for obs in observations:
        evidence.extend(_observation_to_sentences(obs))
    if not evidence:
        return None

    # Extract verifiable claims
    claims = _split_claims(response)
    if not claims:
        return None

    # Batch NLI: all (evidence, claim) pairs in one call
    import numpy as np

    pairs = [(ev, cl) for ev in evidence for cl in claims]
    try:
        logits_batch = nli_model.predict(pairs, batch_size=32)
    except Exception:
        return None

    n_ev = len(evidence)
    # For each claim, track the best score across all evidence sentences
    claim_best = [0.5] * len(claims)  # default: neutral

    for i, claim in enumerate(claims):
        for j in range(n_ev):
            idx = j * len(claims) + i
            if idx >= len(logits_batch):
                continue
            try:
                logits = logits_batch[idx]
                ex = np.exp(logits - np.max(logits))
                probs = ex / ex.sum()
                # NLI label order: 0=contradiction, 1=entailment, 2=neutral
                entail_prob = float(probs[1])
                contrad_prob = float(probs[0])

                if entail_prob >= entail_threshold:
                    claim_best[i] = max(claim_best[i], 1.0)
                elif contrad_prob >= contradict_threshold:
                    claim_best[i] = min(claim_best[i], 0.0)
                # else neutral → keep current
            except Exception:
                continue

    # Filter out claims that couldn't be evaluated (shouldn't happen)
    valid = [s for s in claim_best]
    if not valid:
        return None

    return sum(valid) / len(valid)


def compute_semantic_faithfulness_batch(
    responses: List[str],
    step_traces: List[List[dict]],
) -> List[Optional[float]]:
    """Batch version of compute_semantic_faithfulness."""
    return [
        compute_semantic_faithfulness(resp, steps)
        for resp, steps in zip(responses, step_traces)
    ]
