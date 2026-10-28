"""Cross-specialist consistency validation.

An earlier implementation used two regexes that matched the wrong numbers:

    \\b(?:p|phosphorus|avail_p)\\D*?(\\d+)     # "crude protein ... 8.8" -> P=8.8
    \\b(?:ca|calcium)\\D*?(\\d+)               # "cage density is 6"   -> Ca=6

and compared values of *different* attributes (CP vs ME) against a shared
mean. The resulting warnings were largely noise.

This module replaces that logic with a conservative, claim-aware validator:

1. Split text into short segments and skip any segment that mentions more than
   one attribute (this prevents mis-aligned values such as
   "calcium and phosphorus are 4.04% and 0.45%").
2. Extract numeric claims as (entity, attribute, value, unit, basis) using
   attribute tokens with real word boundaries (``\\bP\\b`` never matches
   "protein"; ``\\bCa\\b`` never matches "cage").
3. Compare values only when entity, attribute and basis all match.
4. Convert compatible units (%, g/kg, mg/kg; kcal/kg <-> MJ/kg) first.
5. If the entity cannot be resolved, do not compare across specialists
   ("no confident match" instead of a guess).
6. The generic Ca:P ratio rule is disabled by default: layer diets commonly
   span ratios that a generic poultry rule would flag as abnormal. It can be
   enabled explicitly for structured contexts where the phosphorus basis is
   known.

The public entry point is :func:`validate_specialist_outputs`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Attribute patterns ──────────────────────────────────────
# Order matters: specific phosphorus forms must win over the generic "P".
# Every pattern requires an explicit unit, and the gap between the attribute
# and the number cannot cross a sentence/segment boundary. The gap may include
# words such as "content of maize is", but the per-segment single-attribute
# guard prevents cross-attribute misalignment.
_GAP = r"[^.;\n]{0,50}?"
_ATTR_PATTERNS: Sequence[Tuple[str, str]] = (
    (
        "available_phosphorus",
        r"\b(?:available\s+phosphorus|available\s+P|non-?phytate\s+phosphorus|"
        r"NPP|aP)\b" + _GAP + r"([0-9]+(?:\.[0-9]+)?)\s*(%)",
    ),
    (
        "total_phosphorus",
        r"\b(?:total\s+phosphorus|total\s+P|P_total)\b" + _GAP +
        r"([0-9]+(?:\.[0-9]+)?)\s*(%)",
    ),
    (
        "phosphorus",
        r"\b(?:phosphorus|P)\b" + _GAP + r"([0-9]+(?:\.[0-9]+)?)\s*(%)",
    ),
    (
        "calcium",
        r"\b(?:calcium|Ca)\b" + _GAP + r"([0-9]+(?:\.[0-9]+)?)\s*(%|g/kg|mg/kg)",
    ),
    (
        "crude_protein",
        r"\b(?:crude\s+protein|CP)\b" + _GAP + r"([0-9]+(?:\.[0-9]+)?)\s*(%|g/kg)",
    ),
    (
        "metabolizable_energy",
        r"\b(?:metabolizable\s+energy|ME)\b" + _GAP +
        r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(kcal/kg|kcal\s*/\s*kg|MJ/kg|MJ\s*/\s*kg)",
    ),
    (
        "lysine",
        r"\b(?:lysine|Lys)\b" + _GAP + r"([0-9]+(?:\.[0-9]+)?)\s*(%|g/kg)",
    ),
    (
        "methionine",
        r"\b(?:methionine|Met)\b" + _GAP + r"([0-9]+(?:\.[0-9]+)?)\s*(%|g/kg)",
    ),
)

_COMPILED = [(attr, re.compile(p, re.IGNORECASE)) for attr, p in _ATTR_PATTERNS]

# Value-before-attribute order ("35% calcium", "350 g/kg calcium") is common in
# forward-only patterns missed it entirely, so E3's conflict recall was zero.
_ATTR_PATTERNS_REVERSED: Sequence[Tuple[str, str]] = (
    (
        "available_phosphorus",
        r"([0-9]+(?:\.[0-9]+)?)\s*(%)\s*" + _GAP +
        r"\b(?:available\s+phosphorus|available\s+P|non-?phytate\s+phosphorus|NPP|aP)\b",
    ),
    (
        "total_phosphorus",
        r"([0-9]+(?:\.[0-9]+)?)\s*(%)\s*" + _GAP +
        r"\b(?:total\s+phosphorus|total\s+P|P_total)\b",
    ),
    (
        "phosphorus",
        r"([0-9]+(?:\.[0-9]+)?)\s*(%)\s*" + _GAP + r"\b(?:phosphorus|P)\b",
    ),
    (
        "calcium",
        r"([0-9]+(?:\.[0-9]+)?)\s*(%|g/kg|mg/kg)\s*" + _GAP +
        r"\b(?:calcium|Ca)\b",
    ),
    (
        "crude_protein",
        r"([0-9]+(?:\.[0-9]+)?)\s*(%|g/kg)\s*" + _GAP +
        r"\b(?:crude\s+protein|CP)\b",
    ),
    (
        "metabolizable_energy",
        r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(kcal/kg|kcal\s*/\s*kg|MJ/kg|MJ\s*/\s*kg)\s*" +
        _GAP + r"\b(?:metabolizable\s+energy|ME)\b",
    ),
    (
        "lysine",
        r"([0-9]+(?:\.[0-9]+)?)\s*(%|g/kg)\s*" + _GAP + r"\b(?:lysine|Lys)\b",
    ),
    (
        "methionine",
        r"([0-9]+(?:\.[0-9]+)?)\s*(%|g/kg)\s*" + _GAP + r"\b(?:methionine|Met)\b",
    ),
)

_COMPILED_REVERSED = [
    (attr, re.compile(p, re.IGNORECASE)) for attr, p in _ATTR_PATTERNS_REVERSED
]

# Used to decide whether a segment mentions more than one attribute. The
# generic phosphorus token covers available/total phosphorus as well.
_ATTR_KEYWORD_RE = {
    "phosphorus": re.compile(r"\b(?:phosphorus|P)\b", re.IGNORECASE),
    "calcium": re.compile(r"\b(?:calcium|Ca)\b", re.IGNORECASE),
    "crude_protein": re.compile(r"\b(?:crude\s+protein|CP)\b", re.IGNORECASE),
    "metabolizable_energy": re.compile(
        r"\b(?:metabolizable\s+energy|ME)\b", re.IGNORECASE
    ),
    "lysine": re.compile(r"\b(?:lysine|Lys)\b", re.IGNORECASE),
    "methionine": re.compile(r"\b(?:methionine|Met)\b", re.IGNORECASE),
}

# Unit family + canonical conversion to a comparable float.
# Percent-family values are expressed in %; energy in kcal/kg.
_UNIT_FACTOR = {
    "%": ("percent", 1.0),
    "g/kg": ("percent", 0.1),          # 1 g/kg = 0.1 %
    "mg/kg": ("percent", 0.0001),      # 1 mg/kg = 0.0001 %
    "kcal/kg": ("energy", 1.0),
    "kcal / kg": ("energy", 1.0),
    "mj/kg": ("energy", 239.0),        # 1 MJ = 239 kcal
    "mj / kg": ("energy", 239.0),
}


def _canon_unit(unit: str) -> Tuple[str, float]:
    key = re.sub(r"\s+", " ", unit.strip().lower())
    return _UNIT_FACTOR[key]


@dataclass
class Claim:
    specialist: str
    attribute: str
    value: float
    unit: str
    basis: str
    entity: Optional[str]
    text: str

    @property
    def canonical(self) -> float:
        _, factor = _canon_unit(self.unit)
        return self.value * factor


@dataclass
class ValidationPolicy:
    """Thresholds for cross-specialist consistency checks."""

    relative_tolerance: float = 0.05
    # 35%-vs-38% Ca case (7.9% relative). A 10% default silently missed it.
    # Ca:P is opt-in. Generic poultry ranges are unsafe for layer diets because
    # Ca is commonly 3.5-4.2% while available P is 0.3-0.5% (ratios 8-14:1).
    ca_p_check: bool = False
    ca_p_range: Tuple[float, float] = (1.0, 20.0)
    max_entity_distance: int = 120


@dataclass
class ValidationResult:
    warnings: List[str] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    compared_pairs: int = 0


# ── Entity lexicon ──────────────────────────────────────────
def _usable_alias(key: str) -> bool:
    """Drop aliases that are too short to match safely.

    Chinese aliases are allowed at two characters; Latin aliases need at least
    three characters. Ambiguous aliases are excluded separately.
    """
    if re.search(r"[\u4e00-\u9fff]", key):
        return len(key) >= 2
    return len(key) >= 3


@lru_cache(maxsize=1)
def _entity_lexicon() -> Tuple[Tuple[str, str], ...]:
    """Map alias -> canonical entity, longest alias first.

    Sources: corpus ingredient/breed names plus the curated alias table
    (``corpus/shared/ingredient_alias.csv``). Aliases flagged ``ambiguous``
    (e.g. bare soybean-meal aliases across 46/48/50% variants) are skipped so the resolver
    never guesses a variant.
    """
    import pandas as pd

    mapping: dict[str, str] = {}

    def add(alias: str, canonical: str) -> None:
        key = str(alias).strip().lower()
        if not key or not _usable_alias(key):
            return
        mapping.setdefault(key, str(canonical).strip())

    for rel, column in (
        ("corpus/shared/feed_ingredient_bank.csv", "Ingredient"),
        ("corpus/layer/nutrient_requirements.csv", "Breed"),
        ("corpus/layer/performance_management.csv", "Breed"),
    ):
        path = PROJECT_ROOT / rel
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=[column])
        except Exception:
            continue
        for value in df[column].dropna():
            add(value, value)

    alias_path = PROJECT_ROOT / "corpus" / "shared" / "ingredient_alias.csv"
    if alias_path.exists():
        try:
            aliases = pd.read_csv(alias_path)
            for _, row in aliases.iterrows():
                if bool(row.get("ambiguous", False)):
                    continue
                add(row["alias"], row["canonical_name"])
        except Exception:
            pass

    # Commodity-level short names. The feed tables use long names such as
    # "Soybean meal, solvent extracted, protein 44%", but free text says
    # "soybean meal". Map the first comma-separated component to a family
    # entity; when several table rows share the family, compare at the family
    # level rather than guessing a specific row.
    families: dict[str, set[str]] = {}
    for canonical in set(mapping.values()):
        family = canonical.split(",")[0].strip().lower()
        if _usable_alias(family):
            families.setdefault(family, set()).add(canonical)
    for family, canonicals in families.items():
        target = next(iter(canonicals)) if len(canonicals) == 1 else family
        add(family, target)

    # Breed/phase labels (e.g. "Lay I") are legitimate entities for requirement
    # claims; add the Phase column so those claims can be compared too.
    for rel in (
        "corpus/layer/nutrient_requirements.csv",
        "corpus/layer/performance_management.csv",
    ):
        path = PROJECT_ROOT / rel
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "Phase" in df.columns:
            for value in df["Phase"].dropna():
                add(value, value)
                # "Lay I 21-40w" -> "Lay I" for free-text mentions.
                short = re.sub(r"\s+\d+\s*[-–]\s*\d+\s*w$", "", str(value)).strip()
                if short and short != str(value):
                    add(short, short)

    return tuple(sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True))


def _resolve_entity(text: str, attr_pos: int, max_distance: int) -> Optional[str]:
    """Resolve the nearest known entity alias to its canonical name."""
    before = text[max(0, attr_pos - max_distance):attr_pos].lower()
    for alias, canonical in _entity_lexicon():
        if alias in before:
            return canonical
    after = text[attr_pos:attr_pos + 60].lower()
    for alias, canonical in _entity_lexicon():
        if alias in after:
            return canonical
    return None


# ── Claim extraction ────────────────────────────────────────
def _segments(text: str) -> List[Tuple[int, int]]:
    """Split text on sentence/segment separators, returning absolute spans.

    Decimal points and digit-grouping commas must NOT split a number:
    "8.5%" and "3,350 kcal/kg" stay in one piece.
    """
    spans: List[Tuple[int, int]] = []
    start = 0
    for m in re.finditer(
        r"(?<=[.!?。！？])\s+|;|\n|(?<![0-9]),(?![0-9])\s*", text
    ):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def extract_claims(
    text: str,
    specialist: str,
    policy: Optional[ValidationPolicy] = None,
) -> List[Claim]:
    """Extract numeric nutritional claims with entity/attribute/unit context."""
    policy = policy or ValidationPolicy()
    if not text:
        return []

    claims: List[Claim] = []
    for seg_start, seg_end in _segments(text):
        segment = text[seg_start:seg_end]
        present = [
            key for key, rx in _ATTR_KEYWORD_RE.items() if rx.search(segment)
        ]
        # Ambiguous multi-attribute segment: skip rather than guess which value
        # belongs to which attribute (the old implementation guessed wrong).
        if len(present) != 1:
            continue

        occupied: List[Tuple[int, int]] = []
        for attribute, pattern in _COMPILED + _COMPILED_REVERSED:
            if not _attribute_matches_keyword(attribute, present[0]):
                continue
            for m in pattern.finditer(segment):
                seg_s, seg_e = m.span()
                # Specific phosphorus forms win over the generic "P" pattern;
                # skip any overlapping match so one value yields one claim.
                if any(seg_s < oe and seg_e > os_ for os_, oe in occupied):
                    continue
                raw_value = m.group(1).replace(",", "")
                try:
                    value = float(raw_value)
                except ValueError:
                    continue
                unit = m.group(2)
                try:
                    basis, _ = _canon_unit(unit)
                except KeyError:
                    continue
                abs_pos = seg_start + m.start()
                entity = _resolve_entity(text, abs_pos, policy.max_entity_distance)
                claims.append(
                    Claim(
                        specialist=specialist,
                        attribute=attribute,
                        value=value,
                        unit=unit,
                        basis=basis,
                        entity=entity,
                        text=m.group(0),
                    )
                )
                occupied.append((seg_s, seg_e))
    return claims


def _attribute_matches_keyword(attribute: str, keyword: str) -> bool:
    if attribute == keyword:
        return True
    if keyword == "phosphorus" and attribute in (
        "available_phosphorus",
        "total_phosphorus",
        "phosphorus",
    ):
        return True
    return False


# ── Comparison ──────────────────────────────────────────────
def _fmt(value: float, unit: str) -> str:
    if unit == "kcal/kg":
        return f"{value:,.0f} kcal/kg"
    return f"{value:.2f} {unit}"


def compare_claims(
    claims: Iterable[Claim],
    policy: Optional[ValidationPolicy] = None,
) -> ValidationResult:
    """Compare claims that share entity, attribute and basis."""
    policy = policy or ValidationPolicy()
    claims = list(claims)
    result = ValidationResult(claims=claims)

    groups: dict[Tuple[str, str, str], List[Claim]] = {}
    for c in claims:
        if c.entity is None:
            continue  # no confident match -> do not compare
        groups.setdefault((c.entity, c.attribute, c.basis), []).append(c)

    for (entity, attribute, basis), items in groups.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a.specialist == b.specialist:
                    continue
                result.compared_pairs += 1
                av, bv = a.canonical, b.canonical
                denom = max(abs(av), abs(bv), 1e-9)
                rel = abs(av - bv) / denom
                if rel > policy.relative_tolerance:
                    unit = "kcal/kg" if basis == "energy" else "%"
                    result.warnings.append(
                        f"Cross-specialist inconsistency for {entity} "
                        f"{attribute.replace('_', ' ')}: {a.specialist} reports "
                        f"{_fmt(av, unit)} vs {b.specialist} reports "
                        f"{_fmt(bv, unit)} (relative difference {rel:.1%})."
                    )

    if policy.ca_p_check:
        result.warnings.extend(_ca_p_warnings(claims, policy))

    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for w in result.warnings:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    result.warnings = unique
    return result


def _ca_p_warnings(claims: List[Claim], policy: ValidationPolicy) -> List[str]:
    """Opt-in Ca:P plausibility check.

    Only applied when calcium and phosphorus claims resolve to the *same*
    entity and the phosphorus basis is explicit. This avoids the old rule that
    flagged normal layer diets (Ca 4%, available P 0.4% -> 10:1).
    """
    by_entity: dict[str, dict[str, List[Claim]]] = {}
    for c in claims:
        if c.entity is None:
            continue
        by_entity.setdefault(c.entity, {}).setdefault(c.attribute, []).append(c)

    warnings = []
    for entity, attrs in by_entity.items():
        ca_claims = attrs.get("calcium", [])
        p_claims = attrs.get("available_phosphorus", []) + attrs.get(
            "total_phosphorus", []
        )
        if not ca_claims or not p_claims:
            continue
        ca = ca_claims[0]
        p = p_claims[0]
        if ca.basis != "percent" or p.basis != "percent":
            continue
        if p.canonical <= 0:
            continue
        ratio = ca.canonical / p.canonical
        lo, hi = policy.ca_p_range
        if ratio < lo or ratio > hi:
            warnings.append(
                f"Note: Ca:{p.attribute.replace('_', ' ')} ratio for {entity} "
                f"is {ratio:.1f}:1, outside the plausibility band "
                f"{lo:g}-{hi:g}:1."
            )
    return warnings


def validate_specialist_outputs(
    results: Iterable,
    policy: Optional[ValidationPolicy] = None,
) -> List[str]:
    """Validate a list of AgentResult-like objects (needs .agent_name/.response)."""
    policy = policy or ValidationPolicy()
    claims: List[Claim] = []
    for r in results:
        response = getattr(r, "response", "") or ""
        name = getattr(r, "agent_name", "unknown")
        claims.extend(extract_claims(response, name, policy))
    return compare_claims(claims, policy).warnings
