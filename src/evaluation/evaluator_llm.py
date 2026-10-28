"""EvaluatorLLM — LLM-based scoring for NutriCHAT assessment.

Implements the paper's EvaluatorLLM methodology:
1. Present all response variations for a query simultaneously
2. Identify the best-performing response for each metric (correctness, precision, depth)
3. Score the best response on a 5-point scale
4. Score remaining responses relative to the best
"""
import json
import logging
import os
import re
import threading
import time
from typing import Dict, List, Optional

from src.config import Config
from src.retry import llm_retry

logger = logging.getLogger(__name__)


EVALUATOR_PROMPT = """You are an expert evaluator of poultry nutrition responses. Evaluate {n} responses to the following question using the detailed rubrics below.

QUESTION: {query}

RESPONSES:
{responses}

Rate each response on three metrics using a 5-point scale with the following rubrics:

**Correctness** — Factual accuracy and relevance to the query.
  5 = All facts accurate, directly answers the query, cites/groups authoritative data correctly
  4 = Mostly accurate with minor imprecision or slightly incomplete
  3 = Partially correct but has noticeable factual errors or missing key information
  2 = Several factual errors or largely off-topic
  1 = Completely incorrect, hallucinated information, or irrelevant

**Precision** — Numerical accuracy. Set to null ONLY for definition/terminology queries.
  5 = Exact point estimate matching reference value; or narrow, tightly-grounded range
  4 = Point estimate close to reference; or moderately precise range
  3 = Estimate directionally correct but noticeably off; or overly wide range
  2 = Vague or significantly inaccurate numeric answer
  1 = No numeric answer when quantitative response is expected, or completely wrong value

**Scientific Depth** — Depth of domain knowledge, context, and explanatory quality.
  5 = Demonstrates thorough understanding, provides mechanistic explanation and practical context
  4 = Good domain knowledge with relevant context, minor depth missing
  3 = Adequate but surface-level; lacks mechanism or broader context
  2 = Superficial; minimal domain-specific insight beyond generic knowledge
  1 = No scientific content or factually misleading explanation

Respond with ONLY a JSON object containing a "scores" array:
{{"scores": [
    {{"response_id": "R1", "correctness": 5, "precision": 4, "scientific_depth": 5, "justification": "..."}},
    ...
]}}

IMPORTANT:
- The "scores" array must contain ALL {n} responses (R1 to R{n}).
- If precision is not applicable (definition/terminology only), set it to null.
- Compare each response against the others (best=5, slightly worse=4, etc.).
- No text before or after the JSON object."""

# Use the absolute-scoring prompt when n=1 (the ablation framework passes one response at a time, so comparative scoring is meaningless)
EVALUATOR_PROMPT_SINGLE = """You are an expert evaluator of poultry nutrition responses. Evaluate the following response to the question using the detailed rubrics below. Score each metric independently based on the rubric — do NOT compare with other responses.

QUESTION: {query}

RESPONSE:
{responses}

Rate the response on three metrics using a 5-point scale with the following rubrics:

**Correctness** — Factual accuracy and relevance to the query.
  5 = All facts accurate, directly answers the query, cites/groups authoritative data correctly
  4 = Mostly accurate with minor imprecision or slightly incomplete
  3 = Partially correct but has noticeable factual errors or missing key information
  2 = Several factual errors or largely off-topic
  1 = Completely incorrect, hallucinated information, or irrelevant

**Precision** — Numerical accuracy. Set to null ONLY for definition/terminology queries.
  5 = Exact point estimate matching reference value; or narrow, tightly-grounded range
  4 = Point estimate close to reference; or moderately precise range
  3 = Estimate directionally correct but noticeably off; or overly wide range
  2 = Vague or significantly inaccurate numeric answer
  1 = No numeric answer when quantitative response is expected, or completely wrong value

**Scientific Depth** — Depth of domain knowledge, context, and explanatory quality.
  5 = Demonstrates thorough understanding, provides mechanistic explanation and practical context
  4 = Good domain knowledge with relevant context, minor depth missing
  3 = Adequate but surface-level; lacks mechanism or broader context
  2 = Superficial; minimal domain-specific insight beyond generic knowledge
  1 = No scientific content or factually misleading explanation

IMPORTANT: Apply the rubric strictly. A score of 5 requires ALL criteria to be fully met. Most well-formed responses should score 3-4; reserve 5 for truly exceptional answers and 1-2 for clearly deficient ones.

Respond with ONLY a JSON object containing a "scores" array:
{{"scores": [
    {{"response_id": "R1", "correctness": 4, "precision": 4, "scientific_depth": 4, "justification": "..."}}
]}}

IMPORTANT:
- The "scores" array must contain exactly 1 entry for R1.
- If precision is not applicable (definition/terminology only), set it to null.
- Score based on the rubric above, not relative to other responses.
- No text before or after the JSON object."""

BULK_EVALUATOR_PROMPT = """You are an expert evaluator of poultry nutrition responses. Your task is to evaluate ALL {n} query-response pairs below using the detailed rubrics.

{items}

Rate each response on three metrics using a 5-point scale with the following rubrics:

**Correctness** — Factual accuracy and relevance to the query.
  5 = All facts accurate, directly answers the query, cites authoritative data
  4 = Mostly accurate with minor imprecision
  3 = Partially correct but has noticeable factual errors
  2 = Several factual errors or largely off-topic
  1 = Completely incorrect or hallucinated information

**Precision** — Numerical accuracy. Set to null ONLY for definition/terminology queries.
  5 = Exact point estimate matching reference; narrow precise range
  4 = Point estimate close to reference; moderately precise range
  3 = Directionally correct but noticeably off; overly wide range
  2 = Vague or significantly inaccurate numeric answer
  1 = No numeric answer when expected, or completely wrong value

**Scientific Depth** — Depth of domain knowledge and explanatory quality.
  5 = Thorough understanding with mechanistic explanation and practical context
  4 = Good domain knowledge with relevant context
  3 = Adequate but surface-level
  2 = Superficial, minimal domain insight
  1 = No scientific content or factually misleading

CRITICAL: Return ONLY valid JSON in this exact format, no other text:
{{
    "scores": [
        {{
            "query_id": "...",
            "correctness": 5,
            "precision": 4,
            "scientific_depth": 5,
            "justification": "..."
        }},
        ...
    ]
}}
The query_id must match the IDs given above exactly (1 per pair, in the same order)."""


class EvaluatorLLM:
    """LLM-based evaluation scorer for NutriCHAT responses.

    Uses SiliconFlow API (configurable model) to implement the paper's
    comparative evaluation methodology.

    Warns if EvaluatorLLM model is not clearly stronger than the NutriCHAT
    base model, since a weaker evaluator introduces systemic bias.

    When shuffle=True, randomizes response order in the prompt each call
    to mitigate position bias (primacy/recency effects).
    """

    shuffle: bool = False

    # Class-level scoring cache: hash(query+response) → scores dict
    # Shared across instances, so dimensions skipped by incremental ablation can reuse scores directly
    _score_cache: Dict[str, Dict] = {}
    _cache_lock: threading.Lock = threading.Lock()
    _cache_path: Optional[str] = None  # on-disk persistence path

    @classmethod
    def _load_cache(cls):
        if cls._cache_path and os.path.exists(cls._cache_path):
            try:
                with open(cls._cache_path, "r", encoding="utf-8") as f:
                    cls._score_cache = json.load(f)
                logger.info("Loaded %d cached scores from %s", len(cls._score_cache), cls._cache_path)
            except Exception as e:
                logger.warning("Failed to load evaluator cache: %s", e)

    @classmethod
    def _save_cache(cls):
        if cls._cache_path:
            os.makedirs(os.path.dirname(cls._cache_path), exist_ok=True)
            tmp = cls._cache_path + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(cls._score_cache, f)
                os.replace(tmp, cls._cache_path)
            except Exception as e:
                logger.warning("Failed to save evaluator cache: %s", e)

    def __init__(self, model: Optional[str] = None):
        self.model = model or Config.EVALUATOR_MODEL
        self._client = None
        self._check_model_capability()
        self.shuffle = self.__class__.shuffle
        self._load_cache()

    @staticmethod
    def _score_cache_key(query: str, response: str) -> str:
        import hashlib
        return hashlib.sha256(f"{query}|||{response}".encode()).hexdigest()[:16]

    @staticmethod
    def _normalize_model(name: str) -> str:
        """Normalize model name for capability comparison."""
        n = name.lower().strip()
        # Strip provider prefixes: "deepseek-ai/DeepSeek-V4-Flash" -> "deepseek-v4-flash"
        n = n.split("/")[-1] if "/" in n else n
        # Strip version suffixes like -Instruct, -Beta
        for suffix in ["-instruct", "-beta", "-stable"]:
            if n.endswith(suffix):
                n = n[: -len(suffix)]
        return n

    @classmethod
    def _model_flavor(cls, name: str) -> str:
        """Extract the flavor tier from a model name (e.g. 'flash', 'pro', 'ultra')."""
        n = cls._normalize_model(name)
        for tier in ["ultra", "pro", "flash", "lite", "mini"]:
            if tier in n:
                return tier
        return "unknown"

    def _check_model_capability(self):
        """Warn if evaluator model is not clearly more capable than the base model."""
        try:
            eval_norm = self._normalize_model(self.model)
            base_norm = self._normalize_model(Config.LLM_MODEL)
            eval_flavor = self._model_flavor(self.model)
            base_flavor = self._model_flavor(Config.LLM_MODEL)

            # Same normalized name -> same model
            if eval_norm == base_norm:
                logger.warning(
                    "EvaluatorLLM (%s) appears to be the same model as NutriCHAT base (%s). "
                    "This introduces evaluator bias — consider using a stronger evaluator model.",
                    self.model, Config.LLM_MODEL,
                )
            # Same flavor tier but different provider -> same capability tier
            elif eval_flavor == base_flavor and eval_flavor != "unknown":
                logger.info(
                    "EvaluatorLLM (%s, %s-tier) and NutriCHAT base (%s, %s-tier) are in the "
                    "same capability tier. Consider a stronger evaluator for unbiased scoring.",
                    self.model, eval_flavor, Config.LLM_MODEL, base_flavor,
                )
            elif eval_flavor == "flash" and base_flavor in ("pro", "ultra"):
                logger.warning(
                    "EvaluatorLLM (%s, flash-tier) is WEAKER than NutriCHAT base (%s, %s-tier)!"
                    " Scoring will be unreliable.",
                    self.model, Config.LLM_MODEL, base_flavor,
                )
        except Exception:
            pass  # non-critical

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=Config.EVALUATOR_BASE_URL,
                api_key=Config.EVALUATOR_API_KEY,
                timeout=120,
            )
        return self._client

    @property
    def _eval_extra_body(self) -> Optional[dict]:
        """Return extra_body for evaluator API calls.

        EvaluatorLLM uses temperature=0 and should NOT use deep thinking
        (it slows down scoring dramatically and adds no value for evaluation).
        """
        from src.config import _thinking_disabled_extra_body, _get_provider_config
        # Determine provider from base_url
        if "deepseek" in Config.EVALUATOR_BASE_URL:
            return _thinking_disabled_extra_body("deepseek")
        elif "siliconflow" in Config.EVALUATOR_BASE_URL:
            return _thinking_disabled_extra_body("siliconflow")
        elif "aliyun" in Config.EVALUATOR_BASE_URL:
            return _thinking_disabled_extra_body("aliyun")
        return None

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON content from text that may contain markdown fences or surrounding content.

        Handles:
        - ```json ... ``` blocks
        - ``` ... ``` blocks
        - Bare JSON objects
        - Text before/after the JSON object
        """
        import re
        # Try ```json or ``` block first
        code_block = re.search(
            r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL
        )
        if code_block:
            return code_block.group(1).strip()

        # Try to find a top-level JSON object {...}
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            return brace_match.group(0).strip()

        return text.strip()

    @staticmethod
    def _repair_json(text: str) -> str:
        """Repair common JSON formatting errors in LLM output, then retry parsing.

        Reasoning models such as DeepSeek occasionally produce JSON formatting issues:
        - trailing commas
        - keys in single quotes instead of double quotes
        - truncated brackets/quotes (when max_tokens is insufficient)
        - newlines inside string values
        """
        import re
        repaired = text.strip()
        # 1. Remove the outermost ``` markers (if _extract_json did not catch them)
        if repaired.startswith("```"):
            repaired = re.sub(r"^```(?:json)?\s*", "", repaired)
            repaired = re.sub(r"\s*```$", "", repaired)
        # 2. Remove trailing commas (a comma before ] or })
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        # 3. Convert single-quoted keys to double quotes (e.g. 'key': → "key":)
        repaired = re.sub(r"'([^']+)'\s*:", r'"\1":', repaired)
        # 4. Convert single-quoted string values to double quotes (e.g. : 'value' → : "value")
        #    Only handles the single quote after a colon to avoid damaging hyphens
        repaired = re.sub(r":\s*'([^']*)'\s*([,\}\]])", r': "\1"\2', repaired)
        # 5. Try truncating to the outermost closing } (drop extra truncated content)
        brace_depth = 0
        last_close = -1
        for i, ch in enumerate(repaired):
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    last_close = i
        if last_close > 0 and last_close < len(repaired) - 1:
            repaired = repaired[:last_close + 1]
        return repaired

    def score_responses(
        self, query: str, responses: Dict[str, str]
    ) -> Optional[Dict]:
        """Score multiple responses for a single query.

        Args:
            query: The original question
            responses: {response_id: response_text, ...}
                       e.g. {"NutriCHAT": "...", "GPT-4o": "...", ...}

        Returns:
            Dict with per-response correctness, precision, depth scores,
            or None if scoring failed.
        """
        # Optional shuffle to mitigate position bias
        items = list(responses.items())
        if self.shuffle:
            import random
            random.shuffle(items)

        # Build the prompt
        n = len(responses)
        resp_text = "\n\n".join([
            f"RESPONSE {i+1} ({rid}):\n{text[:2000]}"
            for i, (rid, text) in enumerate(items)
        ])

        # Use the absolute-scoring prompt when n=1 to avoid score inflation from comparative scoring without a reference
        prompt_template = EVALUATOR_PROMPT_SINGLE if n == 1 else EVALUATOR_PROMPT
        prompt = prompt_template.format(
            n=n,
            query=query,
            responses=resp_text,
        )

        content = ""
        try:
            _eb = self._eval_extra_body
            @llm_retry
            def _call():
                kwargs = dict(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=Config.EVALUATOR_MAX_TOKENS,
                )
                if _eb:
                    kwargs["extra_body"] = _eb
                return self.client.chat.completions.create(**kwargs)
            resp = _call()
            content = resp.choices[0].message.content.strip()
            if not content:
                logger.warning("Empty response from evaluator LLM")
                return None

            # Extract JSON from markdown code fences (```json ... ```)
            content = self._extract_json(content)
            if not content:
                logger.warning("No JSON content found in evaluator response")
                return None

            result = json.loads(content)
            logger.debug("Parsed evaluator result keys: %s", list(result.keys()))
            return self._parse_result(result, list(responses.keys()))
        except json.JSONDecodeError as e:
            logger.warning("JSON parse error: %s | raw: %s...", e, content[:200])
            # Try repairing and retrying once
            repaired = self._repair_json(content)
            try:
                result = json.loads(repaired)
                logger.info("Repaired JSON successfully")
                return self._parse_result(result, list(responses.keys()))
            except json.JSONDecodeError:
                logger.warning("JSON repair also failed")
                salvaged = self._salvage_scores(content)
                if salvaged:
                    logger.info("Salvaged %d score entries from truncated JSON", len(salvaged))
                    return self._parse_result({"scores": salvaged}, list(responses.keys()))
                return None
        except Exception as e:
            logger.error("EvaluatorLLM error: %s", e)
            return None

    @staticmethod
    def _salvage_scores(text: str) -> List[Dict]:
        """Recover metric fields from a truncated/malformed evaluator JSON.

        The rubric asks for ``justification`` after the three numeric fields, so a
        truncation usually leaves the scores intact. This is a fallback, not the
        primary parser; it never invents a value.
        """
        entries: List[Dict] = []
        for m in re.finditer(r'"(?:query_id|response_id)"\s*:\s*"([^"]+)"', text):
            rid = m.group(1)
            chunk = text[m.start():m.start() + 600]
            vals = {}
            for key in ("correctness", "precision", "scientific_depth"):
                km = re.search(rf'"{key}"\s*:\s*(\d+|null)', chunk)
                vals[key] = None if not km or km.group(1) == "null" else int(km.group(1))
            if all(v is None for v in vals.values()):
                continue
            entries.append({"query_id": rid, "response_id": rid, **vals})
        return entries

    def _parse_result(self, result: Dict, response_ids: List[str]) -> Dict:
        """Parse LLM output into a clean {response_id: {metric: score}} dict."""
        scores: Dict[str, Dict] = {}

        # Initialize all responses with null scores
        for rid in response_ids:
            scores[rid] = {
                "correctness": None,
                "precision": None,
                "scientific_depth": None,
                "justification": "",
            }

        # Build response_id mapping: LLM sees "RESPONSE {i+1} ({rid})"
        # and may return response_id as "{rid}" or "R{i+1}" or "RESPONSE {i+1}"
        rid_to_name = {}
        for i, name in enumerate(response_ids):
            rid_to_name[f"R{i+1}"] = name
            rid_to_name[f"RESPONSE {i+1}"] = name
            rid_to_name[f"response {i+1}"] = name
            rid_to_name[name] = name

        # Fill from scores (new format) or relative_scores (legacy)
        rel = result.get("scores") or result.get("relative_scores", [])
        for entry in rel:
            raw_rid = entry.get("response_id", "")
            actual_rid = rid_to_name.get(raw_rid, raw_rid)
            if actual_rid in scores:
                scores[actual_rid]["correctness"] = entry.get("correctness")
                scores[actual_rid]["precision"] = entry.get("precision")
                scores[actual_rid]["scientific_depth"] = entry.get("scientific_depth")
                scores[actual_rid]["justification"] = entry.get("justification", "")

        return scores

    def score_batch(
        self, queries: List[dict], all_responses: Dict[str, List[str]]
    ) -> Dict[str, List[Dict]]:
        """Score all responses for all queries.

        Args:
            queries: List of query dicts with "id", "query"
            all_responses: {response_id: [response_per_query, ...]}
                           All lists must have the same length as queries.

        Returns:
            {response_id: [{"correctness": ..., "precision": ..., ...}, ...]}
        """
        n = len(queries)
        results: Dict[str, List[Dict]] = {
            rid: [] for rid in all_responses
        }

        for i, q in enumerate(queries):
            qid = q.get("id", f"Q{i+1:03d}")
            logger.info("[%d/%d] %s: %s...", i + 1, n, qid, q.get('query', '')[:50])

            responses = {
                rid: texts[i] for rid, texts in all_responses.items()
            }

            t0 = time.time()
            scores = self.score_responses(q["query"], responses)
            elapsed = time.time() - t0

            if scores:
                logger.info("(%0.1fs)", elapsed)
            else:
                logger.warning("FAILED (%0.1fs)", elapsed)

            for rid in all_responses:
                if scores and rid in scores:
                    results[rid].append(scores[rid])
                else:
                    results[rid].append({
                        "correctness": None,
                        "precision": None,
                        "scientific_depth": None,
                        "justification": "",
                    })

        return results

    def score_bulk(
        self,
        queries: List[dict],
        responses: List[str],
        batch_size: int = 10,
    ) -> Optional[List[dict]]:
        """Score all query-response pairs in parallel LLM calls (batched).

        Args:
            queries: [{"id": ..., "query": ...}, ...]
            responses: [response_text, ...]  (same order as queries)
            batch_size: Queries per batch (default 10, reduces JSON size per call).

        Returns:
            List of {"query_id": ..., "correctness": ..., "precision": ...,
                     "scientific_depth": ..., "justification": ...}
            or None if scoring failed entirely.
        """
        if not queries or not responses or len(queries) != len(responses):
            logger.warning("score_bulk: mismatched queries(%d) and responses(%d)",
                           len(queries), len(responses))
            return None

        # Check cache for all pairs (thread-safe)
        all_scores: List[dict] = [None] * len(queries)
        uncached_indices: List[int] = []
        with self._cache_lock:
            for i, (q, r) in enumerate(zip(queries, responses)):
                key = self._score_cache_key(q.get("query", ""), r)
                if key in self._score_cache:
                    all_scores[i] = self._score_cache[key]
                else:
                    uncached_indices.append(i)

        if not uncached_indices:
            logger.info("score_bulk: all %d cached, skipping API", len(queries))
            return all_scores

        logger.info("score_bulk: %d cached, %d to score", len(queries) - len(uncached_indices), len(uncached_indices))

        # Only score uncached pairs
        uncached_queries = [queries[i] for i in uncached_indices]
        uncached_responses = [responses[i] for i in uncached_indices]
        n_batches = (len(uncached_queries) + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(uncached_queries))
            batch_queries = uncached_queries[start:end]
            batch_responses = uncached_responses[start:end]

            # Build prompt items
            items = []
            for i, (q, r) in enumerate(zip(batch_queries, batch_responses)):
                qid = q.get("id", f"Q{start + i + 1:03d}")
                query_text = q.get("query", "")
                items.append(
                    f"[Pair {i+1}]\n"
                    f"Query ID: {qid}\n"
                    f"Query: {query_text}\n"
                    f"Response: {r[:1500]}\n"
                )

            prompt = BULK_EVALUATOR_PROMPT.format(
                n=len(batch_queries),
                items="\n---\n".join(items),
            )

            logger.info("score_bulk batch %d/%d: %d queries", batch_idx + 1, n_batches, len(batch_queries))

            try:
                _eb = self._eval_extra_body
                @llm_retry
                def _call():
                    kwargs = dict(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=4000,
                    )
                    if _eb:
                        kwargs["extra_body"] = _eb
                    return self.client.chat.completions.create(**kwargs)
                resp = _call()
                content = resp.choices[0].message.content.strip()
                if not content:
                    logger.warning("score_bulk batch %d: empty response", batch_idx + 1)
                    continue

                # Extract JSON and parse
                json_str = self._extract_json(content)
                batch_scores = []
                if json_str:
                    try:
                        result = json.loads(json_str, strict=False)
                        batch_scores = result.get("scores", [])
                        logger.info("score_bulk batch %d: %d entries", batch_idx + 1, len(batch_scores))
                    except json.JSONDecodeError as e:
                        logger.warning("score_bulk batch %d: JSON error (%s), trying repair", batch_idx + 1, e)
                        repaired = self._repair_json(json_str)
                        try:
                            result = json.loads(repaired, strict=False)
                            batch_scores = result.get("scores", [])
                        except json.JSONDecodeError:
                            batch_scores = self._salvage_scores(json_str)
                            if batch_scores:
                                logger.info(
                                    "score_bulk batch %d: salvaged %d entries from truncated JSON",
                                    batch_idx + 1, len(batch_scores),
                                )

                # Map scores back to original indices by query_id (not array position,
                # since LLM may return scores in a different order or skip entries)
                id_to_score = {}
                for score in batch_scores:
                    qid = score.get("query_id", "")
                    if qid:
                        id_to_score[qid] = score

                with self._cache_lock:
                    for j in range(len(batch_queries)):
                        qid = batch_queries[j].get("id", f"Q{start + j + 1:03d}")
                        score = id_to_score.get(qid)
                        if score is not None:
                            orig_idx = uncached_indices[start + j]
                            all_scores[orig_idx] = score
                            # Cache
                            q = uncached_queries[start + j]
                            r = uncached_responses[start + j]
                            key = self._score_cache_key(q.get("query", ""), r)
                            self._score_cache[key] = score

            except Exception as e:
                logger.error("score_bulk batch %d failed: %s", batch_idx + 1, e)

        scored_count = sum(1 for s in all_scores if s is not None)
        if scored_count == 0:
            return None

        # Fill uncached entries with null scores
        for i in range(len(all_scores)):
            if all_scores[i] is None:
                all_scores[i] = {
                    "query_id": queries[i].get("id", f"Q{i+1:03d}"),
                    "correctness": None, "precision": None,
                    "scientific_depth": None, "justification": "",
                }

        logger.info("score_bulk total: %d/%d scored (%d cached)", scored_count, len(queries), len(queries) - len(uncached_indices))
        self._save_cache()
        return all_scores
