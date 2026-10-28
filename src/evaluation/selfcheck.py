"""SelfCheckGPT-NLI cross-sample inconsistency detection for NutriCHAT.

Implements the SelfCheckGPT methodology using a cross-encoder NLI model
for sentence-level cross-sample consistency checking.

Reference: SelfCheckGPT (Manakul et al., 2023)
"""
import logging
import os
import re
import time
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Set HF mirror for users in China (fallback if not already set)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from src.config import Config
from src.retry import llm_retry


class SelfCheckGPT:
    """SelfCheckGPT-NLI cross-sample inconsistency detector.

    For each main response (T=0), generates N stochastic samples (T=1),
    then checks each sentence against sampled passages using NLI.

    The NLI model (CrossEncoder) is shared at class level so each instance does not load its own copy on the GPU.

    Args:
        device: NLI inference device
        batch_size: NLI batch size
        sample_model: model name used for T=1 sampling (None = use Config.SELFCHECK_MODEL)
        sample_base_url: sampling API base URL (None = use Config.SELFCHECK_BASE_URL)
        sample_api_key: sampling API key (None = use Config.SELFCHECK_API_KEY)
    """

    _shared_nli_model = None  # class-level shared GPU NLI model
    _shared_nli_device = None
    _sample_cache: dict = {}  # class-level sampling cache: (query, model, n) → samples
    _sample_cache_max = 2048

    def __init__(self, device: str = "cpu", batch_size: int = 32,
                 sample_model: Optional[str] = None,
                 sample_base_url: Optional[str] = None,
                 sample_api_key: Optional[str] = None,
                 sample_extra_body: Optional[dict] = None):
        self.device = device
        self.batch_size = batch_size
        self._nli_model = None
        self._sample_model = sample_model
        self._sample_base_url = sample_base_url
        self._sample_api_key = sample_api_key
        self._sample_extra_body = sample_extra_body
        self._sample_client = None  # cached OpenAI client for sample generation

    def _load_model(self):
        if self._nli_model is not None:
            return
        # Use the class-level shared model (loaded on the GPU only once)
        if SelfCheckGPT._shared_nli_model is not None and SelfCheckGPT._shared_nli_device == self.device:
            self._nli_model = SelfCheckGPT._shared_nli_model
            return

        import warnings
        t0 = time.time()
        from sentence_transformers import CrossEncoder

        model_candidates = [
            Config.NLI_MODEL_PATH,
            "cross-encoder/nli-deberta-v3-base",
            "cross-encoder/nli-MiniLM2-L6",
        ]

        for model_name in model_candidates:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    if os.path.isdir(str(model_name)):
                        logger.info("Loading local NLI model from %s...", model_name)
                    import torch as _torch
                    model = CrossEncoder(model_name, device=self.device, model_kwargs={"torch_dtype": _torch.float32})
                    logger.info("Loaded NLI model %s (%.1fs)", model_name, time.time() - t0)
                    # Cache at class level so other instances can reuse it
                    SelfCheckGPT._shared_nli_model = model
                    SelfCheckGPT._shared_nli_device = self.device
                    self._nli_model = model
                    return
                except Exception as e:
                    logger.warning("Could not load NLI model %s: %s", model_name, e)

        logger.warning("No NLI model available. SelfCheckGPT disabled.")
        self._nli_model = None

    def _split_sentences(self, text: str) -> List[str]:
        """Split response text into individual sentences."""
        text = re.sub(r"\s+", " ", text.strip())
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    @staticmethod
    def _concurrency_levels(n: int) -> List[int]:
        """Return adaptive concurrency levels for sample generation.

        Starts at up to 20 concurrent requests, falling back to 10 then 5
        when the API returns server errors.  Clamps each level to n so that
        small N doesn't waste threads.
        """
        return [min(n, 20), min(n, 10), min(n, 5)]

    def _generate_samples(self, query: str, n: int = 5) -> List[str]:
        """Generate N stochastic samples for a query.

        Uses the SelfCheckGPT model from config (temperature=1.0) to generate
        diverse completions for inconsistency detection.

        Instance-level sample_model/base_url/api_key can override the global Config settings,
        so different models can use their own sampling model (instead of all sharing one).

        Adaptive concurrency: start with 20 concurrent requests, then automatically
        degrade to 10 → 5 when the API returns server errors such as 503/429,
        avoiding extra load on an already overloaded API.

        Sampling results are cached at class level by (query, model, n), so the same
        query reuses the same samples across ablation variants and avoids duplicate API calls.
        """
        from openai import OpenAI

        sample_model = self._sample_model or Config.SELFCHECK_MODEL

        # Check class-level sample cache
        cache_key = (query, sample_model, n)
        if cache_key in SelfCheckGPT._sample_cache:
            return SelfCheckGPT._sample_cache[cache_key]
        sample_base_url = self._sample_base_url or Config.SELFCHECK_BASE_URL
        sample_api_key = self._sample_api_key or Config.SELFCHECK_API_KEY
        sample_extra_body = self._sample_extra_body or Config.SELFCHECK_EXTRA_BODY

        # Reuse cached client (avoids re-creating on every call)
        if self._sample_client is None:
            self._sample_client = OpenAI(
                base_url=sample_base_url,
                api_key=sample_api_key,
                timeout=Config.API_TIMEOUT,
            )
        client = self._sample_client

        all_samples = [None] * n
        pending = list(range(n))

        # Adaptive concurrency levels: start high, fall back on server errors
        for level in self._concurrency_levels(n):
            if not pending:
                break
            workers = min(len(pending), level)
            if workers == 0:
                break

            def _gen(idx):
                @llm_retry
                def _sample():
                    kwargs = dict(
                        model=sample_model,
                        messages=[
                            {"role": "system", "content": "You are a poultry nutrition assistant."},
                            {"role": "user", "content": query},
                        ],
                        temperature=1.0,
                        max_tokens=Config.SELFCHECK_MAX_TOKENS,
                    )
                    if sample_extra_body:
                        kwargs["extra_body"] = sample_extra_body
                    return client.chat.completions.create(**kwargs)
                resp = _sample()
                text = resp.choices[0].message.content or ""
                return text.strip() if text.strip() else None

            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_gen, i): i for i in pending}
                new_pending = []
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        text = future.result()
                        if text:
                            all_samples[i] = text
                        else:
                            new_pending.append(i)
                    except Exception as e:
                        logger.warning(
                            "Sample %d failed (concurrency=%d): %s",
                            i + 1, level, e,
                        )
                        new_pending.append(i)
                pending = new_pending

            if pending and level > min(n, 5):
                logger.info(
                    "Retrying %d/%d failed SelfCheck samples at lower concurrency (%d)...",
                    len(pending), n, min(n, level // 2),
                )

        if pending:
            logger.warning(
                "SelfCheck: %d/%d samples failed after adaptive retries",
                len(pending), n,
            )

        samples = [s for s in all_samples if s is not None]

        # Write to class-level cache (bounded size)
        if samples and len(SelfCheckGPT._sample_cache) < SelfCheckGPT._sample_cache_max:
            SelfCheckGPT._sample_cache[cache_key] = samples

        return samples

    def _batch_nli_predict(
        self, sentences: list, samples: list,
    ) -> List[List[Optional[float]]]:
        """Batch NLI inference for all sentence×sample pairs.

        Returns sentence_sample_probs[sent_idx][sample_idx] = contrad_prob or None.
        Uses a single batched predict() call instead of one pair at a time.
        """
        # Build flat list of (sample, sent) pairs, tracking positions
        pairs = []
        sent_indices: List[int] = []
        sample_indices: List[int] = []
        for si, sent in enumerate(sentences):
            for sj, sample in enumerate(samples):
                if sample.strip():
                    pairs.append((sample, sent))
                    sent_indices.append(si)
                    sample_indices.append(sj)

        # Initialize result matrix
        n_sents = len(sentences)
        n_samps = len(samples)
        probs: List[List[Optional[float]]] = [
            [None] * n_samps for _ in range(n_sents)
        ]

        if not pairs:
            return probs

        try:
            all_logits = self._nli_model.predict(pairs, batch_size=32)
            for k, logits in enumerate(all_logits):
                si, sj = sent_indices[k], sample_indices[k]
                ex = np.exp(logits - np.max(logits))
                pa = ex / ex.sum()
                probs[si][sj] = float(pa[0])  # contradiction prob
        except Exception as e:
            logger.warning("Batch NLI failed, falling back to per-pair: %s", e)
            # Fallback: per-pair inference
            for k, (sample, sent) in enumerate(pairs):
                si, sj = sent_indices[k], sample_indices[k]
                try:
                    logits = self._nli_model.predict([(sample, sent)])[0]
                    ex = np.exp(logits - np.max(logits))
                    pa = ex / ex.sum()
                    probs[si][sj] = float(pa[0])
                except Exception:
                    pass

        return probs

    def compute_score(
        self,
        main_response: str,
        query: str = "",
        samples: Optional[List[str]] = None,
        n_samples: int = 5,
    ) -> Optional[float]:
        """Compute SelfCheckGPT-NLI cross-sample inconsistency score for a response.

        Score ranges from 0.0 (highly consistent) to 1.0 (highly inconsistent).
        Returns None if NLI model is not available.

        Args:
            main_response: The deterministic response (T=0)
            query: Original query (used to generate samples if not provided)
            samples: Pre-generated stochastic samples (optional)
            n_samples: Number of samples to generate (if samples not provided)
        """
        self._load_model()

        if self._nli_model is None:
            logger.warning("SelfCheckGPT: no NLI model available, skipping (query: %.50s)", query)
            return None

        # Get or generate samples
        if not samples:
            if not query:
                return None
            samples = self._generate_samples(query, n=n_samples)

        if not samples:
            return None

        # Split main response into sentences
        sentences = self._split_sentences(main_response)
        if not sentences:
            return None

        # Batch NLI: all sentence×sample pairs in one call
        sent_sample_probs = self._batch_nli_predict(sentences, samples)

        sentence_scores = []
        for probs in sent_sample_probs:
            valid = [p for p in probs if p is not None]
            if valid:
                sentence_scores.append(sum(valid) / len(valid))

        if not sentence_scores:
            return 0.0

        return sum(sentence_scores) / len(sentence_scores)

    def compute_score_multi_n(
        self,
        main_response: str,
        query: str = "",
        n_values: Optional[List[int]] = None,
    ) -> Dict[int, Optional[float]]:
        """Compute inconsistency scores at multiple sample sizes in one pass.

        Generates max(n_values) stochastic samples once, then computes
        inconsistency scores at each specified N by averaging over the
        first N samples.  Avoids redundant API calls and NLI inference.

        Args:
            main_response: The deterministic response (T=0)
            query: Original query used to generate samples
            n_values: List of sample sizes, e.g. [5, 10, 20].
                      Defaults to [5, 10, 20].

        Returns:
            Dict[int, Optional[float]] — {n: inconsistency_score or None}
        """
        if n_values is None:
            n_values = [5, 10, 20]
        n_values = sorted(set(n_values))
        max_n = max(n_values)

        self._load_model()
        if self._nli_model is None:
            return {n: None for n in n_values}

        samples = self._generate_samples(query, n=max_n)
        if not samples:
            return {n: None for n in n_values}

        sentences = self._split_sentences(main_response)
        if not sentences:
            return {n: 0.0 for n in n_values}

        # Pre-compute NLI for all sentence × sample pairs (at max_n)
        sentence_sample_probs = self._batch_nli_predict(sentences, samples)

        # Score at each N: average first N probs per sentence, then average sentences
        results = {}
        for n in n_values:
            if n > len(samples):
                results[n] = None
                continue
            sentence_avg = []
            for probs in sentence_sample_probs:
                valid = [p for p in probs[:n] if p is not None]
                if valid:
                    sentence_avg.append(sum(valid) / len(valid))
            results[n] = sum(sentence_avg) / len(sentence_avg) if sentence_avg else 0.0

        return results
