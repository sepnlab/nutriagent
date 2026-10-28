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

    NLI 模型（CrossEncoder）在类级别共享，避免每个实例在 GPU 上独立加载一份。

    Args:
        device: NLI 推理设备
        batch_size: NLI batch 大小
        sample_model: T=1 采样用的模型名（None=用 Config.SELFCHECK_MODEL）
        sample_base_url: 采样 API 地址（None=用 Config.SELFCHECK_BASE_URL）
        sample_api_key: 采样 API key（None=用 Config.SELFCHECK_API_KEY）
    """

    _shared_nli_model = None  # 类级别共享 GPU NLI 模型
    _shared_nli_device = None
    _sample_cache: dict = {}  # 类级别采样缓存：(query, model, n) → samples
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
        # 使用类级别共享模型（GPU 只加载一次）
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
                    # 缓存到类级别供其他实例复用
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

        可用实例级 sample_model/base_url/api_key 覆盖全局 Config 的设置，
        这样不同模型可以用各自对应的采样模型（而非全用同一个）。

        自适应并发：先尝试 20 并发，API 返回 503/429 等服务器错误时
        自动降级到 10 → 5，避免高并发加剧 API 过载。

        采样结果按 (query, model, n) 缓存到类级别，同一查询在不同
        消融变体中复用同一批采样，避免重复 API 调用。
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
