"""API call retry utilities (exponential backoff + adaptive rate limiting).

Retry parameters are loaded from config/settings.yaml:
  api.retry.max_attempts  — maximum number of retries
  api.retry.min_wait      — minimum backoff wait (seconds)
  api.retry.max_wait      — maximum backoff wait (seconds)

Usage:
    from src.retry import llm_retry

    @llm_retry
    def call_api():
        ...
"""
import threading
import time
from typing import Callable

from openai import APIError, APITimeoutError, RateLimitError


class _RateLimiter:
    """Global API rate limiter: caps concurrent requests + minimum request interval."""

    def __init__(self, max_concurrent: int = 6, min_interval: float = 0.15):
        self._semaphore = threading.Semaphore(max_concurrent)
        self._min_interval = min_interval
        self._last_call = 0.0
        self._lock = threading.Lock()

    def acquire(self):
        self._semaphore.acquire()
        with self._lock:
            elapsed = time.time() - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.time()

    def release(self):
        self._semaphore.release()


# Global rate limiter (6 concurrent, 150 ms interval, tuned for SiliconFlow rate limits)
_rate_limiter = _RateLimiter(max_concurrent=6, min_interval=0.15)


def _make_retry(max_attempts: int = 3, min_wait: int = 2, max_wait: int = 30,
                exceptions: tuple = None):
    """Create a tenacity retry decorator with rate limiting.

    Args:
        exceptions: Exception types to retry on. Defaults to OpenAI exceptions.
    """
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        before_sleep_log,
    )
    import logging
    if exceptions is None:
        exceptions = (APIError, APITimeoutError, RateLimitError, ConnectionError)

    def _decorator(func):
        _retryed = retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(exceptions),
            reraise=True,
            before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        )(func)

        def _wrapper(*args, **kwargs):
            _rate_limiter.acquire()
            try:
                return _retryed(*args, **kwargs)
            finally:
                _rate_limiter.release()

        return _wrapper

    return _decorator


_retry_init_lock = threading.Lock()


def _get_llm_retry():
    """Get the llm_retry decorator, configured from settings.yaml via Config.

    Note: caller (llm_retry) already holds _retry_init_lock, so do NOT
    re-acquire here — threading.Lock() is not reentrant.
    """
    try:
        from src.config import Config
        # Update rate limiter to match config
        _rate_limiter._semaphore = threading.Semaphore(Config.API_MAX_CONCURRENT)
        return _make_retry(
            max_attempts=Config.API_RETRY_MAX_ATTEMPTS,
            min_wait=Config.API_RETRY_MIN_WAIT,
            max_wait=Config.API_RETRY_MAX_WAIT,
        )
    except Exception:
        return _make_retry()


# Default retry decorator (lazily loads Config to avoid an import cycle)
_cached_retry_decorator = None


def llm_retry(func: Callable = None, **kwargs):
    global _cached_retry_decorator
    if _cached_retry_decorator is None:
        with _retry_init_lock:
            if _cached_retry_decorator is None:
                _cached_retry_decorator = _get_llm_retry()
    if func is not None:
        return _cached_retry_decorator(func)
    return _cached_retry_decorator


# More conservative retry configuration (for non-critical paths such as sampling)
def _get_aggressive_retry():
    try:
        from src.config import Config
        max_attempts = max(Config.API_RETRY_MAX_ATTEMPTS, 5)
        return _make_retry(max_attempts=max_attempts, min_wait=1, max_wait=60)
    except Exception:
        return _make_retry(max_attempts=5, min_wait=1, max_wait=60)


_cached_aggressive_decorator = None


def llm_retry_aggressive(func: Callable = None, **kwargs):
    global _cached_aggressive_decorator
    if _cached_aggressive_decorator is None:
        with _retry_init_lock:
            if _cached_aggressive_decorator is None:
                _cached_aggressive_decorator = _get_aggressive_retry()
    if func is not None:
        return _cached_aggressive_decorator(func)
    return _cached_aggressive_decorator
