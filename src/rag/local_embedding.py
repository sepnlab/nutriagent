"""Local embedding adapter for Qwen3-Embedding model.

Wraps the Sentence Transformers model as a LangChain-compatible embedding class,
supporting the same interface as OpenAIEmbeddings.

NOTE: the module-level _shared_model ensures only one copy of the model is loaded on the GPU
      when multiple instances run concurrently, preventing concurrent ablation variants from
      each loading an independent copy and exhausting GPU memory.
"""
import logging
import threading
from typing import List

logger = logging.getLogger(__name__)

# Module-level shared embedding model (protects GPU memory)
_shared_model = None
_shared_model_lock = threading.Lock()
_shared_model_key = ("", "")  # (model_path, device)


class Qwen3Embedding:
    """LangChain-compatible embedding wrapper for Qwen3-Embedding-0.6B.

    Uses sentence-transformers to load and run the local Qwen3 model.
    The GPU model is shared through a module-level singleton, avoiding duplicate copies under concurrency.
    Implements embed_query() and embed_documents() for LangChain integration.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        show_progress: bool = False,
        batch_size: int = 32,
    ):
        self.model_path = model_path
        self.device = device
        self.show_progress = show_progress
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        """Lazy-load the Sentence Transformer model with CPU fallback.

        Uses a module-level singleton: the same (model_path, device) is loaded only once,
        and later instances reuse the existing model instance.
        """
        if self._model is not None:
            return
        global _shared_model, _shared_model_key
        key = (self.model_path, self.device)
        if _shared_model is not None and _shared_model_key == key:
            self._model = _shared_model
            return
        with _shared_model_lock:
            if _shared_model is not None and _shared_model_key == key:
                self._model = _shared_model
                return
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading Qwen3 embedding model from {self.model_path}...")
                _shared_model = SentenceTransformer(
                    self.model_path,
                    device=self.device,
                )
                _shared_model_key = key
                self._model = _shared_model
            except Exception:
                if self.device != "cpu":
                    logger.warning(
                        "Failed to load on '%s', falling back to CPU: %s",
                        self.device, self.model_path,
                    )
                    try:
                        _shared_model = SentenceTransformer(self.model_path, device="cpu")
                        _shared_model_key = (self.model_path, "cpu")
                        self._model = _shared_model
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to load Qwen3 embedding model from {self.model_path}: {e}"
                        ) from e
                else:
                    raise RuntimeError(
                        f"Failed to load Qwen3 embedding model from {self.model_path}"
                    )
        dim = self._model.get_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {dim}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (passages)."""
        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query with the query-specific instruction prefix."""
        self._load_model()

        config = getattr(self._model, "prompts", {})
        if "query" in config:
            query_prefix = config["query"]
            prefixed_text = f"{query_prefix}{text}"
        else:
            prefixed_text = text

        embedding = self._model.encode(
            prefixed_text,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        self._load_model()
        return self._model.get_embedding_dimension()
