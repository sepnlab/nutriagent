"""PoultryDB - Vector knowledge base for NutriCHAT RAG component.

Builds and queries a ChromaDB vector store from the four expert tool datasets.
Converts structured CSV data into natural language text documents for semantic retrieval.
"""
import hashlib
import json
import logging
import os
from typing import List, Optional
import pandas as pd
from langchain_chroma import Chroma

from src.config import Config

logger = logging.getLogger(__name__)

_EMBEDDING_CACHE: dict = {}


def create_embedding_model():
    """Create the appropriate embedding model based on config.

    Supported providers:
    - "local": Qwen3-Embedding-0.6B via sentence-transformers (default, no API key)
    - "siliconflow": Qwen/Qwen3-Embedding-8B or 4B via SiliconFlow API
    """
    from langchain_openai import OpenAIEmbeddings

    global _EMBEDDING_CACHE
    cache_key = Config.EMBEDDING_PROVIDER.lower()
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]

    provider = cache_key

    if provider == "local":
        from src.rag.local_embedding import Qwen3Embedding

        logger.info(f"Using local model: {Config.LOCAL_EMBEDDING_PATH}")
        model = Qwen3Embedding(
            model_path=Config.LOCAL_EMBEDDING_PATH,
            device=Config.LOCAL_EMBEDDING_DEVICE,
        )
        _EMBEDDING_CACHE[cache_key] = model
        return model

    if provider == "siliconflow":
        model_name = Config.SILICONFLOW_EMBEDDING_MODEL
        logger.info(f"Using SiliconFlow: {model_name}")
        model = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=Config.LLM_API_KEY,
            openai_api_base=f"{Config.LLM_BASE_URL.rstrip('/')}",
        )
        _EMBEDDING_CACHE[cache_key] = model
        return model

    raise ValueError(
        f"Unknown embedding provider: '{provider}'. "
        f"Supported: 'local', 'siliconflow'. "
        f"Set EMBEDDING_PROVIDER in .env"
    )


class PoultryDB:
    """PoultryDB vector store for semantic knowledge retrieval.

    Built from all four tool datasets, converted to natural language text,
    embedded and stored in ChromaDB. Supports OpenAI and local embedding providers.
    """

    def __init__(self, persist_dir: Optional[str] = None, embedding_model=None):
        self.persist_dir = persist_dir or Config.CHROMA_PERSIST_DIR
        self.embedding_model = embedding_model or create_embedding_model()
        self.collection_name = Config.CHROMA_COLLECTION_NAME
        self._db = None
        self._cache_dir = Config.CACHE_DIR
        from collections import OrderedDict
        self._search_cache: OrderedDict = OrderedDict()  # (query, k) -> results (LRU)
        self._cache_max = 512

    def _data_source_files(self) -> List[str]:
        """List of all CSV files that feed into the vector DB."""
        return [
            Config.FEED_INGREDIENT_BANK_FILE,
            Config.FEED_INGREDIENT_AA_FILE,
            Config.FEED_INGREDIENT_DIGESTIBILITY_FILE,
            Config.PERFORMANCE_MANAGEMENT_FILE,
            Config.NUTRIENT_REQUIREMENTS_FILE,
            Config.NUTRIENT_VITAMIN_MINERAL_FILE,
            Config.DEFINITIONS_FILE,
        ]

    def _compute_file_hashes(self) -> dict:
        """MD5 hash of each source CSV for change detection."""
        hashes = {}
        for path in self._data_source_files():
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    hashes[path] = hashlib.sha256(f.read()).hexdigest()
        return hashes

    def _load_saved_hashes(self) -> dict:
        """Load previously saved hashes from .cache."""
        path = os.path.join(self._cache_dir, f"db_hashes_{Config.SPECIES}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def _save_hashes(self, hashes: dict):
        """Persist current hashes for future comparison."""
        os.makedirs(self._cache_dir, exist_ok=True)
        path = os.path.join(self._cache_dir, f"db_hashes_{Config.SPECIES}.json")
        with open(path, "w") as f:
            json.dump(hashes, f, indent=2)

    def _load_and_merge_ingredient_csvs(self) -> Optional[pd.DataFrame]:
        """Load and merge feed_ingredient_bank, amino_acids, and digestibility CSVs on 'Ingredient'.

        Returns a single DataFrame with all columns from all three files,
        or None if the files cannot be loaded.
        """
        paths = [
            (Config.FEED_INGREDIENT_BANK_FILE, "bank"),
            (Config.FEED_INGREDIENT_AA_FILE, "aa"),
            (Config.FEED_INGREDIENT_DIGESTIBILITY_FILE, "dig"),
        ]
        dfs = {}
        for fp, tag in paths:
            if not fp or not os.path.exists(fp):
                logger.warning(f"Ingredient CSV missing: {fp}")
                continue
            df = pd.read_csv(fp)
            dfs[tag] = df
        if not dfs:
            return None
        merged = None
        for tag, df in dfs.items():
            if merged is None:
                merged = df
            else:
                merged = pd.merge(merged, df, on="Ingredient", how="outer", suffixes=(None, f"_{tag}"))
        merged = merged.loc[:, ~merged.columns.str.endswith(("_aa", "_dig", "_bank"))]
        return merged

    def _get_natural_language_text(self, df: pd.DataFrame, source: str) -> List[str]:
        """Convert DataFrame rows into natural language text documents.

        Matches the paper's methodology: CSV rows are converted to text format,
        e.g., 'Ingredient: Barley Grain, Magnesium: 0.13%' → 'Ingredient: Barley\n Magnesium: 0.13%'
        """
        docs = []
        for _, row in df.iterrows():
            parts = [f"Source: {source}"]
            for col, val in row.items():
                if pd.notna(val):
                    parts.append(f"{col}: {val}")
            docs.append("\n".join(parts))
        return docs

    def _get_metadata(self, df: pd.DataFrame, source: str) -> List[dict]:
        """Generate metadata for each document."""
        metadata_list = []
        for _, row in df.iterrows():
            meta = {"source": source}
            # Add key identifying fields as metadata
            if "Ingredient" in row:
                meta["ingredient"] = str(row["Ingredient"])
            if "Genetic_Line" in row:
                meta["genetic_line"] = str(row["Genetic_Line"])
            if "Breed" in row:
                meta["breed"] = str(row["Breed"])
            if "Phase" in row:
                meta["phase"] = str(row["Phase"])
            if "Term" in row:
                meta["term"] = str(row["Term"])
            if "Sex" in row:
                meta["sex"] = str(row["Sex"])
            if "Day" in row and pd.notna(row["Day"]):
                meta["day"] = int(row["Day"])
            if "Week" in row and pd.notna(row["Week"]):
                meta["week"] = int(row["Week"])
            metadata_list.append(meta)
        return metadata_list

    def _check_hash_change(self) -> bool:
        """Check if CSV files changed since last build. Returns True if changed."""
        current = self._compute_file_hashes()
        saved = self._load_saved_hashes()
        if not saved:
            return False  # first build, nothing to compare
        return current != saved

    def build(self, force_rebuild: bool = False) -> int:
        """Build PoultryDB from all tool datasets.

        If CSV data changed since last build and force_rebuild is False,
        logs a warning suggesting --force-db.

        Returns total document count.
        """
        # Check if already built
        if os.path.exists(os.path.join(self.persist_dir, "chroma.sqlite3")) and not force_rebuild:
            if self._check_hash_change():
                logger.warning(
                    "CSV data files changed since last PoultryDB build. "
                    "Run with --force-db to rebuild."
                )
            self._db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.persist_dir,
            )
            count = self._db._collection.count()
            logger.info(
                "Loaded existing PoultryDB (%d docs at %s)", count, self.persist_dir
            )
            return count

        logger.info("Building PoultryDB from tool datasets...")
        self._search_cache.clear()  # invalidate cache on rebuild

        # Load and convert each dataset
        all_docs = []
        all_metadatas = []
        all_ids = []
        doc_id = 0

        sources_config = [
            (None, "Feed Ingredient Bank", True),  # merged from 3 ingredient CSVs
            (Config.PERFORMANCE_MANAGEMENT_FILE, "Performance Management", False),
            (Config.NUTRIENT_REQUIREMENTS_FILE, "Nutrient Requirements", False),
            (Config.NUTRIENT_VITAMIN_MINERAL_FILE, "Nutrient Requirements", False),
            (Config.DEFINITIONS_FILE, "Definitions", False),
        ]

        for filepath, source_name, is_merged in sources_config:
            if is_merged:
                merged_df = self._load_and_merge_ingredient_csvs()
                if merged_df is None:
                    logger.warning("Merged ingredient CSV not available")
                    continue
                docs = self._get_natural_language_text(merged_df, source_name)
                metadatas = self._get_metadata(merged_df, source_name)
                count = len(docs)
            else:
                if not filepath or not os.path.exists(filepath):
                    logger.warning(f"File not found: {filepath}")
                    continue
                try:
                    df = pd.read_csv(filepath)
                    docs = self._get_natural_language_text(df, source_name)
                    metadatas = self._get_metadata(df, source_name)
                    count = len(docs)
                except Exception as e:
                    logger.error(f"{source_name}: {e}")
                    continue
            ids = [f"doc_{source_name.replace(' ', '_')}_{i}" for i in range(doc_id, doc_id + count)]
            all_docs.extend(docs)
            all_metadatas.extend(metadatas)
            all_ids.extend(ids)
            doc_id += count
            logger.info(f"{source_name}: {count} documents")

        # Build ChromaDB
        if all_docs:
            # Remove old database if exists
            if os.path.exists(self.persist_dir) and force_rebuild:
                self._db = None  # release old client before deleting files
                import shutil
                shutil.rmtree(self.persist_dir)

            os.makedirs(self.persist_dir, exist_ok=True)

            self._db = Chroma.from_texts(
                texts=all_docs,
                embedding=self.embedding_model,
                metadatas=all_metadatas,
                ids=all_ids,
                collection_name=self.collection_name,
                persist_directory=self.persist_dir,
            )

            self._save_hashes(self._compute_file_hashes())
            logger.info(
                "PoultryDB built with %d docs at %s", len(all_docs), self.persist_dir
            )
            return len(all_docs)
        else:
            logger.warning("PoultryDB: No documents loaded!")
            return 0

    def _ensure_loaded(self):
        """Ensure the database is loaded."""
        if self._db is not None:
            return
        if os.path.exists(os.path.join(self.persist_dir, "chroma.sqlite3")):
            self._db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.persist_dir,
            )
        else:
            self.build()

    def search(self, query: str, k: Optional[int] = None) -> List[dict]:
        """Search PoultryDB for documents relevant to the query.

        Uses cosine distance (via Chroma's default) for text embedding similarity.
        Returns top-k nearest neighbor documents. Results are cached per (query, k).
        """
        self._ensure_loaded()

        if not self._db:
            return [{"error": "PoultryDB not built yet. Call build() first."}]

        k = k if k is not None else Config.RAG_K

        # Check cache
        cache_key = (query, k)
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        results = self._db.similarity_search_with_score(query, k=k)
        output = []
        for doc, score in results:
            output.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "similarity_score": float(score),
            })

        # Cache result with LRU eviction
        if len(self._search_cache) >= self._cache_max:
            self._search_cache.popitem(last=False)  # evict oldest
        self._search_cache[cache_key] = output
        return output

    def get_retrieved_domain(self, query: str) -> str:
        """Get the knowledge domain of the most relevant document for a query."""
        results = self.search(query, k=Config.RAG_K)
        if results and "source" in results[0]:
            return results[0]["source"]
        return "Unknown"

    @property
    def document_count(self) -> int:
        """Get total document count in the database."""
        self._ensure_loaded()
        if self._db:
            return self._db._collection.count()
        return 0
