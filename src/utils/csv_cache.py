"""Shared CSV loading with module-level cache.

Prevents re-reading the same CSV file from disk on every tool invocation.
"""
import os
from typing import List, Optional
import pandas as pd

_df_cache: dict = {}


def load_csv(filepath: str, normalize_cols: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
    """Load a CSV file with module-level caching.

    Cache key includes file path and modification time so that changes
    to CSV files are picked up without restarting the process.

    Args:
        filepath: Path to CSV file.
        normalize_cols: Optional list of column names to strip+lowercase at cache time.
                        Avoids redundant normalization on every call.
    """
    if not filepath or not os.path.exists(filepath):
        return pd.DataFrame()

    stat = os.stat(filepath)
    norm_key = tuple(normalize_cols) if normalize_cols else None
    key = (filepath, stat.st_mtime_ns, norm_key)

    if key not in _df_cache:
        df = pd.read_csv(filepath, **kwargs)
        if normalize_cols:
            for col in normalize_cols:
                normalize_column(df, col)
        _df_cache[key] = df

    return _df_cache[key].copy()


def normalize_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Strip whitespace and lowercase a column in-place."""
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df
