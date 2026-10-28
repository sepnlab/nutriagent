"""Definitions Tool.

Provides access to 600+ poultry nutrition definitions and terminology.
"""
from typing import Type
import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.config import Config
from src.utils.csv_cache import load_csv


class DefinitionsInput(BaseModel):
    term: str = Field(description="The poultry nutrition term to look up (e.g., 'Broiler', 'Feed Conversion Ratio', 'Amino Acid')")


class DefinitionsTool(BaseTool):
    name: str = "definitions"
    description: str = (
        "Queries the poultry nutrition definitions database containing 600+ terms and concepts. "
        "Covers: feed ingredients, nutritional parameters, broiler nutrition principles, "
        "digestive physiology, poultry species, feed processing, feed evaluation, "
        "environmental management, health and disease, egg production, meat quality, "
        "metabolism, vitamins, minerals, additives, anti-nutritional factors, "
        "nutritional disorders, food safety, and emerging topics. "
        "Input: the term or concept to define."
    )
    args_schema: Type[BaseModel] = DefinitionsInput

    def _run(self, term: str) -> str:
        try:
            df = load_csv(Config.DEFINITIONS_FILE, normalize_cols=["Term"])
            query = term.strip().lower()

            # Exact match
            exact = df[df["Term"] == query]
            if not exact.empty:
                row = exact.iloc[0]
                return f"{row['Term']}: {row['Definition']}"

            # Partial match
            partial = df[df["Term"].str.contains(query, case=False, na=False, regex=False)]
            if not partial.empty:
                row = partial.iloc[0]
                return f"{row['Term']}: {row['Definition']}"

            # Fuzzy: score all terms by number of matching words, pick best
            words = [w for w in query.split() if len(w) > 3]
            if words:
                all_terms = df["Term"].tolist()
                scored = []
                for idx, t in enumerate(all_terms):
                    t_lower = t.lower()
                    score = sum(1 for w in words if w in t_lower)
                    # Bonus for exact substring match of full query
                    if query in t_lower:
                        score += 3
                    if score > 0:
                        scored.append((score, idx))
                if scored:
                    scored.sort(reverse=True)
                    row = df.iloc[scored[0][1]]
                    return f"{row['Term']}: {row['Definition']}"

            # No match found
            # Show similar terms
            all_terms = df["Term"].unique().tolist()
            scored = []
            for t in all_terms:
                common = len(set(query.split()) & set(t.split()))
                if common > 0:
                    scored.append((common, t))
            scored.sort(reverse=True)

            if scored:
                similar = [t for _, t in scored[:5]]
                return (f"Term '{term}' not found in definitions database. "
                        f"Similar terms: {', '.join(similar)}. "
                        f"Try a more specific search term.")
            else:
                return (f"Term '{term}' not found in definitions database. "
                        f"The database contains {len(all_terms)} poultry nutrition terms. "
                        f"Try searching for a related concept.")

        except Exception as e:
            return f"Error querying definitions: {e}"

    async def _arun(self, term: str) -> str:
        return self._run(term)
