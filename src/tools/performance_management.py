"""Performance Management Tool.

Provides production performance data for broiler or layer based on SPECIES config.
- Broiler: BW, ADG, FCR, mortality, processing yield (Ross 308/708, Cobb 500)
- Layer: Hen-day production, egg weight, egg mass, feed/dozen, shell quality, Haugh unit
  (Hy-Line W-36, Hy-Line Brown, ISA Brown, Lohmann LSL-Classic)
"""
from typing import Optional, Type
import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.config import Config
from src.utils.csv_cache import load_csv


def _get_input_model() -> Type[BaseModel]:
    """Return a Pydantic model accepting both day/week fields.

    Both age fields are optional; the tool selects the correct one
    based on the current species at runtime.
    """
    class PerformanceInput(BaseModel):
        breed: str = Field(description="Breed/genetic line name")
        day: Optional[int] = Field(default=None, description="Day of age (broiler)")
        week: Optional[int] = Field(default=None, description="Week of age (layer)")
        parameter: Optional[str] = Field(
            default=None,
            description="Specific parameter. If None, returns all."
        )
    return PerformanceInput


class PerformanceManagementTool(BaseTool):
    name: str = "performance_management"
    description: str = ""
    args_schema: Type[BaseModel] = _get_input_model()
    species: str = Config.SPECIES

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.description = Config.get_species_handler().performance_tool_description()

    def _run(self, breed: str, week: Optional[int] = None,
             day: Optional[int] = None, parameter: Optional[str] = None) -> str:
        try:
            handler = Config.get_species_handler()
            breed_col = handler.performance_breed_column
            age_col = handler.performance_age_column
            age_val = week if handler.key == "layer" else day

            df = load_csv(Config.PERFORMANCE_MANAGEMENT_FILE, normalize_cols=[breed_col])
            query = breed.strip().lower()

            match = df[df[breed_col].str.contains(query, case=False, na=False, regex=False)].copy()
            if match.empty:
                available = df[breed_col].unique().tolist()
                return f"Breed '{breed}' not found. Available: {available}"

            if age_val is not None:
                age_match = match[match[age_col] == age_val]
                if age_match.empty:
                    ages = sorted(match[age_col].unique())
                    nearest = min(ages, key=lambda a: abs(a - age_val))
                    return f"{age_col} {age_val} not found. Nearest: {nearest}"
                match = age_match

            # Merge processing yield data if available (broiler only)
            if handler.processing_params:
                try:
                    proc_df = load_csv(Config.PERFORMANCE_PROCESSING_FILE, normalize_cols=[breed_col])
                    proc_match = proc_df[proc_df[breed_col].str.contains(query, case=False, na=False, regex=False)]
                    if not proc_match.empty and age_val is not None:
                        proc_row = proc_match[proc_match[age_col] == age_val]
                        if not proc_row.empty:
                            for col in handler.processing_params:
                                if col in proc_row.columns:
                                    match.loc[match.index[0], col] = proc_row.iloc[0, proc_row.columns.get_loc(col)]
                except Exception:
                    pass  # processing data is optional

            if len(match) == 1:
                return self._format_single_row(match.iloc[0], breed, breed_col, age_col, handler)
            else:
                return self._format_multi_rows(match, breed, breed_col, age_col, handler)

        except Exception as e:
            return f"Error querying performance management: {e}"

    def _format_single_row(self, row, breed, breed_col, age_col, handler) -> str:
        result = row.to_dict()
        lines = []

        label = result.get(breed_col, breed)
        if handler.key == "layer":
            lines.append(f"Performance: {label} | Week {result.get(age_col, '')}")
        else:
            lines.append(f"Performance: {label} | Day {result.get(age_col, '')} | {result.get('Sex', '')}")
        lines.append("-" * 50)

        for k in handler.performance_params:
            if k in result:
                lines.append(f"  {k}: {result[k]}")

        proc_keys = handler.processing_params
        if proc_keys and any(k in result for k in proc_keys):
            lines.append("\nPROCESSING YIELD:")
            for k in proc_keys:
                if k in result:
                    lines.append(f"  {k}: {result[k]}")

        return "\n".join(lines)

    def _format_multi_rows(self, match, breed, breed_col, age_col, handler) -> str:
        species = handler.key

        key_lookup = {
            "layer": [0, 4, 8, 12, 16, 20, 24, 30, 40, 50, 60, 70, 80, 90, 100],
            "broiler": [0, 7, 14, 21, 28, 35, 42, 49, 56],
        }
        key_ages = key_lookup.get(species, [0])
        match = match[match[age_col].isin(key_ages)] if not match.empty else match

        lines = [f"Performance Data: {breed}"]
        lines.append("-" * 50)
        for _, row in match.iterrows():
            age = row.get(age_col, "?")
            if species == "layer":
                values = [
                    f"Prod={row.get('HenDay_Prod_Pct', 'N/A')}%",
                    f"EW={row.get('Egg_Wt_g', 'N/A')}g",
                    f"FI={row.get('FI_g_per_d', 'N/A')}g",
                ]
                lines.append(f"  {age_col} {age}: {', '.join(values)}")
            else:
                values = [
                    f"BW={row.get('BW_g', 'N/A')}g",
                    f"FCR={row.get('FCR', 'N/A')}",
                ]
                sex = row.get("Sex", "")
                lines.append(f"  {age_col} {age} ({sex}): {', '.join(values)}")

        return "\n".join(lines)

    async def _arun(self, breed: str, week: Optional[int] = None,
                    day: Optional[int] = None, parameter: Optional[str] = None) -> str:
        return self._run(breed, week, day, parameter)
