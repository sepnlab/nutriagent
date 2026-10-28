"""Nutrient Requirements Tool.

Provides nutrient specifications for:
- Broiler: Ross 308, Ross 708, Cobb 500 across feeding phases
- Layer: Hy-Line W-36, Hy-Line Brown, ISA Brown, Lohmann LSL-Classic across rearing & lay phases
"""
from typing import Optional, Type
import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.config import Config
from src.utils.csv_cache import load_csv


def _get_input_model() -> Type[BaseModel]:
    """Return Pydantic input schema with species-agnostic descriptions.

    Descriptions are kept generic so the schema works for any species
    without needing to be regenerated at import time.
    """
    class NutrientInput(BaseModel):
        breed: str = Field(description="Breed or genetic line name")
        phase: Optional[str] = Field(default=None, description="Feeding phase (e.g. 'Starter', 'Grower'). If None, returns all phases.")
        parameter: Optional[str] = Field(
            default=None,
            description="Specific nutrient parameter. Examples: 'ME_kcal_kg', 'CP_pct', 'Ca_pct', "
                        "'Avail_P_pct', 'Lys_pct'/'Lys_min_pct', 'Met_pct'/'Met_min_pct', "
                        "'SID_Lys_pct'. If None, returns all."
        )
    return NutrientInput


class NutrientRequirementsTool(BaseTool):
    name: str = "nutrient_requirements"
    description: str = ""
    args_schema: Type[BaseModel] = _get_input_model()
    species: str = Config.SPECIES

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.description = Config.get_species_handler().nutrient_tool_description()

    def _run(self, breed: str, phase: Optional[str] = None,
             parameter: Optional[str] = None) -> str:
        try:
            handler = Config.get_species_handler()
            breed_col = handler.nutrient_breed_column
            df = load_csv(Config.NUTRIENT_REQUIREMENTS_FILE, normalize_cols=[breed_col])
            query = breed.strip().lower()

            match = df[df[breed_col].str.contains(query, case=False, na=False, regex=False)]
            if match.empty:
                available = df[breed_col].unique().tolist()
                return f"Breed '{breed}' not found. Available: {available}"

            if phase:
                match = match[match["Phase"].str.contains(phase, case=False, na=False, regex=False)]
                if match.empty:
                    phases = df["Phase"].unique().tolist()
                    return f"Phase '{phase}' not found. Available: {phases}"

            # Load vitamin/mineral data
            vm_path = Config.NUTRIENT_VITAMIN_MINERAL_FILE
            vm_df = load_csv(vm_path, normalize_cols=[breed_col])
            vm_match = None
            if not vm_df.empty:
                vm_match = vm_df[vm_df[breed_col].str.contains(query, case=False, na=False, regex=False)]

            if len(match) == 1:
                return self._format_single_row(match.iloc[0], breed, breed_col, vm_match, phase)
            else:
                return self._format_multi_rows(match, breed, parameter)

        except Exception as e:
            return f"Error querying nutrient requirements: {e}"

    def _format_single_row(self, row, breed, breed_col, vm_match, phase) -> str:
        result = row.to_dict()

        if vm_match is not None and not vm_match.empty and len(vm_match) == 1 and phase:
            vm_phase_match = vm_match[vm_match["Phase"].str.contains(phase, case=False, na=False, regex=False)]
            if not vm_phase_match.empty:
                vm_row = vm_phase_match.iloc[0].to_dict()
                for k, v in vm_row.items():
                    if k not in result and k != breed_col:
                        result[f"VM_{k}"] = v

        lines = [
            f"Nutrient Requirements: {result.get(breed_col, breed)} | {result.get('Phase', '')}",
            "-" * 50,
            "MACRO NUTRIENTS:",
        ]
        for k in ["ME_kcal_kg", "CP_pct", "Ca_pct", "Avail_P_pct", "Na_pct", "Linoleic_Acid_pct"]:
            if k in result:
                lines.append(f"  {k}: {result[k]}")

        lines.append("\nAMINO ACIDS (%):")
        aa_keys = ["Lys_pct", "Lys_min_pct", "Met_pct", "Met_min_pct",
                   "M_C_pct", "M_C_min_pct", "Thr_pct", "Thr_min_pct",
                   "Trp_pct", "Trp_min_pct", "Val_pct", "Val_min_pct",
                   "Arg_pct", "Arg_min_pct", "Ile_pct", "Ile_min_pct"]
        for k in aa_keys:
            if k in result:
                lines.append(f"  {k}: {result[k]}")

        lines.append("\nSID AMINO ACIDS (%):")
        sid_keys = ["SID_Lys_pct", "SID_Met_pct", "SID_M_C_pct",
                    "SID_Thr_pct", "SID_Trp_pct", "SID_Val_pct"]
        for k in sid_keys:
            if k in result:
                lines.append(f"  {k}: {result[k]}")

        vm_keys = [k for k in result if k.startswith("VM_")]
        if vm_keys:
            lines.append("\nVITAMINS & MINERALS:")
            for k in vm_keys:
                lines.append(f"  {k.replace('VM_', '')}: {result[k]}")

        if "Choline_mg_kg" in result:
            lines.append(f"\n  Choline_mg_kg: {result['Choline_mg_kg']}")

        return "\n".join(lines)

    def _format_multi_rows(self, match, breed, parameter) -> str:
        if parameter:
            param = parameter.strip()
            lines = [f"Nutrient Requirements: {breed} ({parameter})"]
            lines.append("-" * 50)
            for _, row in match.iterrows():
                val = "N/A"
                for k in row.index:
                    if k.lower() == param.lower():
                        val = row[k]
                lines.append(f"  {row['Phase']}: {val}")
            return "\n".join(lines)

        lines = [f"Nutrient Requirements: {breed} (all phases)"]
        lines.append("-" * 50)
        for _, row in match.iterrows():
            lines.append(f"\n--- {row['Phase']} ---")
            for k in ["ME_kcal_kg", "CP_pct", "Ca_pct", "Avail_P_pct", "Lys_pct", "Lys_min_pct", "Met_pct", "Met_min_pct"]:
                if k in row and pd.notna(row[k]):
                    lines.append(f"  {k}: {row[k]}")
        return "\n".join(lines)

    async def _arun(self, breed: str, phase: Optional[str] = None,
                    parameter: Optional[str] = None) -> str:
        return self._run(breed, phase, parameter)
