"""Feed Ingredient Bank Tool.

Provides access to nutritional composition data for ~100 feed ingredients.
Integrates 12,100 nutritional parameters covering composition, digestibility,
and energy values from trusted industry sources.
"""
from typing import Optional, Type
import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.config import Config
from src.utils.csv_cache import load_csv


class FeedIngredientBankInput(BaseModel):
    ingredient: str = Field(description="Name of the feed ingredient to query (e.g., 'Corn Grain', 'Soybean Meal 48%')")
    parameter: Optional[str] = Field(
        default=None,
        description="Specific nutritional parameter to retrieve (e.g., 'CP', 'ME_kcal_kg', 'Ca', 'Lys'). "
                    "If None, returns all available parameters."
    )


class FeedIngredientBankTool(BaseTool):
    name: str = "feed_ingredient_bank"
    description: str = (
        "Queries the Feed Ingredient Bank database for nutritional composition data of feed ingredients. "
        "Covers 77 feedstuffs with parameters including: DM (dry matter %), CP (crude protein %), "
        "EE (ether extract %), ME_kcal_kg (metabolizable energy), Ca (calcium %), P_total (total phosphorus %), "
        "Lys (lysine %), Met (methionine %), Thr (threonine %), Trp (tryptophan %), Val (valine %), "
        "CFiber (crude fiber %), Na (sodium %), K (potassium %), "
        "SID_Lys, SID_Met, SID_Thr (standardized ileal digestibility coefficients). "
        "Also includes amino acid profiles and digestibility data. "
        "Input: ingredient name and optional specific parameter."
    )
    args_schema: Type[BaseModel] = FeedIngredientBankInput

    def _run(self, ingredient: str, parameter: Optional[str] = None) -> str:
        try:
            # Load main composition data
            comp_df = load_csv(Config.FEED_INGREDIENT_BANK_FILE, normalize_cols=["Ingredient"])
            ing = ingredient.strip().lower()

            # Find matching ingredient (partial match)
            match = comp_df[comp_df["Ingredient"].str.contains(ing, case=False, na=False, regex=False)]
            if match.empty:
                similar = comp_df[comp_df["Ingredient"].str.contains(
                    ing.split()[0] if " " in ing else ing[:4], case=False, na=False, regex=False
                )]
                if similar.empty:
                    available = comp_df["Ingredient"].unique().tolist()
                    return f"Ingredient '{ingredient}' not found. Available ingredients include: {', '.join(available[:15])}..."
                match = similar

            row = match.iloc[0]
            result = row.to_dict()
            result["Ingredient"] = row.get("Ingredient", ingredient)

            # Load amino acid profile
            aa_df = load_csv(Config.FEED_INGREDIENT_AA_FILE, normalize_cols=["Ingredient"])
            aa_match = aa_df[aa_df["Ingredient"].str.contains(ing, case=False, na=False, regex=False)]
            if not aa_match.empty:
                aa_row = aa_match.iloc[0].to_dict()
                # Add amino acid data, skipping DM/CP/Ingredient already in main
                for k, v in aa_row.items():
                    if k not in result and k != "Ingredient":
                        result[k] = v

            # Load digestibility data
            dig_df = load_csv(Config.FEED_INGREDIENT_DIGESTIBILITY_FILE, normalize_cols=["Ingredient"])
            dig_match = dig_df[dig_df["Ingredient"].str.contains(ing, case=False, na=False, regex=False)]
            if not dig_match.empty:
                dig_row = dig_match.iloc[0].to_dict()
                for k, v in dig_row.items():
                    if k not in result and k != "Ingredient":
                        result[f"{k}"] = v

            if parameter:
                param = parameter.strip()
                # Check exact match
                if param in result:
                    return f"{result['Ingredient']} → {param}: {result[param]}"
                # Check case-insensitive match
                for k, v in result.items():
                    if k.lower() == param.lower():
                        return f"{result['Ingredient']} → {k}: {v}"
                return f"Parameter '{parameter}' not found for {ingredient}. Available: {', '.join(result.keys())}"

            # Format output nicely
            lines = [f"Feed Ingredient: {result.get('Ingredient', ingredient)}"]
            lines.append("-" * 40)
            lines.append("COMPOSITION:")
            comp_keys = ["DM", "CP", "EE", "ME_kcal_kg", "Ca", "P_total", "CFiber", "Na", "K"]
            for k in comp_keys:
                if k in result:
                    lines.append(f"  {k}: {result[k]}")
            lines.append("\nAMINO ACIDS (% of ingredient):")
            aa_keys = ["Arg", "His", "Ile", "Leu", "Lys", "Met", "Cys", "Phe", "Tyr", "Thr", "Trp", "Val"]
            for k in aa_keys:
                if k in result:
                    lines.append(f"  {k}: {result[k]}%")
            lines.append("\nDIGESTIBILITY (%):")
            dig_keys = ["DM_dig", "CP_dig", "EE_dig", "Ca_dig", "P_dig"]
            for k in dig_keys:
                if k in result:
                    lines.append(f"  {k}: {result[k]}%")
            lines.append(f"\nSID COEFFICIENTS (%):")
            # Old format: SID_Lys / SID_Met / SID_Thr
            # New format: Lys_sid / Met_sid / ... (20 AAs)
            sid_keys_old = ["SID_Lys", "SID_Met", "SID_Thr"]
            sid_keys_new = ["Lys_sid", "Met_sid", "Thr_sid", "Trp_sid", "Val_sid",
                            "Arg_sid", "His_sid", "Ile_sid", "Leu_sid", "Phe_sid",
                            "Tyr_sid", "Cys_sid", "MetCys_sid", "PheTyr_sid",
                            "Gly_sid", "Ser_sid", "Pro_sid", "Ala_sid", "Asp_sid", "Glu_sid"]
            for k in sid_keys_old + sid_keys_new:
                if k in result:
                    lines.append(f"  {k}: {result[k]}%")

            return "\n".join(lines)

        except Exception as e:
            return f"Error querying feed ingredient bank: {e}"

    async def _arun(self, ingredient: str, parameter: Optional[str] = None) -> str:
        return self._run(ingredient, parameter)
