"""Broiler (meat-type chicken) handler."""

from src.species.base import SpeciesHandler


class BroilerHandler(SpeciesHandler):
    key = "broiler"

    # ── 元数据 ──────────────────────────────────────────────
    display_name = "Broiler (meat-type chicken)"
    breeds = (
        "Ross 308, Ross 708, Cobb 500, "
        "Shengze 901, Guangming 2, Wode 188, "
        "Qingyuan Partridge, Sanhuang, Crossbred 817"
    )
    phases = "Starter → Grower → Finisher → Withdrawal (conventional); extended cycles (63-90d) for yellow-feathered"
    metrics = "Body weight, ADG, FCR, feed intake, mortality, processing yield, breast meat, feather color, meat quality"

    # ── 性能管理 ────────────────────────────────────────────
    performance_breed_column = "Genetic_Line"
    performance_age_column = "Day"
    performance_age_field = "day"
    performance_age_desc = "Day (0-56)"

    @property
    def performance_params(self) -> dict:
        return {
            "BW_g": "Body weight (g)",
            "ADG_g": "Average daily gain (g/d)",
            "FI_g_per_day": "Feed intake per day (g)",
            "Cumulative_FI_g": "Cumulative feed intake (g)",
            "FCR": "Feed conversion ratio",
            "Cumulative_FCR": "Cumulative feed conversion ratio",
            "Mortality_Pct": "Mortality percentage",
        }

    @property
    def processing_params(self) -> list:
        return [
            "Breast_Meat_Yield_Pct", "Fat_Pad_Pct",
            "Leg_Yield_Pct", "Wing_Yield_Pct", "Processing_Weight_g",
        ]

    def performance_tool_description(self) -> str:
        return (
            "Queries broiler performance data for multiple genetic lines including Western breeds "
            "(Ross 308, Ross 708, Cobb 500) and Chinese domestic breeds "
            "(Shengze 901, Guangming 2, Wode 188, Qingyuan Partridge, Sanhuang, Crossbred 817). "
            "Parameters include BW_g (body weight in grams), ADG_g (average daily gain), "
            "FI_g_per_day (feed intake per day), Cumulative_FI_g (cumulative feed intake), "
            "FCR (feed conversion ratio), Cumulative_FCR, and Mortality_Pct. "
            "Also provides processing yield data including breast meat yield, fat pad percentage, "
            "leg yield, wing yield, and processing weight. "
            "Input: genetic line, sex (optional), day (optional), parameter (optional)."
        )

    # ── 营养需要量 ────────────────────────────────────────────
    nutrient_breed_column = "Genetic_Line"
    nutrient_breed_label = "Broiler genetic lines"
    nutrient_available_breeds = "Ross 308, Ross 708, Cobb 500, Shengze 901, Guangming 2, Wode 188, Qingyuan Partridge, Sanhuang, Crossbred 817"
    nutrient_phase_help = (
        "Phase: 'Starter 0-10d', 'Grower 11-24d', "
        "'Finisher 25-35d', 'Withdrawal 36-42d'"
    )

    def nutrient_tool_description(self) -> str:
        return (
            f"Queries poultry nutrient requirement specifications. "
            f"For broilers ({self.nutrient_available_breeds}): phases include Starter 0-10d, "
            f"Grower 11-24d, Finisher 25-35d, Withdrawal 36-42d. "
            f"Parameters: ME_kcal_kg, CP_pct, Ca_pct, Avail_P_pct, Lys_min_pct, Met_min_pct, "
            f"SID amino acids, choline, linoleic acid, vitamins, minerals."
        )
