"""Layer (egg-type chicken) handler."""

from src.species.base import SpeciesHandler


class LayerHandler(SpeciesHandler):
    key = "layer"

    # ── 元数据 ──────────────────────────────────────────────
    display_name = "Layer (egg-type chicken)"
    breeds = "Hy-Line W-36, Hy-Line Brown, Hy-Line Pink, ISA Brown, Lohmann LSL-Classic, Lohmann Pink, Lohmann Grey, Hisex Brown, Dawu Jinfeng, Dawu He, Dawu Fen 1, Nongda 3, Nongda 6, Beinong 1, Beinong 2, Jinghong 1 v3.0, Jingbai 939, Jingfen 1, Jingfen 2, Jingfen 6"
    phases = "Starter → Grower → Developer → Pre-Lay → Lay I → Lay II → Lay III"
    metrics = "Hen-day production, egg weight, egg mass, feed/dozen, feed/kg egg, body weight, shell quality, mortality"

    # ── 性能管理 ────────────────────────────────────────────
    performance_breed_column = "Breed"
    performance_age_column = "Week"
    performance_age_field = "week"
    performance_age_desc = "Week (0-100)"

    @property
    def performance_params(self) -> dict:
        return {
            "HenDay_Prod_Pct": "Hen-day egg production percentage",
            "Egg_Wt_g": "Average egg weight (g)",
            "Egg_Mass_g_per_d": "Egg mass per hen per day (g)",
            "FI_g_per_d": "Feed intake per hen per day (g)",
            "Feed_per_Dozen_kg": "Feed per dozen eggs (kg)",
            "Feed_per_kg_Egg_kg": "Feed per kg egg mass (kg)",
            "BW_g": "Body weight (g)",
            "Mortality_Pct": "Mortality percentage",
            "Shell_Strength_N": "Egg shell breaking strength (N)",
            "Haugh_Unit": "Haugh unit (internal egg quality)",
        }

    @property
    def processing_params(self) -> list:
        return []

    def performance_tool_description(self) -> str:
        return (
            "Queries layer performance data including egg production, egg quality, and body weight. "
            "Parameters: HenDay_Prod_Pct (hen-day egg production %), Egg_Wt_g (egg weight), "
            "Egg_Mass_g_per_d (egg mass per hen per day), FI_g_per_d (feed intake), "
            "Feed_per_Dozen_kg (feed per dozen eggs), Feed_per_kg_Egg_kg, BW_g (body weight), "
            "Mortality_Pct, Shell_Strength_N (shell breaking strength), Haugh_Unit (internal egg quality). "
            "Breeds: Hy-Line W-36, Hy-Line Brown, Hy-Line Pink, ISA Brown, Lohmann LSL-Classic, "
            "Lohmann Pink, Lohmann Grey, Hisex Brown, Dawu Jinfeng, Dawu He, Dawu Fen 1, "
            "Nongda 3, Nongda 6, Beinong 1, Beinong 2, "
            "Jinghong 1 v3.0, Jingbai 939, Jingfen 1, Jingfen 2, Jingfen 6. "
            "Age: weeks (0-100). Input: breed, week (optional), parameter (optional)."
        )

    # ── 营养需要量 ────────────────────────────────────────────
    nutrient_breed_column = "Breed"
    nutrient_breed_label = "Layer breeds"
    nutrient_available_breeds = "Hy-Line W-36, Hy-Line Brown, Hy-Line Pink, ISA Brown, Lohmann LSL-Classic, Lohmann Pink, Lohmann Grey, Hisex Brown, Dawu Jinfeng, Dawu He, Dawu Fen 1, Nongda 3, Nongda 6, Beinong 1, Beinong 2, Jinghong 1 v3.0, Jingbai 939, Jingfen 1, Jingfen 2, Jingfen 6"
    nutrient_phase_help = (
        "Phase: 'Starter 0-6w', 'Grower 7-12w', 'Developer 13-17w', 'Pre-Lay 18-20w', "
        "'Lay I 21-40w', 'Lay II 41-60w', 'Lay III 61w+'"
    )

    def nutrient_tool_description(self) -> str:
        return (
            f"Queries poultry nutrient requirement specifications. "
            f"For layers ({self.nutrient_available_breeds}): phases include Starter 0-6w, "
            f"Grower 7-12w, Developer 13-17w, Pre-Lay 18-20w, Lay I 21-40w, "
            f"Lay II 41-60w, Lay III 61w+. "
            f"Parameters: ME_kcal_kg, CP_pct, Ca_pct, Avail_P_pct, Lys_pct, Met_pct, "
            f"SID amino acids, vitamins, minerals."
        )
