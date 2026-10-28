"""Species handler 单元测试。"""

from src.species import get_handler


class TestBroilerHandler:
    def setup_method(self):
        self.h = get_handler("broiler")

    def test_metadata(self):
        assert self.h.key == "broiler"
        assert "Ross 308" in self.h.breeds
        assert "Starter" in self.h.phases
        assert "ADG" in self.h.metrics

    def test_performance(self):
        assert self.h.performance_breed_column == "Genetic_Line"
        assert self.h.performance_age_column == "Day"
        assert self.h.performance_age_field == "day"
        assert len(self.h.performance_params) == 7
        assert "BW_g" in self.h.performance_params
        assert "FCR" in self.h.performance_params

    def test_processing_params(self):
        assert "Breast_Meat_Yield_Pct" in self.h.processing_params

    def test_nutrient(self):
        assert self.h.nutrient_breed_column == "Genetic_Line"
        assert "Ross 308" in self.h.nutrient_available_breeds
        assert "Starter 0-10d" in self.h.nutrient_phase_help

    def test_descriptions_nonempty(self):
        assert len(self.h.performance_tool_description()) > 50
        assert len(self.h.nutrient_tool_description()) > 50


class TestLayerHandler:
    def setup_method(self):
        self.h = get_handler("layer")

    def test_metadata(self):
        assert self.h.key == "layer"
        assert "Hy-Line W-36" in self.h.breeds
        assert "Lay I" in self.h.phases
        assert "egg mass" in self.h.metrics

    def test_performance(self):
        assert self.h.performance_breed_column == "Breed"
        assert self.h.performance_age_column == "Week"
        assert self.h.performance_age_field == "week"
        assert len(self.h.performance_params) == 10
        assert "HenDay_Prod_Pct" in self.h.performance_params

    def test_processing_params_empty(self):
        assert self.h.processing_params == []

    def test_nutrient(self):
        assert self.h.nutrient_breed_column == "Breed"
        assert "Hy-Line" in self.h.nutrient_available_breeds
        assert "Lay I" in self.h.nutrient_phase_help

    def test_descriptions_nonempty(self):
        assert len(self.h.performance_tool_description()) > 50
        assert len(self.h.nutrient_tool_description()) > 50


class TestGetHandler:
    def test_default_from_config(self):
        from src.config import Config
        h = Config.get_species_handler()
        assert h.key == Config.SPECIES

    def test_unknown(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown species"):
            get_handler("duck")
