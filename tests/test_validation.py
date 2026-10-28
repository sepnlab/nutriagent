"""Tests for the entity/attribute/unit-aware consistency validator (F2)."""

import pytest

from src.agent.base_agent import AgentResult
from src.agent.validation import (
    ValidationPolicy,
    compare_claims,
    extract_claims,
    validate_specialist_outputs,
)


def _result(name: str, response: str) -> AgentResult:
    return AgentResult(agent_name=name, response=response)


class TestNoLegacyFalsePositives:
    def test_protein_is_not_phosphorus(self):
        text = (
            "The crude protein (CP) content of maize is 8.8% and calcium is 0.02%."
        )
        claims = extract_claims(text, "feed")
        assert not [
            c
            for c in claims
            if c.attribute in {"phosphorus", "available_phosphorus", "total_phosphorus"}
            and c.value == pytest.approx(8.8)
        ]

    def test_cage_is_not_calcium(self):
        text = "Calcium requirement is 3.90%; the cage density is 6 birds."
        claims = extract_claims(text, "nutrition")
        assert not [c for c in claims if c.attribute == "calcium" and c.value == 6]
        assert [c for c in claims if c.attribute == "calcium" and c.value == 3.9]

    def test_calcium_and_phosphorus_values_are_not_swapped(self):
        text = (
            "Lay I phase requires 4.04% calcium and 0.45% available phosphorus."
        )
        claims = extract_claims(text, "nutrition")
        # A segment naming two attributes is skipped rather than mis-aligned.
        assert claims == []


class TestClaimExtraction:
    def test_entity_before_attribute(self):
        claims = extract_claims("For maize, crude protein is 8.5%.", "feed")
        cp = [c for c in claims if c.attribute == "crude_protein"]
        assert len(cp) == 1
        assert cp[0].entity == "Maize"
        assert cp[0].value == pytest.approx(8.5)

    def test_entity_after_attribute(self):
        claims = extract_claims("The crude protein of maize is 8.8%.", "feed")
        cp = [c for c in claims if c.attribute == "crude_protein"]
        assert len(cp) == 1
        assert cp[0].entity == "Maize"

    def test_energy_with_thousands_separator(self):
        claims = extract_claims(
            "The metabolizable energy of maize is 3,350 kcal/kg.", "feed"
        )
        me = [c for c in claims if c.attribute == "metabolizable_energy"]
        assert len(me) == 1
        assert me[0].value == pytest.approx(3350)
        assert me[0].basis == "energy"

    def test_chinese_alias_resolves_to_canonical(self):
        claims = extract_claims("玉米 CP 8.5%", "feed")
        cp = [c for c in claims if c.attribute == "crude_protein"]
        assert len(cp) == 1
        assert cp[0].entity == "Maize"

    def test_ambiguous_chinese_alias_is_not_used(self):
        # Bare 豆粕 spans 46/48/50% variants and is flagged ambiguous.
        claims = extract_claims("豆粕 CP 8.5%", "feed")
        assert claims and claims[0].entity is None

    def test_specific_chinese_alias_resolves(self):
        claims = extract_claims("豆粕48 CP 45%", "feed")
        assert claims and claims[0].entity == (
            "Soybean meal, oil < 5%, 48% protein + oil"
        )

    def test_multi_attribute_sentence_skipped(self):
        claims = extract_claims(
            "Maize has 8.5% CP and 3350 kcal/kg ME.", "feed"
        )
        assert claims == []


class TestComparison:
    def test_same_entity_conflict_warns(self):
        warnings = validate_specialist_outputs(
            [
                _result("feed", "For maize, crude protein is 8.5%."),
                _result("nutrition", "For maize, crude protein is 9.8%."),
            ]
        )
        assert len(warnings) == 1
        assert "Maize" in warnings[0]
        assert "crude protein" in warnings[0]

    def test_different_entities_do_not_warn(self):
        warnings = validate_specialist_outputs(
            [
                _result("feed", "For maize, crude protein is 8.5%."),
                _result("feed2", "For limestone, crude protein is 9.8%."),
            ]
        )
        assert warnings == []

    def test_different_attributes_do_not_warn(self):
        warnings = validate_specialist_outputs(
            [
                _result("feed", "For maize, crude protein is 8.5%."),
                _result("nutrition", "For maize, metabolizable energy is 3350 kcal/kg."),
            ]
        )
        assert warnings == []

    def test_unit_conversion_g_kg_to_percent(self):
        warnings = validate_specialist_outputs(
            [
                _result("feed", "For maize, calcium is 0.2 g/kg."),
                _result("nutrition", "For maize, calcium is 0.02%."),
            ]
        )
        assert warnings == []

    def test_unit_conversion_mj_to_kcal(self):
        warnings = validate_specialist_outputs(
            [
                _result("feed", "For maize, metabolizable energy is 14 MJ/kg."),
                _result("nutrition", "For maize, metabolizable energy is 3346 kcal/kg."),
            ]
        )
        assert warnings == []

    def test_unresolved_entity_is_not_compared(self):
        warnings = validate_specialist_outputs(
            [
                _result("feed", "Crude protein is 8.5%."),
                _result("nutrition", "Crude protein is 9.8%."),
            ]
        )
        assert warnings == []

    def test_ca_p_check_is_opt_in(self):
        results = [
            _result("nutrition", "For maize, calcium is 4.0%."),
            _result("feed", "For maize, available phosphorus is 0.1%."),
        ]
        assert validate_specialist_outputs(results) == []
        warnings = validate_specialist_outputs(
            results, policy=ValidationPolicy(ca_p_check=True)
        )
        assert len(warnings) == 1
        assert "Ca:" in warnings[0]

    def test_ca_p_normal_layer_ratio_not_flagged(self):
        # Ca 4.0% with available P 0.4% is a normal layer ratio (10:1).
        # The old generic 1.2-5.0 band flagged this; the opt-in policy must not.
        results = [
            _result("nutrition", "For maize, calcium is 4.0%."),
            _result("feed", "For maize, available phosphorus is 0.4%."),
        ]
        assert validate_specialist_outputs(
            results, policy=ValidationPolicy(ca_p_check=True)
        ) == []


class TestSameSpecialistIgnored:
    def test_same_specialist_not_compared(self):
        claims = extract_claims(
            "For maize, crude protein is 8.5%. For maize, crude protein is 9.8%.",
            "feed",
        )
        result = compare_claims(claims)
        assert result.compared_pairs == 0
        assert result.warnings == []


class TestValueBeforeAttribute:
    """The earlier 38%-Ca failure used 'value before attribute' order."""

    def test_extracts_value_before_attribute(self):
        claims = extract_claims("Limestone contains 35% calcium.", "feed")
        assert len(claims) == 1
        claim = claims[0]
        assert claim.attribute == "calcium"
        assert claim.value == pytest.approx(35.0)
        assert claim.entity is not None

    def test_detects_conflict_value_before_attribute(self):
        claims = extract_claims(
            "Limestone contains 35% calcium.", "feed"
        ) + extract_claims("Limestone contains 38% calcium.", "nutrition")
        result = compare_claims(claims)
        assert result.compared_pairs == 1
        assert len(result.warnings) == 1
        assert "limestone" in result.warnings[0].lower()

    def test_unit_conversion_value_before_attribute(self):
        claims = extract_claims(
            "Limestone contains 350 g/kg calcium.", "feed"
        ) + extract_claims("Limestone contains 35% calcium.", "nutrition")
        result = compare_claims(claims)
        assert result.compared_pairs == 1
        assert result.warnings == []

    def test_commodity_family_alias_resolves(self):
        # The feed tables use "Soybean meal, solvent extracted, ..."; free text
        # says "soybean meal". The family alias must resolve without guessing a
        # specific table row.
        claims = extract_claims("Soybean meal contains 44% crude protein.", "feed")
        assert len(claims) == 1
        assert claims[0].entity is not None

    def test_phase_alias_resolves(self):
        # "Lay I 21-40w" in the table; "Lay I" in free text.
        claims = extract_claims("Lay I requires 4.04% calcium.", "nutrition")
        assert len(claims) == 1
        assert claims[0].entity is not None
