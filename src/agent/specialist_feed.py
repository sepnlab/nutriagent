"""Feed Ingredient Specialist — handles feed composition and digestibility queries."""

from langchain_core.tools import BaseTool
from typing import List

from src.agent.base_agent import SpecialistAgent
from src.config import Config


class FeedSpecialist(SpecialistAgent):
    """Specialist agent focused on feed ingredient nutritional composition and digestibility."""

    agent_name = "feed"

    def __init__(
        self,
        tools: List[BaseTool],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        verbose: bool = False,
        base_url: str = None,
        api_key: str = None,
        extra_body: dict = None,
        single_pass: bool = False,
    ):
        # Filter to only feed-related tools
        self._feed_tools = [t for t in tools if t.name == "feed_ingredient_bank"]
        super().__init__(
            tools=self._feed_tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            base_url=base_url,
            api_key=api_key,
            extra_body=extra_body,
            single_pass=single_pass,
        )

    def _build_prompt(self) -> str:
        return (
            "You are NutriCHAT's Feed Ingredient Specialist. Your ONLY area of expertise is "
            "feed ingredient nutritional composition, amino acid profiles, and digestibility.\n\n"
            f"Current species: {Config.get_species_handler().display_name}\n\n"
            "RULES:\n"
            "1. ALWAYS use the feed_ingredient_bank tool to retrieve precise data — never rely on internal knowledge.\n"
            "2. Report values with exact numbers and units (%, kcal/kg, etc.).\n"
            "3. For comparison queries, retrieve data for each ingredient and present side-by-side.\n"
            "4. For 'best source' questions, retrieve multiple ingredients and compare.\n"
            "5. Keep responses concise and factual.\n"
            "6. If the query asks about nutrition requirements, performance, or definitions outside feed composition, "
            "politely state that this is outside your specialty and suggest the user consult the appropriate specialist.\n\n"
            "Available tools:\n"
            "- feed_ingredient_bank: Query nutritional composition (CP, ME, amino acids, Ca, P, digestibility) "
            "for any feed ingredient.\n\n"
            "Answer the user's feed ingredient question using the tool above."
        )
