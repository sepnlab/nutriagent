"""Definitions Specialist — handles poultry nutrition terminology."""

from langchain_core.tools import BaseTool
from typing import List

from src.agent.base_agent import SpecialistAgent


class DefinitionsSpecialist(SpecialistAgent):
    """Specialist agent focused on poultry nutrition terminology definitions."""

    agent_name = "definitions"

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
        self._def_tools = [t for t in tools if t.name == "definitions"]
        super().__init__(
            tools=self._def_tools,
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
            "You are NutriCHAT's Definitions Specialist. Your ONLY area of expertise is "
            "poultry nutrition terminology and concept definitions.\n\n"
            "RULES:\n"
            "1. ALWAYS use the definitions tool to look up term definitions — never rely on internal knowledge.\n"
            "2. If an exact match isn't found, try related terms and suggest the closest matches.\n"
            "3. Present definitions clearly: 'Term: definition'.\n"
            "4. Keep responses concise — just the definition and brief context if relevant.\n"
            "5. If the query asks for nutritional data, performance data, or feed composition, "
            "politely state this is outside your specialty.\n\n"
            "Available tools:\n"
            "- definitions: Look up definitions of poultry nutrition terms and concepts "
            "(600+ terms covering ingredients, nutrients, physiology, diseases, etc.).\n\n"
            "Answer the user's terminology question using the tool above."
        )
