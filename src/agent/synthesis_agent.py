"""Synthesis Agent — consistency-checked integration of specialist outputs."""

import random
from typing import List

from src.agent.base_agent import AgentResult, FixedChatOpenAI
from src.agent.validation import validate_specialist_outputs
from src.config import Config


_SYNTHESIS_PROMPT = """You are a senior poultry nutrition expert reviewing answers from multiple specialist agents.
Your job is to cross-validate, resolve conflicts, and produce a single coherent final answer.

Original question: {query}

Specialist responses:
{specialist_outputs}

{rag_context}

{cross_validate_warnings}

Instructions:
1. Synthesize the information from all specialists into one coherent answer.
2. If there are contradictions between specialists, note the discrepancy and explain which value is best supported.
3. If supplementary information is provided, incorporate it where relevant.
4. The final answer should be comprehensive, accurate, and directly address the user's question.
5. Cite the source of each key number (e.g., "[from Performance tool]" or "[from Nutrient tool]").
6. Keep the same concise style as the original responses.

FINAL ANSWER:"""


class SynthesisAgent:
    """Cross-validates specialist outputs and generates a coherent final answer."""

    def __init__(self, model: str = None, base_url: str = None, api_key: str = None,
                 extra_body: dict = None):
        kwargs = dict(
            model=model or Config.LLM_MODEL,
            temperature=Config.SYNTHESIS_TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            openai_api_key=api_key or Config.LLM_API_KEY,
            openai_api_base=base_url or Config.LLM_BASE_URL,
            timeout=Config.API_TIMEOUT,
        )
        eb = extra_body or Config.LLM_EXTRA_BODY
        if eb:
            kwargs["extra_body"] = eb
        self.llm = FixedChatOpenAI(**kwargs)
        self._last_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def cross_validate(self, results: List[AgentResult]) -> List[str]:
        """Check for cross-specialist numerical inconsistencies.

        Delegates to :func:`src.agent.validation.validate_specialist_outputs`,
        which compares only claims sharing the same entity, attribute and
        basis. The legacy regex checks (generic Ca:P range, cross-attribute
        mean deviation) were removed because they read
        "crude protein ... 8.8" as P=8.8 and "cage density is 6" as Ca=6.
        """
        return validate_specialist_outputs(results)

    def synthesize(
        self,
        query: str,
        results: List[AgentResult],
        rag_context: str = "",
        validate: bool = True,
        validation_mode: str = None,
    ) -> str:
        """Merge specialist outputs into a final coherent answer.

        Args:
            validate: legacy boolean switch (kept for backward compatibility).
            validation_mode: "off" | "always" | "shuffled" | None. When given it
                takes precedence over ``validate``. "shuffled" keeps the same
                warnings but randomises their order — a control for whether the
                warning *content* or merely the prompt perturbation matters.
        """
        self._last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        mode = validation_mode or ("always" if validate else "off")
        warnings: List[str] = []
        if mode != "off":
            warnings = self.cross_validate(results)
            if mode == "shuffled":
                random.shuffle(warnings)

        # Format specialist outputs (skip errored results with empty responses)
        outputs = []
        for r in results:
            if r.error and r.response.startswith("["):
                # Fallback response from errored specialist — flag it
                outputs.append(
                    f"[{r.agent_name.upper()} - ERROR]: {r.response}"
                )
            else:
                domain_str = ", ".join(sorted(r.domains)) if r.domains else r.agent_name
                outputs.append(
                    f"[{r.agent_name.upper()} - {domain_str}]:\n{r.response}"
                )

        # Format warnings
        warn_text = ""
        if warnings:
            warn_text = "Cross-validation notes:\n" + "\n".join(
                f"- {w}" for w in warnings
            )

        # Format RAG context
        rag_text = ""
        if rag_context:
            rag_text = f"Supplementary reference:\n{rag_context[:1500]}"

        prompt = _SYNTHESIS_PROMPT.format(
            query=query,
            specialist_outputs="\n\n".join(outputs),
            rag_context=rag_text,
            cross_validate_warnings=warn_text,
        )

        try:
            resp = self.llm.invoke(prompt)
            usage = getattr(resp, "usage_metadata", None) or {}
            self._last_usage = {
                "prompt_tokens": int(usage.get("input_tokens", 0) or 0),
                "completion_tokens": int(usage.get("output_tokens", 0) or 0),
            }
            return resp.content.strip()
        except Exception:
            # Fallback: concatenate specialist responses
            parts = [r.response for r in results]
            return "\n\n".join(parts)

    def _get_rag_context(
        self,
        query: str,
        specialist_domains: set,
        poultry_db=None,
    ) -> str:
        """Get RAG context to supplement specialist knowledge."""
        if not poultry_db:
            return ""

        # manuscript reported k=1). `Config.RAG_K` is the single source of truth.
        rag_results = poultry_db.search(query, k=Config.RAG_K)
        if not rag_results:
            return ""

        contexts = []
        for doc in rag_results:
            domain = doc.get("source", "Unknown")
            content = doc.get("content", "")
            # Flag whether RAG domain overlaps with specialist domains
            tag = "(same domain)" if domain in specialist_domains else "(supplementary)"
            contexts.append(f"[{domain} {tag}]:\n{content[:500]}")
        return "\n\n".join(contexts)
