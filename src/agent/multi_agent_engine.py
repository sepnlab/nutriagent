"""Multi-Agent Engine — orchestrates router, specialists, and synthesis."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from langchain_core.tools import BaseTool

from src.config import Config
from src.agent.base_agent import AgentResult
from src.agent.router_agent import RouterAgent
from src.agent.specialist_feed import FeedSpecialist
from src.agent.specialist_nutrition import NutritionSpecialist
from src.agent.specialist_performance import PerformanceSpecialist
from src.agent.specialist_definitions import DefinitionsSpecialist
from src.agent.synthesis_agent import SynthesisAgent
from src.rag.poultry_db import PoultryDB
from src.tools import TOOL_DOMAINS


class MultiAgentEngine:
    """Orchestrates query routing → specialist agents → synthesis → RAG integration.

    Architecture:
        Router → Specialist(s) → Domain aggregation → RAG filter → Synthesis → Final answer
    """

    def __init__(
        self,
        tools: List[BaseTool],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        verbose: bool = False,
        rag_enabled: bool = True,
        specialist_timeout: int = 120,
        synthesis_enabled: bool = True,
        base_url: str = None,
        api_key: str = None,
        extra_body: dict = None,
        single_pass: bool = False,
        validation_mode: str = "auto",
        synthesis_mode: str = "synthesis",
    ):
        self.tools = tools
        self.verbose = verbose
        self.rag_enabled = rag_enabled
        self.specialist_timeout = specialist_timeout
        self.synthesis_enabled = synthesis_enabled
        # single_pass=True → each specialist makes exactly one tool call
        # (true no-ReAct control for the D1 ablation).
        self.single_pass = single_pass
        # validation_mode: "auto" (router flag) | "always" | "off" | "shuffled"
        self.validation_mode = validation_mode
        # synthesis_mode: "synthesis" (LLM integration) | "concat"
        # (deterministic concatenation, optionally with validation notes)
        self.synthesis_mode = synthesis_mode

        # Initialize router
        self.router = RouterAgent()

        # Initialize specialists（使用独立 token 限制，仅检索工具数据）
        kw = dict(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens or Config.SPECIALIST_MAX_TOKENS,
            verbose=verbose,
            base_url=base_url,
            api_key=api_key,
            extra_body=extra_body,
            single_pass=single_pass,
        )
        self.specialists = {
            "feed": FeedSpecialist(tools, **kw),
            "nutrition": NutritionSpecialist(tools, **kw),
            "performance": PerformanceSpecialist(tools, **kw),
            "definitions": DefinitionsSpecialist(tools, **kw),
        }

        # Initialize synthesis
        self.synthesis = SynthesisAgent(model=model, base_url=base_url, api_key=api_key, extra_body=extra_body)

        # Initialize RAG
        self.poultry_db = PoultryDB() if rag_enabled else None

    def answer(self, query: str) -> dict:
        """Process a query through the multi-agent pipeline.

        Returns dict with: response, steps, rag_context, tools_used, domains, mode
        plus full token accounting (router + specialists + synthesis) and latency.
        """
        t_start = time.perf_counter()
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Multi-Agent Engine processing: {query}")
            print(f"{'='*60}")

        # Step 1: Route the query
        decision = self.router.route(query)
        router_usage = dict(getattr(self.router, "_last_usage", {}) or {})
        if self.verbose:
            print(f"  [Router] intent={decision.intent}, targets={decision.target_agents}")

        # Step 2: Run target specialists in parallel
        results: List[AgentResult] = []
        target_names = [n for n in decision.target_agents if n in self.specialists]
        # Fallback: if Router targeted excluded specialists, run all remaining ones
        if not target_names:
            target_names = list(self.specialists.keys())
        if self.verbose:
            print(f"  [Specialists] Launching {len(target_names)} agents in parallel...")

        with ThreadPoolExecutor(max_workers=len(target_names) or 1) as executor:
            future_map = {
                executor.submit(self.specialists[name].run, query): name
                for name in target_names
            }
            for future in as_completed(future_map):
                agent_name = future_map[future]
                try:
                    result = future.result(timeout=self.specialist_timeout)
                    results.append(result)
                    if result.error:
                        if self.verbose:
                            print(f"  [{agent_name}] Completed with error: {result.error}")
                    elif self.verbose:
                        print(f"  [{agent_name}] Done ({len(result.response)} chars, "
                              f"{result.prompt_tokens}↑+{result.completion_tokens}↓ tokens)")
                except TimeoutError:
                    if self.verbose:
                        print(f"  [{agent_name}] Timeout (>={self.specialist_timeout}s)")
                    future.cancel()
                except Exception as e:
                    if self.verbose:
                        print(f"  [{agent_name}] Error: {e}")

        # Step 3: Aggregate domains from all specialists
        all_domains = set()
        all_tools = []
        for r in results:
            all_domains.update(r.domains)
            for tc in r.tool_calls:
                action = tc.get("action", "")
                if action and action not in all_tools:
                    all_tools.append(action)

        # Step 4: Get RAG context with domain filtering
        rag_context = ""
        if self.rag_enabled and self.poultry_db:
            rag_context = self.synthesis._get_rag_context(
                query, all_domains, self.poultry_db
            )
            if self.verbose:
                print(f"  [RAG] domains={all_domains}, provided={'Yes' if rag_context else 'No'}")

        # Step 5: Synthesize final answer (or concatenate directly)
        validation_mode = self._resolve_validation_mode(decision)
        synth_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        if self.synthesis_enabled and self.synthesis_mode == "synthesis":
            final_response = self.synthesis.synthesize(
                query, results, rag_context, validation_mode=validation_mode
            )
            synth_usage = dict(getattr(self.synthesis, "_last_usage", {}) or {})
        else:
            # Concatenation control: append deterministic warnings if enabled,
            # but never call the synthesis LLM.
            parts = [r.response for r in results]
            if rag_context:
                parts.append(f"Supplementary reference:\n{rag_context[:1500]}")
            if validation_mode != "off":
                warnings = self.synthesis.cross_validate(results)
                if warnings:
                    parts.append(
                        "Cross-validation notes:\n"
                        + "\n".join(f"- {w}" for w in warnings)
                    )
            final_response = "\n\n".join(parts) if parts else "No response generated."

        if self.verbose:
            print(f"\n  [Final] {len(final_response)} chars")
            print(f"{'='*60}")

        specialist_prompt = sum(r.prompt_tokens for r in results)
        specialist_completion = sum(r.completion_tokens for r in results)
        router_prompt = int(router_usage.get("prompt_tokens", 0) or 0)
        router_completion = int(router_usage.get("completion_tokens", 0) or 0)
        synth_prompt = int(synth_usage.get("prompt_tokens", 0) or 0)
        synth_completion = int(synth_usage.get("completion_tokens", 0) or 0)
        total_prompt = specialist_prompt + router_prompt + synth_prompt
        total_completion = specialist_completion + router_completion + synth_completion

        elapsed_s = time.perf_counter() - t_start

        return {
            "query": query,
            "response": final_response,
            "steps": [tc for r in results for tc in r.tool_calls],
            "rag_context": rag_context,
            "tools_used": all_tools,
            "react_domains": list(all_domains),
            "rag_enabled": self.rag_enabled,
            "mode": "multi",
            "specialists_used": decision.target_agents,
            "cross_validation_applied": validation_mode != "off",
            "validation_mode": validation_mode,
            "synthesis_mode": self.synthesis_mode,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "elapsed_s": round(elapsed_s, 3),
            "component_usage": {
                "router": {
                    "prompt_tokens": router_prompt,
                    "completion_tokens": router_completion,
                },
                "specialists": {
                    r.agent_name: {
                        "prompt_tokens": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens,
                        "tool_calls": len(r.tool_calls),
                    }
                    for r in results
                },
                "synthesis": {
                    "prompt_tokens": synth_prompt,
                    "completion_tokens": synth_completion,
                },
            },
        }

    def _resolve_validation_mode(self, decision) -> str:
        """Resolve the consistency-check mode for this query.

        - "off": never
        - "always": always (legacy behaviour)
        - "shuffled": always, but warning order is randomised
        - "auto": follow the router's requires_cross_validation flag, falling
          back to "multiple specialists were activated"
        """
        if self.validation_mode in ("off", "always", "shuffled"):
            return self.validation_mode
        auto = bool(decision.requires_cross_validation) or len(
            decision.target_agents
        ) > 1
        return "always" if auto else "off"

    def _should_validate(self, decision) -> bool:
        """Boolean compatibility wrapper around :meth:`_resolve_validation_mode`."""
        return self._resolve_validation_mode(decision) != "off"
