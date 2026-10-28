"""Hybrid ReAct-RAG Engine with optional multi-agent mode.

Combines structured reasoning with RAG-based semantic retrieval.
Supports two modes:
- single: Original single ReAct agent (NutriCHAT paper architecture)
- multi: Multi-agent system (Router → Specialists → Synthesis)
"""
import os
import time
from typing import List, Optional
from langchain_core.tools import BaseTool

from src.config import Config

from src.agent.react_agent import ReActAgent
from src.rag.poultry_db import PoultryDB
from src.tools import TOOL_DOMAINS


class NutriCHATEngine:
    """Hybrid ReAct-RAG engine with configurable agent architecture.

    Two modes:
    - single: Single ReAct agent with all tools (original NutriCHAT)
    - multi: Multi-agent system with Router + Specialist agents + Synthesis
    """

    def __init__(
        self,
        tools: List[BaseTool],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        max_iterations: int = None,
        verbose: bool = False,
        rag_enabled: bool = True,
        mode: str = "single",
    ):
        self.tools = tools
        self.tool_names = {t.name: t for t in tools}
        self.verbose = verbose
        self.rag_enabled = rag_enabled
        self.max_tokens = max_tokens
        self.mode = mode

        if mode == "multi":
            from src.agent.multi_agent_engine import MultiAgentEngine
            self.multi_agent = MultiAgentEngine(
                tools=tools,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                verbose=verbose,
                rag_enabled=rag_enabled,
            )
            self.react_agent = None
        else:
            # Initialize ReAct agent (original mode)
            self.react_agent = ReActAgent(
                tools=tools,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_iterations=max_iterations,
                verbose=verbose,
            )
            self.multi_agent = None

        # Initialize RAG component (in single mode only; multi mode handles its own RAG)
        if mode != "multi" and rag_enabled:
            self.poultry_db = PoultryDB()
        else:
            self.poultry_db = None

    def _extract_tool_domains(self, steps: List[dict]) -> set:
        """Extract the knowledge domains of tools invoked by the agent(s)."""
        domains = set()
        for step in steps:
            action = step.get("action", "")
            if action in TOOL_DOMAINS:
                domains.add(TOOL_DOMAINS[action])
        return domains

    def _get_rag_context(self, query: str, react_domains: set) -> str:
        """Get RAG context, filtered to avoid redundancy with agent domains.

        The automated filtering mechanism: provide RAG information only when the
        retrieved knowledge domain differs from the tool domains invoked by the agent.
        """
        if not self.rag_enabled or not self.poultry_db:
            return ""

        # Get RAG retrieval
        rag_results = self.poultry_db.search(query, k=1)

        if not rag_results:
            return ""

        rag_domain = rag_results[0].get("source", "Unknown")
        rag_content = rag_results[0].get("content", "")

        if self.verbose:
            print(f"\n[RAG] Retrieved domain: {rag_domain}")
            print(f"[RAG] Agent domains: {react_domains}")

        # Filter: provide RAG info only when domains differ
        if rag_domain not in react_domains:
            if self.verbose:
                print(f"[RAG] Domains differ → providing RAG context")
            return f"\n[RAG Context - {rag_domain}]:\n{rag_content[:1000]}"
        else:
            if self.verbose:
                print(f"[RAG] Domains match → skipping RAG (avoids redundancy)")
            return ""

    def _cache_key(self, query: str) -> str:
        """Generate a cache key for a query."""
        import hashlib
        raw = f"{Config.SPECIES}|{Config.LLM_MODEL}|{self.mode}|{query}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def _cache_get(self, key: str) -> Optional[dict]:
        """Read cached result if available and fresh."""
        import json
        cache_dir = Config.CACHE_DIR
        path = os.path.join(cache_dir, f"{key}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _cache_set(self, key: str, result: dict):
        """Write result to disk cache."""
        import json
        cache_dir = Config.CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{key}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

    def answer(self, query: str, use_cache: bool = True) -> dict:
        """Generate a comprehensive answer.

        In 'single' mode: uses original ReAct + RAG + domain filtering (NutriCHAT paper).
        In 'multi' mode: uses Router → Specialists → Synthesis + RAG.

        Results are cached to disk; set use_cache=False to bypass.
        """
        # Multi-agent mode: delegate to MultiAgentEngine
        if self.mode == "multi":
            return self.multi_agent.answer(query)

        # ===== Single (original) mode =====
        # Check cache
        if use_cache:
            key = self._cache_key(query)
            cached = self._cache_get(key)
            if cached is not None:
                return cached

        t0 = time.time()

        # Step 1: Run ReAct agent
        response, steps = self.react_agent.run(query)
        # Forward token usage for cost tracking
        self._last_agent_usage = self.react_agent.last_usage
        if self.verbose:
            print(f"  [usage] agent tokens: {self._last_agent_usage}")

        # Step 2: Extract tool domains from ReAct steps
        react_domains = self._extract_tool_domains(steps)
        tools_used = [s.get("action", "") for s in steps if s.get("action")]

        rag_context = ""
        # Step 3: Get RAG context with domain filtering (non-fatal)
        if self.rag_enabled and self.poultry_db:
            try:
                rag_context = self._get_rag_context(query, react_domains)
            except Exception:
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                rag_context = ""

        # Step 4: If RAG provided complementary info, synthesize enhanced response
        final_response = response
        if rag_context:
            _synthesis_prompt = (
                "You are a poultry nutrition assistant. Below is an original answer and "
                "supplementary reference information. Revise the answer to naturally "
                "incorporate the additional information where relevant.\n\n"
                f"ORIGINAL ANSWER:\n{response}\n\n"
                f"SUPPLEMENTARY INFORMATION:\n{rag_context}\n\n"
                "REVISED ANSWER:"
            )
            try:
                synth_resp = self.react_agent.llm.invoke(_synthesis_prompt)
                final_response = synth_resp.content.strip() or response
            except Exception:
                final_response = response

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"HYBRID ENGINE RESULT")
            print(f"{'='*60}")
            print(f"Tools used: {tools_used}")
            print(f"RAG provided: {'Yes' if rag_context else 'No'}")
            print(f"Response length: {len(final_response)} chars")

        result = {
            "query": query,
            "response": final_response,
            "steps": steps,
            "rag_context": rag_context,
            "tools_used": tools_used,
            "react_domains": list(react_domains),
            "rag_enabled": self.rag_enabled,
            "mode": "single",
            "prompt_tokens": self._last_agent_usage.get("prompt_tokens", 0),
            "completion_tokens": self._last_agent_usage.get("completion_tokens", 0),
            "elapsed_s": round(time.time() - t0, 1),
        }

        # Save to disk cache
        if use_cache:
            try:
                self._cache_set(self._cache_key(query), result)
            except Exception:
                pass  # cache write failure is non-critical

        return result

    def build_vector_db(self, force_rebuild: bool = False) -> int:
        """Build the PoultryDB vector database."""
        if self.mode == "multi" and self.multi_agent:
            if self.multi_agent.poultry_db:
                return self.multi_agent.poultry_db.build(force_rebuild=force_rebuild)
            return 0
        if self.poultry_db:
            return self.poultry_db.build(force_rebuild=force_rebuild)
        return 0
