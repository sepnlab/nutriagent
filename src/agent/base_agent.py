"""Specialist agent base class for multi-agent architecture."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from src.config import Config


class FixedChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that fixes max_tokens for non-OpenAI providers.

    LangChain converts max_tokens → max_completion_tokens (OpenAI 2024-09+ standard).
    Many OpenAI-compatible providers (SiliconFlow, DeepSeek, Aliyun) silently ignore
    max_completion_tokens, causing the token limit to not take effect.
    This class reverses the conversion so max_tokens is sent directly.
    """

    @property
    def _default_params(self) -> Dict[str, Any]:
        params = super()._default_params
        if "max_completion_tokens" in params:
            params["max_tokens"] = params.pop("max_completion_tokens")
        return params

    def _get_request_payload(
        self, input_: Any, stop: Optional[Any] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        return payload


@dataclass
class AgentResult:
    """Result from a single specialist agent invocation."""
    agent_name: str
    response: str
    tool_calls: list = field(default_factory=list)
    domains: set = field(default_factory=set)
    confidence: float = 1.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None


@dataclass
class RouterDecision:
    """Routing decision from the RouterAgent."""
    query: str
    intent: str  # "feed" | "nutrition" | "performance" | "definitions" | "multi"
    target_agents: list = field(default_factory=list)
    requires_cross_validation: bool = False


class SpecialistAgent(ABC):
    """Base class for domain-specialist agents.

    Each specialist has a focused system prompt and only the tools
    relevant to its domain, reducing context competition.
    """

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
        self.tools = tools
        self.max_tokens = max_tokens or Config.MAX_TOKENS
        self.verbose = verbose
        # single_pass=True implements a true "no ReAct" variant: exactly one
        # tool call, then a final answer from that observation (no iteration).
        self.single_pass = single_pass

        kwargs = dict(
            model=model or Config.LLM_MODEL,
            temperature=temperature if temperature is not None else Config.TEMPERATURE,
            max_tokens=self.max_tokens,
            openai_api_key=api_key or Config.LLM_API_KEY,
            openai_api_base=base_url or Config.LLM_BASE_URL,
            timeout=Config.API_TIMEOUT,
        )
        eb = extra_body or Config.LLM_EXTRA_BODY
        if eb:
            kwargs["extra_body"] = eb
        self.llm = FixedChatOpenAI(**kwargs)

        system = self._build_prompt()
        self.graph = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system,
            name=self.agent_name,
        )

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Unique name for this specialist agent."""
        ...

    @abstractmethod
    def _build_prompt(self) -> str:
        """Build the focused system prompt for this specialist."""
        ...

    def run(self, query: str) -> AgentResult:
        """Execute this specialist's ReAct loop for a single query.

        Returns AgentResult with response, tool calls, and token usage.
        When ``single_pass`` is set, executes one tool call and answers
        directly from that observation (no iterative reasoning).
        """
        if self.single_pass:
            return self._run_single_pass(query)

        inputs = {"messages": [HumanMessage(content=query)]}
        steps = []
        response = "No response generated."
        prompt_tokens = 0
        completion_tokens = 0

        # Recursion limit: prevent infinite tool-call loops (same as react_agent)
        _stream_config = {"recursion_limit": 25}

        try:
            for chunk in self.graph.stream(inputs, stream_mode="updates", config=_stream_config):
                for _node_name, node_data in chunk.items():
                    msgs = node_data.get("messages", [])
                    for msg in msgs:
                        um = getattr(msg, "usage_metadata", None)
                        if um:
                            prompt_tokens += um.get("input_tokens", 0)
                            completion_tokens += um.get("output_tokens", 0)

                    for msg in msgs:
                        tc_attr = getattr(msg, "tool_calls", None)
                        if tc_attr:
                            for tc in tc_attr:
                                tc_name = tc.name if hasattr(tc, "name") else tc.get("name", "unknown")
                                tc_args = tc.args if hasattr(tc, "args") else tc.get("args", {})
                                steps.append({
                                    "thought": msg.content or "Calling tool...",
                                    "action": tc_name,
                                    "action_input": str(tc_args),
                                    "observation": "",
                                })
                        elif msg.type == "ai" and msg.content:
                            response = msg.content
                        elif msg.type == "tool":
                            if steps:
                                steps[-1]["observation"] = msg.content[:500]
        except Exception as e:
            if self.verbose:
                print(f"  [{self.agent_name}] Agent error: {e}")
            response = self._fallback_response(query, str(e))
            return AgentResult(
                agent_name=self.agent_name,
                response=response,
                tool_calls=steps,
                domains=set(),
                error=str(e),
            )

        domains = set()
        for s in steps:
            action = s.get("action", "")
            if action in _TOOL_DOMAIN_MAP:
                domains.add(_TOOL_DOMAIN_MAP[action])

        return AgentResult(
            agent_name=self.agent_name,
            response=response,
            tool_calls=steps,
            domains=domains,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error=None,
        )

    def _run_single_pass(self, query: str) -> AgentResult:
        """One tool call, then answer from that observation (true no-ReAct).

        This is deliberately different from the ReAct loop: the model may
        request exactly one tool, receives its observation, and must produce
        the final answer without any further tool access. It is the control
        for "does iterative reasoning help?" in the D1 ablation.
        """
        steps = []
        prompt_tokens = 0
        completion_tokens = 0

        def _accumulate(msg):
            nonlocal prompt_tokens, completion_tokens
            um = getattr(msg, "usage_metadata", None)
            if um:
                prompt_tokens += int(um.get("input_tokens", 0) or 0)
                completion_tokens += int(um.get("output_tokens", 0) or 0)

        try:
            if not self.tools:
                final = self.llm.invoke([HumanMessage(content=query)])
                _accumulate(final)
                return AgentResult(
                    agent_name=self.agent_name,
                    response=(final.content or "").strip(),
                    tool_calls=[],
                    domains=set(),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            first = self.llm.bind_tools(self.tools).invoke(
                [HumanMessage(content=query)]
            )
            _accumulate(first)
            tool_calls = getattr(first, "tool_calls", None) or []

            if not tool_calls:
                return AgentResult(
                    agent_name=self.agent_name,
                    response=(first.content or "").strip(),
                    tool_calls=[],
                    domains=set(),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            tc = tool_calls[0]
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
            args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
            tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")

            tool = next((t for t in self.tools if t.name == name), None)
            if tool is None:
                return AgentResult(
                    agent_name=self.agent_name,
                    response=(first.content or "").strip(),
                    tool_calls=[],
                    domains=set(),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            observation = str(tool.invoke(args if args is not None else {}))
            steps.append(
                {
                    "thought": first.content or "Calling tool...",
                    "action": name,
                    "action_input": str(args),
                    "observation": observation[:500],
                }
            )

            messages = [
                HumanMessage(content=query),
                first,
                ToolMessage(content=observation, tool_call_id=tc_id or "call_0"),
                HumanMessage(
                    content=(
                        "Using ONLY the tool observation above, answer the user's "
                        "question directly. Do not request any more tools."
                    )
                ),
            ]
            final = self.llm.invoke(messages)
            _accumulate(final)
            response = (final.content or "").strip()
        except Exception as e:
            if self.verbose:
                print(f"  [{self.agent_name}] single-pass error: {e}")
            return AgentResult(
                agent_name=self.agent_name,
                response=self._fallback_response(query, str(e)),
                tool_calls=steps,
                domains=set(),
                error=str(e),
            )

        domains = set()
        for s in steps:
            action = s.get("action", "")
            if action in _TOOL_DOMAIN_MAP:
                domains.add(_TOOL_DOMAIN_MAP[action])

        return AgentResult(
            agent_name=self.agent_name,
            response=response or "No response generated.",
            tool_calls=steps,
            domains=domains,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _fallback_response(self, query: str, error: str = "") -> str:
        """Fallback when agent fails."""
        if self.verbose:
            print(f"    [{self.agent_name}] Using fallback for: {query[:60]}")
        return f"[{self.agent_name}] Unable to process query: {error}"


# Import here to avoid circular imports at module level
def _get_tool_domain_map():
    from src.tools import TOOL_DOMAINS
    return TOOL_DOMAINS


_TOOL_DOMAIN_MAP = _get_tool_domain_map()
