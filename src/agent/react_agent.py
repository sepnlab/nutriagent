"""ReAct Agent - Core reasoning engine for NutriCHAT.

Implements the Thought-Action-Observation cycle using LangChain's new
create_agent (LangGraph-based). LLM backend is configurable via
LLM_BASE_URL (default: SiliconFlow OpenAI-compatible API).
"""
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models import base as oai_base
from src.agent.base_agent import FixedChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

from src.config import Config

# Thread-local storage for per-thread token usage (avoids data race
# when multiple threads call run() on the same agent instance)
_tlocal = threading.local()

# ── Patch _convert_message_to_dict to carry reasoning_content across turns ──
# DeepSeek API requires reasoning_content from assistant messages be passed back
# in subsequent turns. This patch injects it into the serialized dict.
# Wrapped in try/except: if LangChain changes its internal API, the patch
# silently fails and reasoning_content won't carry across turns (non-fatal).
try:
    _original_convert = oai_base._convert_message_to_dict

    def _patched_convert(message: BaseMessage, api="chat/completions") -> dict:
        msg_dict = _original_convert(message, api)
        if isinstance(message, AIMessage):
            rc = message.additional_kwargs.get("reasoning_content")
            if rc:
                msg_dict["reasoning_content"] = rc
        return msg_dict

    oai_base._convert_message_to_dict = _patched_convert
except AttributeError:
    # langchain_openai changed its internal API — reasoning_content
    # won't carry across turns, but the agent still works.
    import logging
    logging.getLogger(__name__).debug(
        "Could not patch _convert_message_to_dict — "
        "reasoning_content will not carry across ReAct turns"
    )


class DeepSeekChatOpenAI(FixedChatOpenAI):
    """ChatOpenAI wrapper that preserves `reasoning_content` across ReAct turns.

    DeepSeek's API requires that `reasoning_content` from assistant messages
    be passed back in subsequent turns. This wrapper captures
    `reasoning_content` from the raw API response and injects it into
    AIMessage.additional_kwargs, which is then serialized by the patched
    _convert_message_to_dict above.
    """

    def _create_chat_result(self, response: Any, generation_info: Optional[Dict] = None) -> Any:
        result = super()._create_chat_result(response, generation_info)
        if result and result.generations:
            choice = response.choices[0]
            msg = choice.message
            rc = getattr(msg, "reasoning_content", None)
            if rc:
                for gen_list in result.generations:
                    for gen in gen_list:
                        gen.message.additional_kwargs["reasoning_content"] = rc
        return result


def _build_system_prompt(max_tokens: int = 200) -> str:
    """Build species-aware system prompt."""
    handler = Config.get_species_handler()

    perf_desc = handler.performance_tool_description()
    nr_desc = handler.nutrient_tool_description()

    return f"""You are NutriCHAT (Nutrition Conversational Helper for Avian Treatment), a domain-specific AI assistant for {handler.display_name} nutrition. You have access to expert-designed tools for accurate nutritional information.

Current species: {handler.display_name}
Breeds available: {handler.breeds}
Production phases: {handler.phases}
Key metrics: {handler.metrics}

CRITICAL RULES:
1. ALWAYS use the appropriate tool to retrieve nutritional data rather than relying on your own knowledge
2. For numeric queries, retrieve precise values from the tools
3. Provide specific numbers with appropriate units
4. If you need multiple pieces of information, use tools one at a time
5. Keep responses concise and under {max_tokens} tokens when possible
6. Synthesize tool outputs into clear, actionable answers

Available tools:
- feed_ingredient_bank: Get nutritional composition data for feed ingredients (CP, ME, amino acids, minerals, digestibility)
- performance_management: {perf_desc}
- nutrient_requirements: {nr_desc}
- definitions: Get definitions of poultry nutrition terms and concepts

Answer the user's poultry nutrition question using these tools as needed."""


class ReActAgent:
    """ReAct reasoning agent using LangChain's create_agent (LangGraph-based).

    Implements Thought-Action-Observation cycle where:
    1. Model determines which tool to call (Thought → Action)
    2. Tool executes and returns results (Observation)
    3. Model synthesizes final answer or continues the cycle
    """

    def __init__(
        self,
        tools: List[BaseTool],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        max_iterations: int = None,
        verbose: bool = False,
        base_url: str = None,
        api_key: str = None,
        extra_body: dict = None,
        system_prompt: str = None,
        fallback_fn=None,
    ):
        self.tools = tools
        self.max_iterations = max_iterations
        self._custom_fallback = fallback_fn
        llm_kwargs = dict(
            model=model or Config.LLM_MODEL,
            temperature=temperature if temperature is not None else Config.TEMPERATURE,
            max_tokens=max_tokens or Config.MAX_TOKENS,
            openai_api_key=api_key or Config.LLM_API_KEY,
            openai_api_base=base_url or Config.LLM_BASE_URL,
            timeout=Config.API_TIMEOUT,
        )
        eb = extra_body or Config.LLM_EXTRA_BODY
        if eb:
            llm_kwargs["extra_body"] = eb
            self.llm = DeepSeekChatOpenAI(**llm_kwargs)
        else:
            self.llm = FixedChatOpenAI(**llm_kwargs)
        self.max_tokens = max_tokens or Config.MAX_TOKENS
        self.verbose = verbose

        system = system_prompt or _build_system_prompt(max_tokens=self.max_tokens)

        # Build agent graph using the new LangChain 1.3 API
        self.graph = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system,
            name="nutrichat_agent",
        )
        # Re-inject extra_body — create_agent creates a new model instance internally
        if Config.LLM_EXTRA_BODY:
            try:
                for node in self.graph.nodes.values():
                    bound = getattr(node, "bound", None) or getattr(node, "runnable", None)
                    if bound is not None and hasattr(bound, "extra_body"):
                        bound.extra_body = Config.LLM_EXTRA_BODY
                        break
            except Exception:
                pass  # LangGraph API changed — extra_body already set on self.llm

    @property
    def last_usage(self) -> dict:
        """Token usage from the last run() in the current thread."""
        pt = getattr(_tlocal, "last_prompt", 0)
        ct = getattr(_tlocal, "last_completion", 0)
        return {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
            "model": Config.LLM_MODEL,
        }

    def run(self, query: str) -> Tuple[str, List[dict]]:
        """Execute the ReAct loop for a single query.

        Uses a single graph.stream() call (stream_mode="updates") to
        capture both intermediate steps and the final response,
        avoiding redundant API calls.

        Returns:
            Tuple of (final_answer, intermediate_steps)
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"ReAct Agent processing: {query}")
            print(f"{'='*60}")

        inputs = {"messages": [HumanMessage(content=query)]}
        steps = []
        response = "No response generated."
        prompt_tokens = 0
        completion_tokens = 0

        # Recursion limit: prevent infinite tool-call loops.
        # Each tool call consumes ~2 graph nodes (agent→tool), so
        # max_iterations tool calls → recursion_limit = max_iterations * 2 + 1.
        _max_iter = self.max_iterations or 12
        _stream_config = {"recursion_limit": _max_iter * 2 + 1}

        try:
            # Single stream — captures both intermediate steps and token usage
            for chunk in self.graph.stream(inputs, stream_mode="updates", config=_stream_config):
                for node_name, node_data in chunk.items():
                    msgs = node_data.get("messages", [])

                    # Track token usage from AI messages
                    for msg in msgs:
                        um = getattr(msg, "usage_metadata", None)
                        if um:
                            prompt_tokens += um.get("input_tokens", 0)
                            completion_tokens += um.get("output_tokens", 0)
                            if self.verbose:
                                print(f"  [tokens] {msg.type}: {um}")

                    for msg in msgs:
                        tc_attr = getattr(msg, "tool_calls", None)

                        if tc_attr:
                            # AI message with tool calls
                            for tc in tc_attr:
                                # LangChain ToolCall objects use attributes;
                                # dicts use .get(). Support both.
                                tc_name = tc.name if hasattr(tc, "name") else tc.get("name", "unknown")
                                tc_args = tc.args if hasattr(tc, "args") else tc.get("args", {})
                                steps.append({
                                    "thought": msg.content or "Calling tool...",
                                    "action": tc_name,
                                    "action_input": str(tc_args),
                                    "observation": "",
                                })
                        elif msg.type == "ai" and msg.content:
                            # Final AI response (no tool calls)
                            response = msg.content
                        elif msg.type == "tool":
                            # Tool observation — attach to last step
                            if steps:
                                steps[-1]["observation"] = msg.content[:500]

        except Exception as e:
            error_str = str(e)
            if self.verbose:
                print(f"  Agent error: {error_str}")
            # For rate limit errors (TPM/RPM), retry with backoff
            if "429" in error_str or "rate" in error_str.lower():
                import time as _t
                for _retry_wait in [15, 30]:
                    _t.sleep(_retry_wait)
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
                        break  # success, exit retry loop
                    except Exception as e2:
                        if "429" not in str(e2) and "rate" not in str(e2).lower():
                            response = self._fallback_response(query, f"{error_str} (retry: {e2})")
                            break
                        # still 429, continue to next retry
                else:
                    response = self._fallback_response(query, error_str)
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
                except Exception as e2:
                    if self.verbose:
                        print(f"  Retry also failed: {e2}")
                    response = self._fallback_response(query, f"{error_str} (retry: {e2})")
            else:
                response = self._fallback_response(query, error_str)

        _tlocal.last_prompt = prompt_tokens
        _tlocal.last_completion = completion_tokens

        if self.verbose:
            print(f"\nFinal: {response[:200]}...")
            print(f"Token usage: {prompt_tokens}↑ + {completion_tokens}↓ = {prompt_tokens + completion_tokens}")

        return response, steps

    def _fallback_response(self, query: str, error: str = "") -> str:
        """Generate a fallback response when the agent encounters an error."""
        # Use custom fallback if provided (e.g., v1 KG tools)
        if self._custom_fallback:
            return self._custom_fallback(query, error)
        # Try to provide a useful response by directly invoking tools
        for tool in self.tools:
            try:
                if "feed_ingredient_bank" in tool.name.lower():
                    for ing in ["corn", "wheat", "soybean", "fish", "limestone"]:
                        if ing in query.lower():
                            result = tool._run(ing)
                            return f"[Tool-aided response]\n{result}"
            except Exception:
                continue
        return f"NutriCHAT is ready to answer: '{query}'. Set SILICONFLOW_API_KEY in .env to enable the ReAct agent."
