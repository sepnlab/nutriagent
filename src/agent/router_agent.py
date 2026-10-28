"""Router Agent — classifies user queries to target specialist agents."""

import json
import re
from langchain_openai import ChatOpenAI

from src.config import Config
from src.agent.base_agent import RouterDecision, FixedChatOpenAI

_ROUTER_PROMPT = """You are a query router for NutriCHAT, a poultry nutrition AI system.
Classify the user's query to determine which specialist agent(s) should handle it.

Available specialist agents:

1. feed: Questions about feed ingredient nutritional composition, digestibility, and raw material properties.
   Examples: "What is the crude protein content of Soybean Meal 48%?", "Calcium content of limestone",
   "Compare energy levels of corn and wheat", "Digestibility of fish meal"

2. nutrition: Questions about nutrient requirements, diet specifications, vitamin/mineral levels, and feeding phases.
   Examples: "What is the calcium requirement for Ross 308 starter?", "SID lysine for Hy-Line W-36",
   "ME requirements for Jingfen 1", "What are the amino acid specs for layer phase 2?"

3. performance: Questions about production performance metrics: body weight, ADG, FCR, egg production, egg quality.
   Examples: "Body weight of Cobb 500 at 35 days", "Peak egg production for ISA Brown",
   "FCR of Shengze 901", "Compare Ross 308 and Cobb 500 growth"

4. definitions: Questions about poultry nutrition terminology and concept definitions.
   Examples: "What is FCR?", "Define standardized ileal digestibility", "What is Haugh unit?"

Query: {query}

Output ONLY a JSON object with these fields:
- "intent": one of "feed", "nutrition", "performance", "definitions", or "multi"
- "target_agents": list of agent names to invoke (e.g., ["feed"], ["nutrition", "performance"])
- "requires_cross_validation": true only if multiple agents are needed AND their answers may conflict

Examples:
{{"intent": "feed", "target_agents": ["feed"], "requires_cross_validation": false}}
{{"intent": "multi", "target_agents": ["nutrition", "feed"], "requires_cross_validation": true}}
{{"intent": "performance", "target_agents": ["performance"], "requires_cross_validation": false}}

JSON:"""


class RouterAgent:
    """Classifies queries into routing decisions for the multi-agent system."""

    def __init__(self):
        kwargs = dict(
            model=Config.LLM_MODEL,
            temperature=Config.ROUTER_TEMPERATURE,
            max_tokens=Config.ROUTER_MAX_TOKENS,
            openai_api_key=Config.LLM_API_KEY,
            openai_api_base=Config.LLM_BASE_URL,
            timeout=Config.API_TIMEOUT,
        )
        if Config.LLM_EXTRA_BODY:
            kwargs["extra_body"] = Config.LLM_EXTRA_BODY
        self.llm = FixedChatOpenAI(**kwargs)
        self._last_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def route(self, query: str) -> RouterDecision:
        """Classify query and return routing decision.

        Uses structured LLM classification with fallback patterns.
        """
        if not query or not query.strip():
            return RouterDecision(
                query=query, intent="definitions",
                target_agents=["definitions"],
            )

        try:
            prompt = _ROUTER_PROMPT.format(query=query[:500])
            resp = self.llm.invoke(prompt)
            usage = getattr(resp, "usage_metadata", None) or {}
            self._last_usage = {
                "prompt_tokens": int(usage.get("input_tokens", 0) or 0),
                "completion_tokens": int(usage.get("output_tokens", 0) or 0),
            }
            text = resp.content.strip()

            # Extract JSON from markdown code block if present
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(text)

            intent = data.get("intent", "multi")
            target = data.get("target_agents", [intent] if intent != "multi" else ["feed"])
            cross = data.get("requires_cross_validation", len(target) > 1)

            return RouterDecision(
                query=query,
                intent=intent,
                target_agents=target,
                requires_cross_validation=cross,
            )

        except Exception:
            # Fallback: keyword-based routing
            return self._keyword_fallback(query)

    def _keyword_fallback(self, query: str) -> RouterDecision:
        """Keyword-based routing when LLM classification fails."""
        q = query.lower()

        # Domain keyword sets
        feed_keywords = [
            "protein", "crude protein", "cp", "amino acid", "lysine", "methionine",
            "energy", "metabolizable energy", "calcium", "phosphorus",
            "ingredient", "corn", "wheat", "soybean", "fish meal", "limestone",
            "digestibility", "sidx", "nutritional composition", "fat", "fiber",
            "原料", "成分", "蛋白质", "能量", "钙", "磷", "氨基酸",
            "消化率", "玉米", "豆粕", "石粉",
        ]
        nutrition_keywords = [
            "requirement", "nutrient requirement", "specification", "phase",
            "starter", "grower", "finisher", "withdrawal", "pre-lay", "lay i",
            "diet", "formulation", "feed formula", "nrc",
            "需要量", "营养需要", "阶段", "配方", "推荐量",
        ]
        performance_keywords = [
            "body weight", "bw", "adg", "daily gain", "fcr", "feed conversion",
            "growth", "performance", "egg production", "egg weight", "hen-day",
            "mortality", "yield", "breast meat",
            "体重", "日增重", "料重比", "生产性能", "产蛋率", "蛋重",
        ]
        definitions_keywords = [
            "what is", "define", "definition", "meaning of", "explain",
            "什么是", "定义", "含义", "解释",
        ]

        scores = {
            "feed": sum(1 for kw in feed_keywords if kw in q),
            "nutrition": sum(1 for kw in nutrition_keywords if kw in q),
            "performance": sum(1 for kw in performance_keywords if kw in q),
            "definitions": sum(1 for kw in definitions_keywords if kw in q),
        }

        # Find top scoring domains
        max_score = max(scores.values())
        if max_score == 0:
            return RouterDecision(
                query=query, intent="multi",
                target_agents=["feed", "nutrition", "performance", "definitions"],
                requires_cross_validation=True,
            )

        top_domains = [k for k, v in scores.items() if v >= max_score * 0.7]
        is_multi = len(top_domains) > 1

        return RouterDecision(
            query=query,
            intent="multi" if is_multi else top_domains[0],
            target_agents=top_domains,
            requires_cross_validation=is_multi,
        )
